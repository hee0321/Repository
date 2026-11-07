"""CPU MuJoCo: 머리 카메라 RGB(HSV)로 빨간 공을 찾아 걸어간다.

제어기(camera_seek.py)는 검출 픽셀과 넘어짐 여부만 본다.
공의 시뮬레이터 좌표는 공을 처음에 놓는 일과, 끝난 뒤 거리를 재는 일에만 쓴다.

  MUJOCO_GL=osmesa python run_camera_ball.py --probe
  MUJOCO_GL=osmesa python run_camera_ball.py --video /opt/cursor/artifacts/microduck_camera_ball.mp4

체크포인트는 공개 Pollen velstand v8 을 익명으로 읽는다. 저장소에 .pt 를 넣지 않는다.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from camera_seek import CameraSeek  # noqa: E402
from camera_vision import detect_red, pixel_bearing  # noqa: E402
from duck_sim import Cmd, DuckSim, HIDE  # noqa: E402

V8 = "hf://pollen-robotics/microduck-training/bundles/velstand/0045i2si_2749/model_2749.pt"
# 공개 ball.xml 은 주황(1 0.55 0)이다. 발·턱도 주황이라, 이 실험의 공만 빨강으로 칠한다.
RED = (0.95, 0.02, 0.04, 1.0)
CAM_W, CAM_H = 320, 180
# 렌더가 옆으로 누우면 여기만 바꾼다. 0, 1, 2, 3 = np.rot90 k. 프로브 영상으로 확정.
ORIENT_K = 0
FONT = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
# 평가 기준 (제어 입력이 아님). 몸통 수평 거리.
SUCCESS_DIST_M = 0.40

# bearing_deg > 0 은 덕의 왼쪽. yaw 는 시작 몸 방향(rad).
TRIALS = [
    {"name": "left_90_1.0m", "bearing_deg": 90.0, "dist": 1.0, "yaw": 0.0, "video": True},
    {"name": "right_90_1.0m", "bearing_deg": -90.0, "dist": 1.0, "yaw": 0.6},
    {"name": "left_150_1.1m", "bearing_deg": 150.0, "dist": 1.1, "yaw": -0.4},
    {"name": "right_120_0.8m", "bearing_deg": -120.0, "dist": 0.8, "yaw": 1.2},
    {"name": "left_70_1.4m", "bearing_deg": 70.0, "dist": 1.4, "yaw": -1.0},
    {"name": "no_target", "bearing_deg": None, "dist": None, "yaw": 0.3},
]


def resolve_ckpt(path: str) -> str:
    if not path.startswith("hf://"):
        return path
    from huggingface_hub import hf_hub_download
    org, repo, *rest = path[5:].split("/")
    return hf_hub_download(f"{org}/{repo}", "/".join(rest))


def orient(rgb: np.ndarray) -> np.ndarray:
    if ORIENT_K % 4 == 0:
        return rgb
    return np.ascontiguousarray(np.rot90(rgb, ORIENT_K))


def open_head_camera(sim: DuckSim):
    import mujoco
    sim.sync_cpu()
    names = [mujoco.mj_id2name(sim.mjm, mujoco.mjtObj.mjOBJ_CAMERA, i) or "" for i in range(sim.mjm.ncam)]
    cid = -1
    for i, n in enumerate(names):
        if n.endswith("head_camera"):
            cid = i
            break
    if cid < 0:
        raise SystemExit(f"head_camera 가 없음. 카메라: {names}")
    # MJCF 카메라는 뒤를 보고, 그대로 돌리면 영상이 옆으로 눕는다 (가로 화각이 좁아짐).
    # 2026-11-07 측정: x축 180°(정면을 봄) 다음 z축 +90°(하늘이 위로) 이면
    # 카메라 up ≈ 세계 +z, 정면 공이 화면 가운데에 온다.
    def qmul(a, b):
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
        return out

    def qaxis(axis, deg):
        half = math.radians(deg) / 2.0
        ax = np.asarray(axis, float)
        ax = ax / np.linalg.norm(ax)
        s = math.sin(half)
        return np.array([math.cos(half), ax[0] * s, ax[1] * s, ax[2] * s])

    q = qmul(sim.mjm.cam_quat[cid], qaxis((1.0, 0.0, 0.0), 180.0))
    q = qmul(q, qaxis((0.0, 0.0, 1.0), 90.0))
    sim.mjm.cam_quat[cid] = q
    renderer = mujoco.Renderer(sim.mjm, height=CAM_H, width=CAM_W)
    return renderer, cid, names


def annotate(sim: DuckSim, cid: int, det):
    """검출에 몸통 기준 방향과 픽셀 지름을 붙인다. 공 좌표는 쓰지 않는다."""
    if det is None:
        return None
    det = dict(det)
    _, yaw, _ = sim.pose()
    det["bearing"] = pixel_bearing(
        det["cx"], det["cy"], sim.mjd.cam_xmat[cid], float(sim.mjm.cam_fovy[cid]), CAM_W, CAM_H, yaw,
    )
    return det


def render_head(sim: DuckSim, renderer, cid: int) -> np.ndarray:
    sim.sync_cpu()
    renderer.update_scene(sim.mjd, camera=cid)
    return orient(renderer.render())


def reset_robot(sim: DuckSim, yaw: float) -> None:
    half = yaw / 2.0
    q = (math.cos(half), 0.0, 0.0, math.sin(half))
    st = torch.zeros(1, 13)
    st[0, 2] = 0.13
    st[0, 3:7] = torch.tensor(q, dtype=torch.float32)
    sim.robot.write_root_state_to_sim(st)
    dj = sim.robot.data.default_joint_pos.clone()
    sim.robot.write_joint_state_to_sim(dj, torch.zeros_like(dj))
    sim.ball_v[:] = 0.0
    sim.set_ball(HIDE, (0.0, 0.0))
    sim.t = 0.0
    for _ in range(int(0.8 / sim.dt)):
        sim.step(Cmd())
    sim.t = 0.0


def place_relative(sim: DuckSim, bearing: float, dist: float) -> np.ndarray:
    """시험 준비: 덕이 선 방향 기준 방위·거리에 공을 놓는다. 제어기가 이 함수를 부르지 않는다."""
    p, yaw, _ = sim.pose()
    xy = np.array([p[0] + dist * math.cos(yaw + bearing), p[1] + dist * math.sin(yaw + bearing)])
    sim.ball_v[:] = 0.0
    sim.set_ball(xy, (0.0, 0.0))
    return xy


def gt_distance(sim: DuckSim) -> float:
    """평가 전용. 제어기로 넘기지 않는다."""
    p, _, _ = sim.pose()
    return float(np.hypot(sim.ball_xy[0] - p[0], sim.ball_xy[1] - p[1]))


def _font(size):
    from PIL import ImageFont
    return ImageFont.truetype(FONT, size)


def compose(external, head, det, said, detail, status, eval_line):
    from PIL import Image, ImageDraw
    ext = Image.fromarray(external).convert("RGBA")
    inset = Image.fromarray(head).convert("RGBA")
    if det is not None:
        d = ImageDraw.Draw(inset)
        x0, y0, x1, y1 = det["bbox"]
        d.rectangle((x0, y0, x1, y1), outline=(255, 40, 40, 255), width=2)
        cx, cy = det["cx"], det["cy"]
        d.line((cx - 6, cy, cx + 6, cy), fill=(255, 255, 0, 255), width=2)
        d.line((cx, cy - 6, cx, cy + 6), fill=(255, 255, 0, 255), width=2)
    scale = 240 / inset.width
    inset = inset.resize((240, int(inset.height * scale)))
    ext.paste(inset, (ext.width - inset.width - 16, 16))
    ov = Image.new("RGBA", ext.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    big, mid, small = _font(36), _font(22), _font(18)
    lines = [f"“{said}”", detail]
    box_w = max(dr.textlength(t, font=big if i == 0 else mid) for i, t in enumerate(lines)) + 36
    dr.rounded_rectangle((16, 16, 16 + box_w, 118), 14, fill=(255, 255, 255, 230))
    dr.text((32, 24), lines[0], font=big, fill=(20, 20, 20, 255))
    dr.text((32, 74), lines[1], font=mid, fill=(180, 50, 20, 255))
    dr.text((ext.width - inset.width - 8, 16 + inset.height + 6), "머리 카메라 · HSV", font=small, fill=(255, 255, 255, 255))
    tag = "HSV 색깔 검출 · v8 보행 재사용 · 새 학습 없음"
    tw = dr.textlength(tag, font=small)
    dr.rounded_rectangle((ext.width - tw - 28, ext.height - 44, ext.width - 12, ext.height - 12), 8, fill=(0, 0, 0, 150))
    dr.text((ext.width - tw - 20, ext.height - 38), tag, font=small, fill=(255, 255, 255, 255))
    sw = dr.textlength(status, font=small)
    dr.rounded_rectangle((14, ext.height - 78, 28 + sw, ext.height - 48), 8, fill=(0, 0, 0, 150))
    dr.text((22, ext.height - 72), status, font=small, fill=(255, 255, 255, 255))
    ew = dr.textlength(eval_line, font=small)
    dr.rounded_rectangle((14, ext.height - 44, 28 + ew, ext.height - 12), 8, fill=(20, 60, 120, 170))
    dr.text((22, ext.height - 38), eval_line, font=small, fill=(255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(ext, ov).convert("RGB"))


def aim_external(sim: DuckSim) -> None:
    cam = sim.env._offline_renderer._cam
    p, yaw, _ = sim.pose()
    cam.lookat[:] = (float(p[0]), float(p[1]), 0.08)
    cam.distance = 1.45
    cam.elevation = -26.0
    cam.azimuth = math.degrees(yaw) + 150.0


def run_trial(sim, renderer, cid, trial, video_path=None, shot_path=None) -> dict:
    import imageio.v2 as imageio
    reset_robot(sim, float(trial.get("yaw") or 0.0))
    has_ball = trial["bearing_deg"] is not None
    if has_ball:
        place_relative(sim, math.radians(trial["bearing_deg"]), float(trial["dist"]))
    else:
        sim.set_ball((30.0, -30.0), (0.0, 0.0))
    # 움직이기 전 첫 화면. 공 시험은 여기서 검출되면 처음부터 화면 안.
    rgb0 = render_head(sim, renderer, cid)
    det0 = annotate(sim, cid, detect_red(rgb0))
    seek = CameraSeek()
    p0 = sim.pose()[0].copy()
    cam_every = max(1, int(round(0.1 / sim.dt)))
    vid_every = max(1, int(round((1.0 / 12.0) / sim.dt)))
    max_s = 32.0 if (has_ball and abs(trial["bearing_deg"]) <= 100) else 48.0
    n = int(max_s / sim.dt)
    writer = None
    if video_path:
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        writer = imageio.get_writer(
            video_path, fps=max(1, int(round(1.0 / (sim.dt * vid_every)))),
            codec="libx264", quality=None, macro_block_size=8,
            output_params=["-crf", "26", "-preset", "veryfast", "-pix_fmt", "yuv420p"],
        )
    dists, upright_n, steps = [], 0, 0
    modes = []
    first_detect_s = None
    shot_saved = False
    cmd = Cmd()
    t_stop = None
    wall0 = time.time()
    rgb, det = rgb0, det0
    for k in range(n):
        if k % cam_every == 0:
            rgb = render_head(sim, renderer, cid)
            det = annotate(sim, cid, detect_red(rgb))
            if det is not None and first_detect_s is None and k > 0:
                first_detect_s = round(k * sim.dt, 2)
            cmd = seek.update(det, sim.upright(), sim.dt * cam_every)
        sim.step(cmd)
        steps += 1
        up = sim.upright()
        upright_n += int(up)
        if has_ball:
            dists.append(gt_distance(sim))
        if seek.mode not in modes:
            modes.append(seek.mode)
        if seek.mode == "stop":
            t_stop = sim.t if t_stop is None else t_stop
            if sim.t - t_stop > 1.0:
                break
        elif seek.mode == "giveup":
            break
        if writer is not None and k % vid_every == 0:
            aim_external(sim)
            snap = seek.snapshot()
            p, _, gz = sim.pose()
            v = sim.robot.data.root_link_lin_vel_b[0].numpy()
            status = f"{k * sim.dt:4.1f}s · 명령 vx {cmd.vx:+.2f} wz {cmd.wz:+.2f} · 실제 vx {float(v[0]):+.2f}"
            if det is None:
                status += " · 화면: 빨강 없음"
            else:
                status += f" · 덩어리 지름 {det['diam']:.0f}px · 보는 방향 {math.degrees(det['bearing']):+.0f}°"
            if has_ball:
                eval_line = f"평가용(제어에 안 씀) 실제 거리 {gt_distance(sim):.2f} m"
            else:
                eval_line = "평가용: 공 없음"
            if not up:
                eval_line += " · 넘어짐"
            frame = compose(sim.render(), rgb, det, snap.said, snap.detail, status, eval_line)
            writer.append_data(frame)
            if shot_path and not shot_saved and seek.mode == "stop" and det is not None:
                os.makedirs(os.path.dirname(shot_path) or ".", exist_ok=True)
                imageio.imwrite(shot_path, frame)
                shot_saved = True
        if k % int(2.0 / sim.dt) == 0:
            dist_s = f"{dists[-1]:.2f}m" if dists else "-"
            vis = "-" if det is None else f"{math.degrees(det['bearing']):+.0f}° d{det['diam']:.0f}"
            print(f"  t={k * sim.dt:5.1f}s mode={seek.mode} gt={dist_s} vis={vis} up={up} hyaw={seek.head_yaw:+.2f}", flush=True)
    if writer is not None:
        # 마지막 자세를 한 장 더
        aim_external(sim)
        snap = seek.snapshot()
        if has_ball:
            eval_line = f"평가용(제어에 안 씀) 실제 거리 {gt_distance(sim):.2f} m"
        else:
            eval_line = "평가용: 공 없음"
        v = sim.robot.data.root_link_lin_vel_b[0].numpy()
        status = f"끝 · 실제 vx {float(v[0]):+.2f}"
        writer.append_data(compose(sim.render(), rgb, det, snap.said, snap.detail, status, eval_line))
        writer.close()
    p1 = sim.pose()[0]
    disp = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    out = {
        "name": trial["name"],
        "bearing_deg": trial["bearing_deg"],
        "start_dist_m": trial["dist"],
        "start_yaw_rad": trial.get("yaw", 0.0),
        "offscreen_at_start": det0 is None,
        "first_detect_s": first_detect_s,
        "modes": modes,
        "final_mode": seek.mode,
        "sim_s": round(steps * sim.dt, 2),
        "wall_s": round(time.time() - wall0, 1),
        "upright_frac": round(upright_n / max(steps, 1), 3),
        "upright_end": bool(sim.upright()),
        "displacement_m": round(disp, 3),
        "final_dist_m": None if not dists else round(dists[-1], 3),
        "min_dist_m": None if not dists else round(min(dists), 3),
    }
    if not has_ball:
        out["success"] = (
            seek.mode == "giveup"
            and "approach" not in modes
            and out["upright_end"]
            and disp < 1.0
        )
        out["success_rule"] = "giveup, never approached, stayed upright, displacement < 1 m"
    else:
        out["success"] = (
            seek.mode == "stop"
            and out["final_dist_m"] is not None
            and out["final_dist_m"] <= SUCCESS_DIST_M
            and out["upright_end"]
            and out["offscreen_at_start"]
        )
        out["success_rule"] = (
            f"stop, final base-to-ball <= {SUCCESS_DIST_M} m, upright, offscreen at start"
        )
    if not out["success"]:
        reasons = []
        if has_ball and not out["offscreen_at_start"]:
            reasons.append("ball already in the first camera frame")
        if has_ball and seek.mode != "stop":
            reasons.append(f"ended in {seek.mode}, not stop")
        if has_ball and out["final_dist_m"] is not None and out["final_dist_m"] > SUCCESS_DIST_M:
            reasons.append(f"final distance {out['final_dist_m']} m > {SUCCESS_DIST_M}")
        if not out["upright_end"]:
            reasons.append("fell")
        if not has_ball and seek.mode != "giveup":
            reasons.append(f"ended in {seek.mode}")
        if not has_ball and "approach" in modes:
            reasons.append("approached with no ball (false detection)")
        if not has_ball and disp >= 1.0:
            reasons.append(f"drifted {disp:.2f} m")
        out["failure"] = "; ".join(reasons) or "criteria not met"
    return out


def probe(sim, renderer, cid, out_dir) -> None:
    import imageio.v2 as imageio
    os.makedirs(out_dir, exist_ok=True)
    reset_robot(sim, 0.0)
    fovy = float(sim.mjm.cam_fovy[cid])
    print("camera fovy", round(fovy, 2), "orient_k", ORIENT_K, "dt", sim.dt)
    shots = []

    def save(name, yaw_cmd=None, bearing=None, dist=0.7):
        if bearing is None:
            sim.set_ball(HIDE, (0, 0))
        else:
            place_relative(sim, math.radians(bearing), dist)
        if yaw_cmd is not None:
            for _ in range(int(0.8 / sim.dt)):
                sim.step(Cmd(head=(0.0, 0.3, yaw_cmd, 0.0)))
        rgb = render_head(sim, renderer, cid)
        det = detect_red(rgb)
        imageio.imwrite(os.path.join(out_dir, name + ".png"), rgb)
        red = int((rgb[..., 0].astype(int) > 150).sum()) and None
        print(f"PROBE {name}: det={None if det is None else {k: (round(det[k], 3) if isinstance(det[k], float) else det[k]) for k in ('cx','cy','ex','ey','area','height')}} "
              f"rgb_mean={rgb.mean(axis=(0,1)).round(1).tolist()} maxR={int(rgb[...,0].max())}")
        shots.append(name)

    save("no_ball")
    save("front", bearing=0.0, dist=0.65)
    # 고개만 +0.6 으로 두고 같은 정면 공을 다시 본다. cx 가 줄면 yaw+ 는 화면 오른쪽을 본다.
    save("front_head_plus", yaw_cmd=0.6, bearing=0.0, dist=0.65)
    reset_robot(sim, 0.0)
    save("left90", bearing=90.0, dist=0.9)
    save("right90", bearing=-90.0, dist=0.9)
    reset_robot(sim, 0.0)
    place_relative(sim, math.radians(90.0), 0.9)
    for yaw_cmd in (0.6, 1.2, -1.2):
        for _ in range(int(0.7 / sim.dt)):
            sim.step(Cmd(head=(0.0, 0.35, yaw_cmd, 0.0)))
        rgb = render_head(sim, renderer, cid)
        det = detect_red(rgb)
        imageio.imwrite(os.path.join(out_dir, f"left90_yaw_{yaw_cmd:+.1f}.png"), rgb)
        print(f"PROBE left90 head {yaw_cmd:+.1f}: det_ex={None if det is None else round(det['ex'], 3)} area={None if det is None else det['area']}")
    print("probe images", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=V8)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--trials", default="all", help="all 또는 쉼표로 이름")
    ap.add_argument("--video", default="")
    ap.add_argument("--shot", default="")
    ap.add_argument("--results", default=os.path.join(HERE, "videos", "camera_ball_results.json"))
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    # 제어 코드가 공 좌표를 참조하지 않는지 소스에서 한 번 확인.
    import camera_seek
    import camera_vision
    for mod in (camera_seek, camera_vision):
        src = open(mod.__file__, encoding="utf-8").read()
        for banned in ("ball_xy", "root_link_pos", "segid", "geom_id"):
            if banned in src:
                raise SystemExit(f"{mod.__name__} contains {banned}")
    print("loading", a.ckpt, flush=True)
    sim = DuckSim(
        resolve_ckpt(a.ckpt), width=960, height=544, distance=1.3,
        with_ball=True, with_obstacles=False, with_beak=True, ball_rgba=RED,
    )
    renderer, cid, names = open_head_camera(sim)
    print("cameras", names, "head id", cid, flush=True)
    if a.probe:
        probe(sim, renderer, cid, "/tmp/camprobe")
        return
    want = {s.strip() for s in a.trials.split(",")} if a.trials != "all" else None
    rows = []
    video_done = False
    for trial in TRIALS:
        if want is not None and trial["name"] not in want:
            continue
        video = a.video if (trial.get("video") and a.video and not a.no_video and not video_done) else None
        shot = a.shot if video else None
        print("TRIAL", trial["name"], "video" if video else "novideo", flush=True)
        row = run_trial(sim, renderer, cid, trial, video_path=video, shot_path=shot)
        video_done = video_done or bool(video)
        rows.append(row)
        print("RESULT", json.dumps(row, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.results), exist_ok=True)
    summary = {
        "checkpoint": a.ckpt,
        "detection": "numpy HSV red hue ±12°, saturation≥0.45, largest component. Not a learned vision model.",
        "locomotion": "Pollen velstand v8 policy, commands only. No new RL.",
        "head": "duck_skills_korean/head.py colors and jaw on a model copy. MJCF head_camera is rotated 180° about its x axis (it faces backward) and +90° about its z axis (the raw image is on its side).",
        "success_dist_m": SUCCESS_DIST_M,
        "trials": rows,
        "n_success": sum(1 for r in rows if r["success"]),
        "n_trials": len(rows),
    }
    with open(a.results, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("WROTE", a.results)


if __name__ == "__main__":
    main()
