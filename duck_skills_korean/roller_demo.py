"""롤러스케이트 덕 한국어 명령 시연 (CPU, 학습 없음): Pollen 공개 roller.onnx + roller_crouch.onnx, 새 머리.

  cd ../../microduck_rl && .venv/bin/python ../hee-repo/duck_skills_korean/roller_demo.py   (영상 ≈ 4–5분 CPU)
  --no-video : 측정만
방향 보정: cmd[2] = clip(0.5 × (목표 방향 − 지금 방향), ±0.5). 공개 roller.onnx 는 cmd[2]=0 으로 학습됐지만
  시험해 보니 + 쪽으로 넣으면 그쪽으로 돌아서 (gain 0.5: 10초 직진 중 방향 흔들림 ±13° → ±4°) 보정용으로 씀.
"""
from __future__ import annotations
import os, sys, math, json, argparse, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import imageio.v2 as imageio
from roller_sim import RollerSim, CROUCH_PERIOD, CROUCH_LEN
from roller_commands import parse_roller
from run_demo import overlay

SCRIPT = [
    (0.0, "덕아, 롤러 타"),
    (7.0, "더 빨리"),
    (13.0, "천천히"),
    (18.0, "왼쪽으로"),
    (25.0, "오른쪽으로"),
    (32.0, "웅크려"),
    (37.0, "앞으로 밀어"),
    (43.0, "멈춰"),
    (48.0, "덕아, 미끄러져"),
]
END = 52.0
GAIN, CLIP = 0.5, 0.5
TAG = "롤러 덕 · Pollen 공개 정책 (학습 없음)"


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "videos", "demo_roller_skating.mp4"))
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    s = RollerSim(render=not a.no_video)
    s.step("roller", (0.0, 0.0, 0.0))
    cam = s.env._offline_renderer._cam if not a.no_video else None
    for _ in range(int(1.0 / s.dt)):   # 1초 안정화 (서 있기)
        s.step("roller", (0.0, 0.0, 0.0))
    target = s.pose()[1]
    w = None if a.no_video else imageio.get_writer(a.out, fps=25, codec="libx264", quality=None, macro_block_size=8,
                                                   output_params=["-crf", "27", "-pix_fmt", "yuv420p"])
    every = int(round(1.0 / (s.dt * 25)))
    idx, rc, said, t_cmd = -1, None, "", 0.0
    segs, seg = [], None
    braked = False
    t0w = time.time()
    p_start = s.pose()[0]
    for k in range(int(END / s.dt)):
        t = k * s.dt
        if idx + 1 < len(SCRIPT) and t >= SCRIPT[idx + 1][0]:
            idx += 1; said = SCRIPT[idx][1]; rc = parse_roller(said); t_cmd = t; braked = False
            if rc.kind == "turn":
                target = wrap(target + rc.turn)
            elif rc.kind == "straight":
                target = s.pose()[1]
            if seg: segs.append(seg)
            seg = dict(t0=t, said=said, kind=rc.kind, vx=[], up=0, n=0, z_min=9.0, yaw_err=[], gz_max=-1.0)
        p, yaw, gz = s.pose()
        err = wrap(target - yaw)
        heading = float(np.clip(GAIN * err, -CLIP, CLIP))
        policy, event = "roller", None
        if rc is None:
            cmd = (0.0, 0.0, 0.0)
        elif rc.kind == "crouch" and t - t_cmd < CROUCH_LEN:
            ph = (t - t_cmd) / CROUCH_PERIOD
            policy, cmd = "roller_crouch", (math.cos(2 * math.pi * ph), math.sin(2 * math.pi * ph), 0.0)
            event = "정책 바꿈 → roller_crouch.onnx (웅크리기)"
        elif rc.kind == "crouch":
            cmd = (0.0, 0.0, heading); event = "끝 → roller.onnx 로 복귀 (미끄러지기)"
        elif rc.kind == "brake":
            v = s.vel()[0][0]
            if abs(v) < 0.03 and t - t_cmd > 0.5:
                braked = True
            cmd = (0.0 if braked else rc.cmd_x, 0.0, heading)
            event = "거의 섰어요 → cmd_x 0" if braked else "브레이크 중"
        else:
            cmd = (rc.cmd_x, 0.0, heading)
        s.step(policy, cmd)
        v, wv = s.vel()
        if seg:
            seg["vx"].append(float(v[0])); seg["n"] += 1; seg["up"] += int(gz < -0.8)
            seg["z_min"] = min(seg["z_min"], float(p[2])); seg["gz_max"] = max(seg["gz_max"], gz)
            seg["yaw_err"].append(abs(math.degrees(err)))
        if w is not None and k % every == 0:
            if cam is not None:
                cam.distance, cam.elevation = 0.62, -12.0
                cam.azimuth = math.degrees(yaw) + 180 - 55   # 앞옆에서 따라가며 (얼굴·바퀴 보이게)
            pose_s = "서 있음" if gz < -0.8 else "기울어짐"
            status = f"{t:4.1f}s · {v[0]:+.2f} m/s · 높이 {p[2]:.3f} m · 방향 {math.degrees(err):+.0f}° · {pose_s}"
            label = rc.label if rc else "대기"
            w.append_data(overlay(s.render(), said, label, status, event, tag=TAG))
        if k % 250 == 0:
            print(f"[{time.time()-t0w:5.0f}s] t={t:5.1f} {said} vx={v[0]:+.2f} z={p[2]:.3f} gz={gz:+.2f} err={math.degrees(err):+.0f}", flush=True)
    segs.append(seg)
    if w is not None:
        w.close()
    out = []
    for sg in segs:
        vx = np.array(sg["vx"]); half = vx[len(vx) // 2:]
        out.append(dict(t0=sg["t0"], said=sg["said"], kind=sg["kind"], mean_vx=round(float(vx.mean()), 3),
                        mean_vx_2nd_half=round(float(half.mean()), 3), max_vx=round(float(vx.max()), 3), end_vx=round(float(vx[-1]), 3),
                        upright_frac=round(sg["up"] / max(sg["n"], 1), 3), z_min=round(sg["z_min"], 3), max_tilt_gz=round(sg["gz_max"], 2),
                        heading_err_mean_deg=round(float(np.mean(sg["yaw_err"])), 1), heading_err_end_deg=round(float(sg["yaw_err"][-1]), 1)))
    p_end = s.pose()[0]
    summary = dict(total_path_m=round(float(np.linalg.norm(p_end[:2] - p_start[:2])), 2), segments=out)
    js = os.path.splitext(a.out)[0] + "_stats.json"
    json.dump(summary, open(js, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
