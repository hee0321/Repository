"""한국어 명령 대본 → 시뮬레이션 → 자막 달린 MP4.

  python run_demo.py                       # 기본 대본(약 60초) → videos/demo_korean_commands.mp4 + 행동별 클립
  python run_demo.py --script my.txt       # 줄마다 "시작초  한국어 명령"  (예: "0  앞으로 가")
  python run_demo.py --ckpt model_xxx.pt   # 다른 체크포인트
"""
from __future__ import annotations
import argparse, json, math, os, subprocess, sys, time
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from duck_sim import DuckSim, Cmd  # noqa: E402
from korean_commands import parse  # noqa: E402
import behaviors  # noqa: E402

V8 = os.path.expanduser("~/.cache/huggingface/hub/models--pollen-robotics--microduck-training/snapshots/"
                        "64d28ce092253c509b51031ec7ccb53790a00d0d/bundles/velstand/0045i2si_2749/model_2749.pt")
FONT = "/usr/share/fonts/truetype/sand-box/google/Nanum Gothic/NanumGothic-Bold.ttf"

# (시작 초, 말, 클립 이름)
DEFAULT_SCRIPT = [
    (0.0, "덕아, 앞으로 가", "walk"),
    (6.0, "왼쪽으로 돌아", "walk"),
    (10.0, "오른쪽으로 돌아", "walk"),
    (14.0, "멈춰", "walk"),
    (16.0, "고개 왼쪽", "head"),
    (18.0, "고개 오른쪽", "head"),
    (20.0, "고개 들어 위를 봐", "head"),
    (22.0, "고개 숙여 아래 봐", "head"),
    (24.0, "춤춰!", "dance"),
    (38.0, "멈춰", "dance"),
    (39.5, "일어나!", "getup"),       # 이 순간 시뮬레이터가 덕을 엎어 놓음
    (43.5, "공 따라가", "ball"),
    (64.0, "멈춰", "ball"),
]
END = 66.0
OUT_FPS = 25


def font(sz):
    return ImageFont.truetype(FONT, sz)


F_BIG, F_MID, F_SMALL = font(40), font(22), font(18)


def overlay(frame, said, label, status, extra=None):
    im = Image.fromarray(frame).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    if said:
        txt = f"“{said}”"
        w = max(d.textlength(txt, font=F_BIG), d.textlength(label, font=F_MID)) + 40
        d.rounded_rectangle((16, 16, 16 + w, 122), 14, fill=(255, 255, 255, 225))
        d.text((34, 26), txt, font=F_BIG, fill=(20, 20, 20, 255))
        d.text((36, 82), "→ " + label, font=F_MID, fill=(200, 80, 0, 255))
    tag = "MuJoCo 시뮬레이션 · v8 정책 (추가 학습 없음)"
    tw = d.textlength(tag, font=F_SMALL)
    d.rounded_rectangle((im.width - tw - 34, im.height - 46, im.width - 14, im.height - 14), 8, fill=(0, 0, 0, 140))
    d.text((im.width - tw - 24, im.height - 41), tag, font=F_SMALL, fill=(255, 255, 255, 255))
    if extra:
        ew = d.textlength(extra, font=F_MID)
        d.rounded_rectangle((16, 132, 40 + ew, 168), 10, fill=(200, 40, 40, 210))
        d.text((28, 137), extra, font=F_MID, fill=(255, 255, 255, 255))
    sw = d.textlength(status, font=F_SMALL)
    d.rounded_rectangle((14, im.height - 46, 34 + sw, im.height - 14), 8, fill=(0, 0, 0, 140))
    d.text((24, im.height - 41), status, font=F_SMALL, fill=(255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=V8)
    ap.add_argument("--script")
    ap.add_argument("--end", type=float, default=END)
    ap.add_argument("--out", default=os.path.join(HERE, "videos", "demo_korean_commands.mp4"))
    ap.add_argument("--no-clips", action="store_true")
    a = ap.parse_args()
    script = DEFAULT_SCRIPT
    if a.script:
        script = []
        for ln in open(a.script, encoding="utf-8"):
            if ln.strip() and not ln.startswith("#"):
                t, s = ln.split(maxsplit=1)
                script.append((float(t), s.strip(), "custom"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    sim = DuckSim(a.ckpt)
    cam = sim.env._offline_renderer._cam
    cam.elevation = -22.0
    # 시작 전 1초 안정화
    for _ in range(int(1.0 / sim.dt)):
        sim.step(Cmd())
    sim.t = 0.0
    every = int(round(1.0 / (sim.dt * OUT_FPS)))
    writer = imageio.get_writer(a.out, fps=OUT_FPS, codec="libx264", quality=None, macro_block_size=8,
                                output_params=["-crf", "27", "-preset", "medium", "-pix_fmt", "yuv420p"])
    log, seg = [], None
    idx, ctrl, said, label = -1, None, "", "대기"
    extra_until = -1
    nsteps = int(a.end / sim.dt)
    t_wall = time.time()
    for k in range(nsteps):
        t = k * sim.dt
        if idx + 1 < len(script) and t >= script[idx + 1][0]:
            idx += 1
            _, said, clip = script[idx]
            b = parse(said)
            if seg:
                log.append(seg)
            if b is None:
                label = "(모르는 말 — 하던 일 계속)"
            else:
                label = b.label
                if b.kind != "follow_ball":
                    sim.ball_v = np.zeros(2)
                ctrl = behaviors.make(b, sim)
            seg = {"t0": t, "said": said, "clip": clip, "kind": b.kind if b else None,
                   "n": 0, "up": 0, "vx": 0.0, "wz": 0.0, "min_gz": -1.0, "ball_d": []}
            yaw_deg = math.degrees(sim.pose()[1])
            if b and b.kind == "follow_ball":       # 덕 뒤 높은 곳에서 공까지 보이게
                cam.distance, cam.elevation, cam.azimuth = 1.15, -30.0, yaw_deg + 20
            elif b and b.kind == "head" and clip == "head" and said.startswith("고개 왼"):  # 정면에서
                cam.distance, cam.elevation, cam.azimuth = 0.8, -18.0, yaw_deg + 180
            elif b and b.kind in ("dance", "getup"):  # 덕 얼굴이 보이게 비스듬히 앞에서
                cam.distance, cam.elevation, cam.azimuth = 0.85, -20.0, yaw_deg + 180 - 35
            elif clip != "ball" and b and b.kind == "walk" and idx == 0:
                cam.distance, cam.elevation, cam.azimuth = 0.95, -20.0, yaw_deg + 120
            extra_until = t + 2.5 if (b and b.kind == "getup") else -1
        c = ctrl.cmd() if ctrl else Cmd()
        sim.step(c)
        p, yaw, gz = sim.pose()
        v = sim.robot.data.root_link_lin_vel_b[0].numpy()
        w = sim.robot.data.root_link_ang_vel_b[0].numpy()
        if seg:
            seg["n"] += 1; seg["up"] += int(gz < -0.8); seg["vx"] += float(v[0]); seg["wz"] += float(w[2])
            seg["min_gz"] = max(seg["min_gz"], gz)  # gz 가 클수록 더 기울어짐
            rl, pt = sim.roll_pitch()
            seg["max_roll_deg"] = max(seg.get("max_roll_deg", 0.0), abs(math.degrees(rl)) if gz < -0.8 else 0.0)
            seg["max_pitch_deg"] = max(seg.get("max_pitch_deg", 0.0), abs(math.degrees(pt)) if gz < -0.8 else 0.0)
            if gz < -0.8 and seg.get("t_up") is None and seg["kind"] == "getup" and seg["n"] > 5:
                seg["t_up"] = round(t - seg["t0"], 2)
            if np.linalg.norm(sim.ball_xy) < 1e3 and seg["kind"] == "follow_ball":
                seg["ball_d"].append(float(np.linalg.norm(sim.ball_xy - p[:2])))
        if k % every == 0:
            pose = "서 있음" if gz < -0.8 else ("일어나는 중" if gz < 0.0 else "넘어짐")
            status = f"{t:4.1f}s · 속도 {v[0]:+.2f} m/s · 회전 {w[2]:+.2f} rad/s · {pose}"
            if seg and seg["kind"] == "follow_ball":
                status += f" · 공 {np.linalg.norm(sim.ball_xy - p[:2]):.2f} m"
            extra = "시뮬레이터가 덕을 엎어 놓음 → 스스로 일어나기" if t < extra_until else None
            writer.append_data(overlay(sim.render(), said, label, status, extra))
        if k % 250 == 0:
            print(f"[{time.time()-t_wall:5.0f}s] t={t:5.1f} {said} gz={gz:+.2f} vx={v[0]:+.2f} wz={w[2]:+.2f}", flush=True)
    if seg:
        log.append(seg)
    writer.close()
    for s in log:
        n = max(s["n"], 1)
        s["upright_frac"] = round(s["up"] / n, 3); s["mean_vx"] = round(s["vx"] / n, 3); s["mean_wz"] = round(s["wz"] / n, 3)
        bd = s.pop("ball_d")
        if bd:
            s["ball_dist_start_mean_end"] = [round(bd[0], 2), round(float(np.mean(bd)), 2), round(bd[-1], 2)]
        for k_ in ("n", "up", "vx", "wz"):
            s.pop(k_)
        s["max_tilt_gz"] = round(s.pop("min_gz"), 2)
        for k_ in ("max_roll_deg", "max_pitch_deg"):
            if k_ in s:
                s[k_] = round(s[k_], 1)
    js = os.path.splitext(a.out)[0] + "_stats.json"
    json.dump(log, open(js, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(log, ensure_ascii=False, indent=1))
    if a.script or a.no_clips:
        return
    # 행동별 클립 자르기
    clips = {}
    for s_ in log:
        c0 = clips.setdefault(s_["clip"], [s_["t0"], None])
    order = [s_["clip"] for s_ in log]
    for i, s_ in enumerate(log):
        clips[s_["clip"]][1] = log[i + 1]["t0"] if i + 1 < len(log) else a.end
    for name, (t0, t1) in clips.items():
        out = os.path.join(os.path.dirname(a.out), f"clip_{name}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.2f}", "-to", f"{t1:.2f}", "-i", a.out,
                        "-c:v", "libx264", "-crf", "27", "-pix_fmt", "yuv420p", "-an", out], check=True)
        print("clip", name, t0, t1, out)


if __name__ == "__main__":
    main()
