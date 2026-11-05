"""인식 시제품 영상 만들기: 카메라 오버레이 | 평면도(예측 vs 정답).
python render_perception_video.py --scenario dcas_stationary_70 --out /workspace/vision_perception_demo.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="dcas_stationary_70")
    ap.add_argument("--seconds", type=float, default=25)
    ap.add_argument("--seg", default="weights/perception_seg.pt")
    ap.add_argument("--policy", default="weights/vision_best.pt")
    ap.add_argument("--out", default="/workspace/vision_perception_demo.mp4")
    ap.add_argument("--png", default="/workspace/vision_perception_demo.png")
    a = ap.parse_args()
    from autodrive import dcas
    from autodrive.scenarios import make_env
    from autodrive.bench import ALL_BENCH
    from autodrive.perception.drive import drive_perception
    from autodrive.perception.seg import load_seg
    from autodrive.perception.viz import compose
    from autodrive.vision.policy import load_policy
    import imageio_ffmpeg
    if a.scenario in dcas.DCAS01:
        env = dcas.make_env(a.scenario)
    else:
        env = make_env(ALL_BENCH[a.scenario], seed=0)
    env.max_steps = int(a.seconds / env.dt) + 50
    res, metrics, frames = drive_perception(env, load_policy(a.policy), load_seg(a.seg), seed=0, measure=True, record=True)
    print("result", {k: res[k] for k in ("finished", "crashed", "completion", "collisions", "planner_brake_share")})
    if not frames:
        sys.exit("no frames")
    compose(frames[len(frames) // 2]).save(a.png)
    fps = 1.0 / (env.dt * 2)
    writer = imageio_ffmpeg.write_frames(a.out, compose(frames[0]).size, fps=fps, quality=7)
    writer.send(None)
    for f in frames:
        writer.send(np.asarray(compose(f)))
    writer.close()
    print("wrote", a.out, a.png, "frames", len(frames))


if __name__ == "__main__":
    main()
