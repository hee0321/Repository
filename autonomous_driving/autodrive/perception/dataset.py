"""분할 학습 데이터 만들기: 무작위 시나리오를 expert(+잡음)로 달리며 256×144 영상과 픽셀 정답을 저장.
학습용 시드와 평가(held-out)용 시드를 나눠요. 저장 위치: /workspace/vis-tmp/perception (저장소 밖)."""
from __future__ import annotations

import argparse
import os

import numpy as np

OUT = os.environ.get("PERC_DATA", "/workspace/vis-tmp/perception")


def _episode(job):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from train_vision import make_any_env, train_spec
    from autodrive.controllers import PurePursuitController
    from autodrive.vision.camera import CameraMount, camera_for
    from .seg import PERC_H, PERC_W
    split, i, seed, every, max_frames = job
    rng = np.random.default_rng(seed)
    spec = train_spec(rng, dcas_frac=0.6)
    env = make_any_env(spec, seed=seed)
    obs, _ = env.reset(seed=seed)
    ex = PurePursuitController(env)
    cam = camera_for(env, mount=CameraMount(width=PERC_W, height_px=PERC_H), rng=rng)
    imgs, labs = [], []
    step, done = 0, False
    off = int(rng.integers(0, every))
    while not done and len(imgs) < max_frames:
        a = ex.act(obs)
        a = np.clip(a + rng.normal(0, 0.05, 2), -1, 1)
        if step % every == off:
            im, c, _ = cam.render_with_labels(noise=float(rng.uniform(0, 5)))
            imgs.append(im)
            labs.append(c)
        obs, r, term, trunc, info = env.step(a)
        step += 1
        done = term or trunc
    if imgs:
        np.savez_compressed(os.path.join(OUT, split, f"ep{i:04d}.npz"), img=np.stack(imgs), lab=np.stack(labs))
    return len(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=120)
    ap.add_argument("--test", type=int, default=30)
    ap.add_argument("--every", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    from multiprocessing import get_context
    jobs = []
    for split, n, s0 in (("train", a.train, 70_000), ("test", a.test, 90_000)):   # 시드 영역이 겹치지 않음
        os.makedirs(os.path.join(OUT, split), exist_ok=True)
        jobs += [(split, i, s0 + i, a.every, 60) for i in range(n)]
    with get_context("spawn").Pool(a.workers) as pool:
        n = sum(pool.imap_unordered(_episode, jobs))
    print("frames", n)


if __name__ == "__main__":
    main()
