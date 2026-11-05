"""분할 CNN 학습 (CPU). 데이터: python -m autodrive.perception.dataset 로 만든 /workspace/vis-tmp/perception/{train,test}.
python train_perception.py --epochs 4  → weights/perception_seg.pt (저장소에 안 올림)"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from autodrive.perception.dataset import OUT
from autodrive.perception.seg import N_CLS, SegNet
from autodrive.vision.camera import LABELS


def load(split):
    I, L = [], []
    for f in sorted(glob.glob(os.path.join(OUT, split, "*.npz"))):
        with np.load(f) as d:
            I.append(d["img"]); L.append(d["lab"])
    return np.concatenate(I), np.concatenate(L)


@torch.no_grad()
def evaluate(m, I, L, bs=32):
    m.eval()
    inter = np.zeros(N_CLS); union = np.zeros(N_CLS); gt = np.zeros(N_CLS)
    for b in range(0, len(I), bs):
        x = torch.from_numpy(I[b:b + bs].transpose(0, 3, 1, 2).copy())
        p = m(x).argmax(1).numpy()
        y = L[b:b + bs]
        for c in range(N_CLS):
            inter[c] += np.sum((p == c) & (y == c)); union[c] += np.sum((p == c) | (y == c)); gt[c] += np.sum(y == c)
    return {LABELS[c]: round(float(inter[c] / union[c]), 3) for c in range(N_CLS) if gt[c] > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default="weights/perception_seg.pt")
    a = ap.parse_args()
    torch.set_num_threads(8)
    I, L = load("train")
    It, Lt = load("test")
    print("train", len(I), "test", len(It), flush=True)
    freq = np.bincount(L.ravel(), minlength=N_CLS).astype(float) + 1
    w = torch.tensor(np.clip((freq.sum() / freq) ** 0.5, 1, 60) / 5, dtype=torch.float32)
    m = SegNet()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = a.epochs * (len(I) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for ep in range(a.epochs):
        m.train()
        perm = rng.permutation(len(I))
        tot = 0
        for k in range(len(I) // a.bs):
            idx = np.sort(perm[k * a.bs:(k + 1) * a.bs])
            x = torch.from_numpy(I[idx].transpose(0, 3, 1, 2).copy()).float()
            g = torch.from_numpy(rng.uniform(0.7, 1.3, (len(idx), 1, 1, 1)).astype(np.float32))
            x = (x * g + torch.from_numpy(rng.uniform(-20, 20, (len(idx), 1, 1, 1)).astype(np.float32))).clamp(0, 255)
            y = torch.from_numpy(L[idx].astype(np.int64))
            if rng.random() < 0.5:   # 좌우 뒤집기 (신호등 색 위치도 뒤집히지만 색으로 구분하므로 괜찮음)
                x, y = x.flip(3), y.flip(2)
            loss = F.cross_entropy(m(x), y, weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss)
        iou = evaluate(m, It, Lt)
        print(f"epoch {ep + 1} loss {tot / (len(I) // a.bs):.4f} ({time.time() - t0:.0f}s) test IoU {iou}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(m.state_dict(), a.out)
    json.dump(dict(test_iou=iou, train_frames=len(I), test_frames=len(It), epochs=a.epochs),
              open("results/perception_seg.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
