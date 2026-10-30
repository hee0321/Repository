"""비전 정책 학습: 행동 복제(BC) + DAgger. 선생님 = 규칙 기반 기준 제어기(expert), 학생 = 카메라+내비만 보는 CNN.

라운드 0: expert 가 운전(실행 행동에 작은 잡음 DART → 벗어났다 돌아오는 장면도 모음)
라운드 k: 학생이 운전(β 확률로 expert), expert 가 매 순간 정답 표시 → 모든 라운드 데이터를 모아 다시 학습
가중치·데이터는 저장소 밖(--data, 기본 /workspace/autodrive-vision-data)에. 점수표만 results/ 에.

사용: python train_vision.py --rounds 3 --episodes 96 --workers 4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

DATA = os.environ.get("VISION_DATA", "/workspace/autodrive-vision-data")


def train_spec(rng):
    """학습 시나리오 하나 (도시 실제 도로 45%, 시골 30%, 합성 25%). 실제 도로는 중간 출발·신호 시드·차 배치 무작위."""
    from autodrive.scenarios import BENCHMARK, RURAL_BENCHMARK, korea_route
    r = rng.random()
    if r < 0.45:
        name = str(rng.choice(["gangnam_station", "gangnam_reverse", "daechi_school_zone", "gangnam_station",
                               "jeongbae_village", "jeongbae_night", "jeongbae_lane"]))
        tr, ru, _ = korea_route(name, 0)
        L = tr.length
        sp = dict(korea=name, signal_seed=int(rng.integers(6)), max_steps=1600,
                  start_s=float(rng.uniform(0, max(1.0, L - 200))) if rng.random() < 0.6 else None)
        if name.startswith("jeongbae"):
            sp.update(traffic_seed=int(rng.integers(1e6)), night=bool(rng.random() < 0.4), difficulty=float(rng.uniform(0.6, 1.6)))
        else:
            obs = []
            s0 = sp["start_s"] or 0.0
            for _ in range(int(rng.integers(0, 4))):
                s_o = float(s0 + rng.uniform(25, 300))
                if s_o > L - 20:
                    continue
                if rng.random() < 0.5:
                    obs.append(dict(s=s_o, d=-1.6, v=float(rng.uniform(3, 9)), r=1.2, l=4.5, kind="car"))
                else:
                    obs.append(dict(s=s_o, d=float(rng.choice([-1.6, -2.0, 1.6])), v=0.0, r=1.1))
            sp["obstacles"] = obs
        return sp
    if r < 0.75:
        return dict(kind="rural", road_seed=int(rng.integers(1000, 1e6)), difficulty=float(rng.uniform(0.6, 1.8)),
                    night=bool(rng.random() < 0.35), length=600.0, max_steps=1500, mu=float(rng.uniform(0.7, 1.0)))
    if r < 0.85:
        sp = dict(BENCHMARK[str(rng.choice(["oval_easy", "default_2laps", "hard_2laps", "hard_wet", "rules_track", "obstacles_track"]))])
        sp["max_steps"] = 1600
        sp["signal_seed"] = int(rng.integers(1e6)) if sp.get("signals") else 0
        return sp
    sp = dict(kind="random", track_seed=int(rng.integers(1e6)), difficulty=1.2, limit_kmh=float(rng.choice([30, 40, 50, 60])),
              signals=int(rng.integers(0, 4)), signal_seed=int(rng.integers(1e6)), lane_ref=-1.6, twoway=True, width=6.4,
              max_steps=1500, mu=float(rng.uniform(0.6, 1.0)))
    if rng.random() < 0.5:
        sp["obstacles"] = [dict(s=float(rng.uniform(40, 220)), d=float(rng.uniform(-1.8, 0.5)), v=0.0, r=1.0),
                           dict(s=float(rng.uniform(40, 220)), d=-1.6, v=float(rng.uniform(3, 7)), r=1.0)]
    return sp


def _worker(job):
    import torch
    torch.set_num_threads(1)
    from autodrive.scenarios import make_env
    from autodrive.vision.drive import drive_episode
    from autodrive.vision.policy import load_policy
    rnd, i, spec, pol, beta, dart, seed = job
    policy = load_policy(pol) if pol else None
    env = make_env(spec, seed=seed)
    res, data, _ = drive_episode(env, policy, seed=seed, beta=beta, dart=dart, collect=True,
                                 rng=np.random.default_rng(seed), cam_noise=float(np.random.default_rng(seed).uniform(0, 6)))
    p = os.path.join(DATA, f"r{rnd:02d}_{i:04d}.npz")
    np.savez(p, **data)
    return dict(i=i, n=len(data["act"]), crashed=res["crashed"], completion=res["completion"],
                red=res["red_runs"] + res.get("right_turn_no_stop", 0), spec=spec.get("korea") or spec.get("kind"))


def collect(rnd, n_ep, workers, pol, beta, dart, seed0):
    from multiprocessing import get_context
    rng = np.random.default_rng(1000 + rnd)
    jobs = [(rnd, i, train_spec(rng), pol, beta, dart, seed0 + i) for i in range(n_ep)]
    out = []
    with get_context("spawn").Pool(workers, maxtasksperchild=8) as pool:
        for k, r in enumerate(pool.imap_unordered(_worker, jobs)):
            out.append(r)
            if k % 16 == 0:
                print(f"  collect r{rnd} {k + 1}/{n_ep} frames {sum(o['n'] for o in out)}", flush=True)
    return out


def build_memmap(rounds_upto):
    files = sorted(f for f in glob.glob(os.path.join(DATA, "r*_*.npz")) if int(os.path.basename(f)[1:3]) <= rounds_upto)
    N = 0
    meta = []
    for f in files:
        with np.load(f) as d:
            n = len(d["act"])
        meta.append((f, n))
        N += n
    img = np.lib.format.open_memmap(os.path.join(DATA, "img.npy"), mode="w+", dtype=np.uint8, shape=(N, 4, 72, 128))
    nav = np.zeros((N, 28), np.float32)
    act = np.zeros((N, 2), np.float32)
    w = np.zeros(N, np.float32)
    o = 0
    for f, n in meta:
        with np.load(f) as d:
            img[o:o + n] = d["img"]
            nav[o:o + n] = d["nav"]
            act[o:o + n] = d["act"]
            w[o:o + n] = d["w"]
        o += n
    img.flush()
    del img
    return np.load(os.path.join(DATA, "img.npy"), mmap_mode="r"), nav, act, w


def train(img, nav, act, w, init=None, epochs=4, bs=256, lr=1e-3, threads=8, log=print):
    import torch
    from autodrive.vision.policy import VisionPolicy
    torch.set_num_threads(threads)
    m = VisionPolicy()
    if init and os.path.exists(init):
        m.load_state_dict(torch.load(init, map_location="cpu"))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    N = len(act)
    # 드문 '결정' 순간 더 가중: 멈춘 상태에서 출발(초록불·우회전 일시정지 뒤·장애물 비킴), 멈춘 상태 유지(빨간불),
    # 달리다 세게 감속. (BC 가 '계속 멈춰 있기'/'계속 달리기'를 따라 하는 문제 완화)
    v = nav[:, 0] * 20.0
    w = w * (1.0 + 3.0 * ((v < 1.0) & (act[:, 1] > 0.2)) + 1.0 * ((v < 1.0) & (act[:, 1] < -0.2))
             + 1.0 * ((v > 5.0) & (act[:, 1] < -0.5)))
    steps = epochs * (N // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.15)
    rng = np.random.default_rng(0)
    t0 = time.time()
    k = 0
    for ep in range(epochs):
        perm = rng.permutation(N)
        tot = 0.0
        for b in range(N // bs):
            idx = np.sort(perm[b * bs:(b + 1) * bs])
            x = torch.from_numpy(np.ascontiguousarray(img[idx]))
            # 증강: 밝기·대비 흔들기 (uint8 그대로 넣고 float 변환 뒤에)
            xf = x.float()
            g = torch.from_numpy(rng.uniform(0.75, 1.25, (len(idx), 1, 1, 1)).astype(np.float32))
            bb = torch.from_numpy(rng.uniform(-20, 20, (len(idx), 1, 1, 1)).astype(np.float32))
            x = (xf * g + bb).clamp(0, 255)
            nv = torch.from_numpy(nav[idx])
            y = torch.from_numpy(act[idx])
            ww = torch.from_numpy(w[idx])
            pred = m(x, nv)
            loss = ((pred - y) ** 2 * torch.tensor([1.5, 1.0])).sum(1)
            loss = (loss * ww).sum() / ww.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            k += 1
        log(f"  epoch {ep + 1}/{epochs} loss {tot / max(N // bs, 1):.4f} ({time.time() - t0:.0f}s)")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--start-round", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=96)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--polish", type=int, default=-1, help="모으지 않고 이 라운드까지 데이터로 다시 학습만 (이어서)")
    a = ap.parse_args()
    import torch
    os.makedirs(DATA, exist_ok=True)
    wdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    os.makedirs(wdir, exist_ok=True)
    pol = os.path.join(wdir, "vision_policy.pt")
    logp = os.path.join(DATA, "train_log.jsonl")
    if a.polish >= 0:
        img, nav, act, w = build_memmap(a.polish)
        print(f"polish: dataset {len(act)} frames", flush=True)
        m = train(img, nav, act, w, init=pol, epochs=a.epochs, lr=3e-4)
        torch.save(m.state_dict(), pol)
        torch.save(m.state_dict(), os.path.join(wdir, f"vision_policy_r{a.polish}p.pt"))
        return
    for rnd in range(a.start_round, a.start_round + a.rounds):
        t = time.time()
        if rnd == 0:
            stats = collect(0, a.episodes, a.workers, None, 0.0, 0.08, 10_000)
        else:
            beta = max(0.0, 0.4 - 0.15 * rnd)
            stats = collect(rnd, a.episodes, a.workers, pol, beta, 0.03, 10_000 + 1000 * rnd)
        print(f"round {rnd}: collected {sum(s['n'] for s in stats)} frames, crashes {sum(s['crashed'] for s in stats)}, "
              f"({time.time() - t:.0f}s)", flush=True)
        img, nav, act, w = build_memmap(rnd)
        print(f"  dataset {len(act)} frames", flush=True)
        m = train(img, nav, act, w, init=pol if rnd > 0 else None, epochs=a.epochs, lr=1e-3 if rnd == 0 else 5e-4)
        torch.save(m.state_dict(), pol)
        torch.save(m.state_dict(), os.path.join(wdir, f"vision_policy_r{rnd}.pt"))
        with open(logp, "a") as f:
            f.write(json.dumps(dict(round=rnd, frames=len(act), episodes=a.episodes,
                                    drive_crashes=sum(s["crashed"] for s in stats),
                                    drive_signal_viol=sum(s["red"] for s in stats),
                                    drive_completion=round(float(np.mean([s["completion"] for s in stats])), 3),
                                    time_s=round(time.time() - t)), ensure_ascii=False) + "\n")
        del img


if __name__ == "__main__":
    main()
