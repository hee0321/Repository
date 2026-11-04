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


def dcas_spec(rng):
    """DCAS 01 시나리오 하나를 조금 바꾼 판 (상대 위치·속도 ±10~15%, 진입 속도 ±10%) — 시험 그대로 외우지 않게."""
    import copy
    from autodrive.dcas import DCAS01
    name = str(rng.choice(list(DCAS01)))
    sc = copy.deepcopy(DCAS01[name])
    sc["name"] = name
    sc["v0"] = float(sc.get("v0", 0) * rng.uniform(0.9, 1.05))
    for o in sc.get("obstacles", []):
        o["s"] = float(o["s"] * rng.uniform(0.85, 1.15)) if o["s"] > 0 else o["s"]
        if o.get("v", 0) > 0:
            o["v"] = float(o["v"] * rng.uniform(0.9, 1.1))
    return {"dcas": sc}


def make_any_env(spec, seed=0):
    if "dcas" in spec:
        from autodrive import dcas
        return dcas.make_env(spec["dcas"], seed=seed)
    from autodrive.scenarios import make_env
    return make_env(spec, seed=seed)


def train_spec(rng, dcas_frac=0.0):
    """학습 시나리오 하나 (dcas_frac 만큼 DCAS 01 변형, 나머지: 도시 실제 도로 45%, 시골 30%, 합성 25%).
    실제 도로는 중간 출발·신호 시드·차 배치 무작위."""
    from autodrive.scenarios import BENCHMARK, RURAL_BENCHMARK, korea_route
    if dcas_frac > 0 and rng.random() < dcas_frac:
        return dcas_spec(rng)
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
    from autodrive.vision.drive import drive_episode
    from autodrive.vision.policy import load_policy
    rnd, i, spec, pol, beta, dart, seed = job
    policy = load_policy(pol) if pol else None
    env = make_any_env(spec, seed=seed)
    res, data, _ = drive_episode(env, policy, seed=seed, beta=beta, dart=dart, collect=True,
                                 rng=np.random.default_rng(seed), cam_noise=float(np.random.default_rng(seed).uniform(0, 6)))
    p = os.path.join(DATA, f"r{rnd:02d}_{i:04d}.npz")
    np.savez(p, **data)
    return dict(i=i, n=len(data["act"]), crashed=res["crashed"], completion=res["completion"],
                red=res["red_runs"] + res.get("right_turn_no_stop", 0), spec=spec.get("korea") or spec.get("kind") or ("dcas:" + spec["dcas"]["name"] if "dcas" in spec else "?"))


def collect(rnd, n_ep, workers, pol, beta, dart, seed0, dcas_frac=0.0):
    from multiprocessing import get_context
    rng = np.random.default_rng(1000 + rnd)
    jobs = [(rnd, i, train_spec(rng, dcas_frac), pol, beta, dart, seed0 + i) for i in range(n_ep)]
    out = []
    with get_context("spawn").Pool(workers, maxtasksperchild=8) as pool:
        for k, r in enumerate(pool.imap_unordered(_worker, jobs)):
            out.append(r)
            if k % 16 == 0:
                print(f"  collect r{rnd} {k + 1}/{n_ep} frames {sum(o['n'] for o in out)}", flush=True)
    return out


def pack_round(rnd):
    """라운드 rnd 의 에피소드 npz 들 → round_XX_img.npy(메모리 맵) + round_XX_meta.npz (nav 는 NAV_DIM 으로 0 채움).
    한 번 묶으면 에피소드 파일은 지운다(디스크 절약). 이미 묶였으면 그대로."""
    from autodrive.vision.nav import NAV_DIM
    ip = os.path.join(DATA, f"round_{rnd:02d}_img.npy")
    mp = os.path.join(DATA, f"round_{rnd:02d}_meta.npz")
    files = sorted(glob.glob(os.path.join(DATA, f"r{rnd:02d}_*.npz")))
    if os.path.exists(mp) and not files:
        return
    if not files:
        return
    ns = []
    for f in files:
        with np.load(f) as d:
            ns.append(len(d["act"]))
    N = sum(ns)
    img = np.lib.format.open_memmap(ip + ".tmp.npy", mode="w+", dtype=np.uint8, shape=(N, 4, 72, 128))
    nav = np.zeros((N, NAV_DIM), np.float32)
    act = np.full((N, 3), np.nan, np.float32)
    w = np.zeros(N, np.float32)
    o = 0
    for f, n in zip(files, ns):
        with np.load(f) as d:
            img[o:o + n] = d["img"]
            nv = d["nav"]
            nav[o:o + n, :nv.shape[1]] = nv
            act[o:o + n, :d["act"].shape[1]] = d["act"]
            w[o:o + n] = d["w"]
        o += n
    img.flush()
    del img
    os.replace(ip + ".tmp.npy", ip)
    np.savez(mp, nav=nav, act=act, w=w)
    for f in files:
        os.remove(f)


class RoundData:
    """라운드별 메모리 맵 묶음. img[idx] (idx 오름차순) → (B,4,72,128) uint8."""

    def __init__(self, rounds):
        self.imgs, navs, acts, ws, self.off = [], [], [], [], [0]
        for r in rounds:
            mp = os.path.join(DATA, f"round_{r:02d}_meta.npz")
            if not os.path.exists(mp):
                continue
            self.imgs.append(np.load(os.path.join(DATA, f"round_{r:02d}_img.npy"), mmap_mode="r"))
            with np.load(mp) as d:
                navs.append(d["nav"]); ws.append(d["w"])
                ac = d["act"]
                if ac.shape[1] < 3:   # 지시등 정답이 없는 예전 데이터: NaN → 손실에서 뺌
                    ac = np.concatenate([ac, np.full((len(ac), 3 - ac.shape[1]), np.nan, np.float32)], 1)
                acts.append(ac)
            self.off.append(self.off[-1] + len(acts[-1]))
        self.nav, self.act, self.w = np.concatenate(navs), np.concatenate(acts), np.concatenate(ws)
        self.off = np.array(self.off)

    def __len__(self):
        return len(self.act)

    def __getitem__(self, idx):
        idx = np.asarray(idx)
        out = np.empty((len(idx), 4, 72, 128), np.uint8)
        r = np.searchsorted(self.off, idx, side="right") - 1
        for k in np.unique(r):
            m = r == k
            out[m] = self.imgs[k][idx[m] - self.off[k]]
        return out


def all_rounds():
    rs = set()
    for f in glob.glob(os.path.join(DATA, "round_*_meta.npz")) + glob.glob(os.path.join(DATA, "r*_*.npz")):
        b = os.path.basename(f)
        rs.add(int(b[6:8]) if b.startswith("round_") else int(b[1:3]))
    return sorted(rs)


def build_memmap(rounds_upto):
    for r in all_rounds():
        if r <= rounds_upto:
            pack_round(r)
    d = RoundData([r for r in all_rounds() if r <= rounds_upto])
    return d, d.nav, d.act, d.w


def train(img, nav, act, w, init=None, epochs=4, bs=256, lr=1e-3, threads=8, log=print, max_samples=None,
          recent_from=None, recent_frac=0.4):
    """max_samples: 에포크마다 이만큼만 뽑기(매일 라운드 시간 제한). recent_from 이후 인덱스(새 라운드)에서 recent_frac."""
    import torch
    from autodrive.vision.policy import VisionPolicy, load_policy
    torch.set_num_threads(threads)
    m = load_policy(init) if init and os.path.exists(init) else VisionPolicy()   # 예전 28칸 가중치도 0 채워 이어 받음
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    N = len(act)
    # 드문 '결정' 순간 더 가중: 멈춘 상태에서 출발(초록불·우회전 일시정지 뒤·장애물 비킴), 멈춘 상태 유지(빨간불),
    # 달리다 세게 감속. (BC 가 '계속 멈춰 있기'/'계속 달리기'를 따라 하는 문제 완화)
    v = nav[:, 0] * 20.0
    w = w * (1.0 + 3.0 * ((v < 1.0) & (act[:, 1] > 0.2)) + 1.0 * ((v < 1.0) & (act[:, 1] < -0.2))
             + 1.0 * ((v > 5.0) & (act[:, 1] < -0.5)))
    steps = epochs * (min(N, max_samples or N) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.15)
    rng = np.random.default_rng(0)
    t0 = time.time()
    k = 0
    for ep in range(epochs):
        perm = rng.permutation(N)
        if max_samples and max_samples < N:
            if recent_from is not None and recent_from < N:
                nr = min(int(max_samples * recent_frac), N - recent_from)
                perm = np.r_[rng.choice(np.arange(recent_from, N), nr, replace=False),
                             rng.choice(recent_from, max_samples - nr, replace=False)]
                perm = rng.permutation(perm)
            else:
                perm = perm[:max_samples]
        tot = 0.0
        for b in range(len(perm) // bs):
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
            y3 = y[:, 2] if y.shape[1] > 2 else torch.full((len(y),), float("nan"))
            has = ~torch.isnan(y3)
            loss = ((pred[:, :2] - y[:, :2]) ** 2 * torch.tensor([1.5, 1.0])).sum(1)
            loss = loss + 0.5 * torch.where(has, (pred[:, 2] - torch.nan_to_num(y3)) ** 2, torch.zeros_like(y3))
            loss = (loss * ww).sum() / ww.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            k += 1
        log(f"  epoch {ep + 1}/{epochs} loss {tot / max(len(perm) // bs, 1):.4f} ({time.time() - t0:.0f}s)")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--start-round", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=96)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--polish", type=int, default=-1, help="모으지 않고 이 라운드까지 데이터로 다시 학습만 (이어서)")
    ap.add_argument("--daily", action="store_true",
                    help="매일 라운드 1개: weights/vision_best.pt 로 DAgger 모으기(DCAS 섞기) → 후보 weights/vision_cand.pt")
    ap.add_argument("--dcas-frac", type=float, default=0.25)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4, help="--daily 미세 조정 학습률 (최고 학습률, OneCycle)")
    a = ap.parse_args()
    import torch
    os.makedirs(DATA, exist_ok=True)
    wdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    os.makedirs(wdir, exist_ok=True)
    pol = os.path.join(wdir, "vision_policy.pt")
    logp = os.path.join(DATA, "train_log.jsonl")
    if a.daily:
        best = os.path.join(wdir, "vision_best.pt")
        if not os.path.exists(best):
            import shutil
            shutil.copy(pol, best)
        rnd = (max(all_rounds()) + 1) if all_rounds() else 0
        t = time.time()
        stats = collect(rnd, a.episodes, a.workers, best, 0.1, 0.03, 10_000 + 1000 * rnd, dcas_frac=a.dcas_frac)
        print(f"daily round {rnd}: collected {sum(s['n'] for s in stats)} frames, crashes "
              f"{sum(s['crashed'] for s in stats)} ({time.time() - t:.0f}s)", flush=True)
        for r in all_rounds():
            pack_round(r)
        d = RoundData(all_rounds())
        recent = int(d.off[-2]) if len(d.off) >= 2 else 0
        print(f"  dataset {len(d)} frames (new {len(d) - recent})", flush=True)
        m = train(d, d.nav, d.act, d.w, init=best, epochs=a.epochs, lr=a.lr,
                  max_samples=a.max_samples, recent_from=recent)
        torch.save(m.state_dict(), os.path.join(wdir, "vision_cand.pt"))
        with open(logp, "a") as f:
            f.write(json.dumps(dict(round=rnd, daily=True, frames=len(d), new=len(d) - recent, episodes=a.episodes,
                                    drive_crashes=sum(s["crashed"] for s in stats),
                                    dcas_episodes=sum(str(s["spec"]).startswith("dcas") for s in stats),
                                    time_s=round(time.time() - t)), ensure_ascii=False) + "\n")
        print(json.dumps(dict(round=rnd, cand="weights/vision_cand.pt", time_s=round(time.time() - t))), flush=True)
        return
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
