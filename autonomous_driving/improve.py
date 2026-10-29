"""한 번의 '개선 사이클'을 돌린다 (CPU용). 매번 이전 최고 모델에서 이어서 학습하고, 벤치마크에서
나빠진 곳 없이 더 좋아질 때만 최고 모델을 바꾼다.

    python improve.py --budget-minutes 6           # 1 사이클
    python improve.py --budget-minutes 6 --cycles 2

순서: ① 최고 모델 불러오기(없으면 새로) → ② 기준 제어기 흉내 내기(BC, DAgger 방식: 모델이 직접 달린 상황에
기준 제어기가 정답을 붙임) → ③ PPO 강화학습(커리큘럼 + 도메인 무작위화 + 실패 사례 섞기) →
④ 고정 벤치마크 점수표 → ⑤ 승격 판단(점수↑, 회귀 없음) → ⑥ 리더보드/실패 사례 기록.
가중치는 weights/ (git 무시), 점수표·리더보드는 results/ (작은 텍스트라 저장소에 올림).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import time

import numpy as np

from autodrive.bench import compare, evaluate, scorecard_md
from autodrive.controllers import PurePursuitController
from autodrive.scenarios import BENCH_VERSION, STAGES, make_env, sample_train_spec

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    raise SystemExit("pip install gymnasium stable-baselines3 torch(CPU) 가 필요해요")

W, R = "weights", "results"
BEST = os.path.join(W, "best.zip")
STATS = os.path.join(W, "obs_stats.npz")
FAIL = os.path.join(R, "failures.jsonl")
LB_CSV = os.path.join(R, "leaderboard.csv")
LB_MD = os.path.join(R, "leaderboard.md")
BEST_CARD = os.path.join(R, "best_scorecard.json")


class MixEnv(gym.Env):
    """reset 때마다 커리큘럼 단계에 맞는 시나리오를 새로 뽑는 학습용 환경(도메인 무작위화 포함)."""

    def __init__(self, seed=0, stage=1, failures=None):
        self.rng = np.random.default_rng(seed)
        self.stage = stage
        self.failures = failures or []
        self.env = make_env(sample_train_spec(self.rng, 1), seed=seed)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, *, seed=None, options=None):
        spec = sample_train_spec(self.rng, self.stage, self.failures)
        self.spec = spec
        self.env = make_env(spec, seed=int(self.rng.integers(1e9)), random_start=spec.get("start_s") is None)
        return self.env.reset(seed=int(self.rng.integers(1e9)))

    def step(self, a):
        return self.env.step(a)


def load_failures():
    if not os.path.exists(FAIL):
        return []
    with open(FAIL, encoding="utf-8") as f:
        return [json.loads(x)["spec"] for x in f if x.strip()][-200:]


def policy_mean(model, obs):
    import torch
    pol = model.policy
    feats = pol.extract_features(torch.as_tensor(obs), pol.pi_features_extractor)
    return pol.action_net(pol.mlp_extractor.forward_actor(feats))


def dagger(model, minutes, stage, failures, rng, beta0=0.5, log=print):
    """모델이 달리는 상태를 모으고 기준 제어기의 행동을 정답으로 붙여 지도학습(BC/DAgger)."""
    import torch
    t_end = time.time() + minutes * 60
    X, Y = [], []
    rounds = 0
    opt = torch.optim.Adam(list(model.policy.mlp_extractor.policy_net.parameters()) +
                           list(model.policy.action_net.parameters()), lr=1e-3)
    while time.time() < t_end:
        beta = beta0 * (0.5 ** rounds)  # 처음엔 기준 제어기가 운전을 더 많이, 점점 모델이 운전
        collect_until = time.time() + max(10.0, (t_end - time.time()) * 0.45)
        while time.time() < collect_until:
            spec = sample_train_spec(rng, stage, failures)
            env = make_env(spec, seed=int(rng.integers(1e9)), random_start=spec.get("start_s") is None)
            obs, _ = env.reset(seed=int(rng.integers(1e9)))
            ex = PurePursuitController(env)
            done, n = False, 0
            while not done and n < 900:
                a_exp = ex.act(obs)
                X.append(obs); Y.append(a_exp)
                if rng.random() < beta:
                    a = a_exp + rng.normal(0, 0.1, 2)   # 기준 제어기 + 약간의 흔들림
                else:
                    a = model.predict(obs, deterministic=True)[0]
                obs, _, term, trunc, _ = env.step(a)
                done = term or trunc
                n += 1
        X, Y = X[-150_000:], Y[-150_000:]   # 최근 데이터 위주(메모리·시간 제한)
        Xa = torch.as_tensor(np.array(X, np.float32)); Ya = torch.as_tensor(np.clip(np.array(Y, np.float32), -1, 1))
        epochs = 6
        for ep in range(epochs):
            perm = torch.randperm(len(Xa))
            tot = 0.0
            for i in range(0, len(Xa), 512):
                b = perm[i:i + 512]
                loss = ((policy_mean(model, Xa[b]) - Ya[b]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach()) * len(b)
            if time.time() > t_end + 30:
                break
        rounds += 1
        log(f"  DAgger 라운드 {rounds}: 데이터 {len(Xa)}개, 흉내 오차(MSE) {tot / len(Xa):.4f}, β={beta:.2f}")
    X = np.array(X, np.float32)
    return X


def make_venv(stage, failures, n_envs, seed):
    from stable_baselines3.common.vec_env import DummyVecEnv
    return DummyVecEnv([lambda i=i: MixEnv(seed + i, stage, failures) for i in range(n_envs)])


def ppo_finetune(model, minutes, stages, log=print):
    from stable_baselines3.common.callbacks import BaseCallback
    venv = model.get_env()
    t0 = time.time()
    total = minutes * 60

    class Clock(BaseCallback):
        def _on_step(self):
            frac = (time.time() - t0) / total
            st = stages[min(int(frac * len(stages)), len(stages) - 1)]
            for e in venv.envs:
                e.stage = st
            return frac < 1.0

    model.learn(total_timesteps=10_000_000, callback=Clock(), reset_num_timesteps=False, progress_bar=False)
    log(f"  PPO: {model.num_timesteps} 누적 스텝, 단계 {stages}, {time.time() - t0:.0f}s")


def new_model(venv, seed):
    from stable_baselines3 import PPO
    return PPO("MlpPolicy", venv, n_steps=512, batch_size=256, learning_rate=1e-4, clip_range=0.1, n_epochs=5,
               gamma=0.99, ent_coef=0.0, policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.2),
               device="cpu", verbose=0, seed=seed)


def append_leaderboard(row):
    os.makedirs(R, exist_ok=True)
    new = not os.path.exists(LB_CSV)
    with open(LB_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    with open(LB_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = list(rows[0])
    md = "# 리더보드 (벤치마크 " + BENCH_VERSION + ")\n\n`python improve.py`가 사이클마다 한 줄씩 추가해요. 점수는 높을수록 좋아요.\n\n"
    md += "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    md += "".join("| " + " | ".join(r.get(c, "") for c in cols) + " |\n" for r in rows)
    with open(LB_MD, "w", encoding="utf-8") as f:
        f.write(md)


def card_row(tag, card, promoted, note, minutes):
    sc = card["scenarios"]
    return {"time": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "model": tag, "total": card["total"],
            "urban": card.get("urban"), "rural": card.get("rural"),
            "finished": f"{sum(v['finished'] for v in sc.values())}/{len(sc)}",
            "crashes": sum(v["crashed"] for v in sc.values()),
            "violations": sum(v["red_runs"] + v.get("right_turn_no_stop", 0) for v in sc.values()),
            "speeding_s": round(sum(v["speeding_s"] for v in sc.values()), 1),
            "school_speeding_s": round(sum(v["school_speeding_s"] for v in sc.values()), 1),
            "mean_lane_dev_m": round(float(np.mean([v["mean_lane_dev_m"] for v in sc.values()])), 3),
            "rural_unsafe": sum(v.get("unsafe_meet", 0) + v.get("ped_unsafe", 0) + v.get("unsafe_overtake", 0)
                                for v in sc.values()),
            "promoted": promoted, "minutes": minutes, "note": note}


def log_print(*a):
    print(*a, flush=True)


def cycle(args, k, log=log_print):
    from stable_baselines3 import PPO
    os.makedirs(W, exist_ok=True); os.makedirs(R, exist_ok=True)
    rng = np.random.default_rng(args.seed + 1000 * k)
    t0 = time.time()
    if not os.path.exists(os.path.join(R, "baseline_scorecard.json")):
        base = evaluate("pure_pursuit")
        json.dump(base, open(os.path.join(R, "baseline_scorecard.json"), "w"), ensure_ascii=False, indent=1)
        append_leaderboard(card_row("pure_pursuit(기준)", base, "-", "규칙 기반 기준 제어기", 0))
    best_card = None
    if os.path.exists(BEST) and os.path.exists(BEST_CARD):
        best_card = json.load(open(BEST_CARD, encoding="utf-8"))
        if best_card.get("bench") != BENCH_VERSION:
            best_card = evaluate(BEST); best_card["bench"] = BENCH_VERSION
        model = PPO.load(BEST, env=make_venv(3, load_failures(), args.n_envs, args.seed + k), device="cpu")
        log(f"[사이클 {k}] 최고 모델에서 이어 학습 (현재 최고 {best_card['total']})")
        stages = [3, 4] if args.rural else [2, 3]
    else:
        model = new_model(make_venv(1, load_failures(), args.n_envs, args.seed + k), args.seed)
        log(f"[사이클 {k}] 최고 모델 없음 → 새로 시작 (기준 제어기 흉내로 시작)")
        stages = [1, 2, 3, 4] if args.rural else [1, 2, 3]
    failures = load_failures()
    log(f"  실패 사례 {len(failures)}개를 학습에 섞음, 커리큘럼 {[STAGES[s] for s in stages]}")
    budget = args.budget_minutes
    X = dagger(model, budget * args.bc_frac, max(stages), failures, rng, log=log)
    if len(X):  # 이 후보가 학습한 관측 분포 (자율 모드의 '처음 보는 상황' 판단용). 승격되면 weights/obs_stats.npz 로 복사
        np.savez(os.path.join(W, f"cand_{k}_stats.npz"), mean=X.mean(0), std=X.std(0) + 1e-3)
    # 후보 ① BC/DAgger 직후, 후보 ② PPO 뒤 → 둘 다 벤치마크해서 더 나은 쪽을 이 사이클의 후보로
    cand_bc = os.path.join(W, f"cand_{k}_bc.zip")
    model.save(cand_bc)
    card_bc = evaluate(cand_bc); card_bc["bench"] = BENCH_VERSION
    log(f"  후보(BC/DAgger) 총점 {card_bc['total']} (도시 {card_bc['urban']}, 시골길 {card_bc['rural']})")
    cand, card, kind = cand_bc, card_bc, "bc"
    if budget * (1 - args.bc_frac) > 0.2:
        for e in model.get_env().envs:
            e.failures = failures
        ppo_finetune(model, budget * (1 - args.bc_frac) - 0.3, stages, log=log)
        cand_ppo = os.path.join(W, f"cand_{k}_ppo.zip")
        model.save(cand_ppo)
        card_ppo = evaluate(cand_ppo); card_ppo["bench"] = BENCH_VERSION
        log(f"  후보(BC+PPO) 총점 {card_ppo['total']} (도시 {card_ppo['urban']}, 시골길 {card_ppo['rural']})")
        key = "total" if args.rural else "urban"
        if card_ppo[key] > card_bc[key]:
            cand, card, kind = cand_ppo, card_ppo, "ppo"
    card["candidate"] = kind
    json.dump(card, open(os.path.join(R, f"scorecard_cycle{k}.json"), "w"), ensure_ascii=False, indent=1)
    return finalize(args, k, cand, card, best_card, t0, log)


def finalize(args, k, cand, card, best_card, t0, log=log_print):
    # 실패 사례 모으기
    from autodrive.bench import ALL_BENCH as BENCHMARK
    nf = 0
    with open(FAIL, "a", encoding="utf-8") as f:
        for name, r in card["scenarios"].items():
            if r["crashed"] or r["red_runs"] or r.get("right_turn_no_stop") or r["school_speeding_s"] > 0.5:
                spec = dict(BENCHMARK[name]); spec["fail_s"] = r.get("fail_s", 0.0)
                spec["mu"] = spec.get("mu", 0.9)
                f.write(json.dumps({"cycle": k, "scenario": name, "spec": spec,
                                    "why": {x: r[x] for x in ("crashed", "red_runs", "school_speeding_s")}},
                                   ensure_ascii=False) + "\n")
                nf += 1
    if best_card is None:
        ok, reasons = card["total"] > -1e9, ["첫 모델"]
    else:
        ok, reasons = compare(card, best_card, suite="all" if args.rural else "urban")
    if ok:
        shutil.copy(cand, BEST)
        st = os.path.join(W, f"cand_{k}_stats.npz")
        if os.path.exists(st):
            shutil.copy(st, STATS)
        json.dump(card, open(BEST_CARD, "w"), ensure_ascii=False, indent=1)
    log(scorecard_md(card, f"사이클 {k} 후보"))
    log(f"  승격: {'예' if ok else '아니오'} ({'; '.join(reasons) if reasons else '모든 조건 통과'}), 새 실패 사례 {nf}개")
    append_leaderboard(card_row(f"cycle{k}" + ("-rural" if args.rural else "") + f"({card.get('candidate', '')})", card, "yes" if ok else "no",
                                ("; ".join(reasons))[:120] if reasons else "통과", round((time.time() - t0) / 60, 1)))
    return card, ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-minutes", type=float, default=6.0, help="사이클 하나의 학습 시간(평가 제외)")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--bc-frac", type=float, default=0.6, help="예산 중 BC/DAgger 비율 (나머지는 PPO)")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rural", action="store_true", help="시골길 단계(4)까지 커리큘럼에 넣기 (도시 점수 회귀 금지 조건은 그대로)")
    ap.add_argument("--refresh-stats", action="store_true", help="관측 분포(weights/obs_stats.npz)만 다시 계산")
    a = ap.parse_args()
    if a.refresh_stats:
        rng = np.random.default_rng(a.seed)
        X = []
        for _ in range(60):
            spec = sample_train_spec(rng, 4 if a.rural else 3)
            env = make_env(spec, seed=int(rng.integers(1e9)), random_start=True)
            obs, _ = env.reset(seed=int(rng.integers(1e9)))
            ex = PurePursuitController(env)
            for _ in range(400):
                X.append(obs)
                obs, _, te, tr, _ = env.step(ex.act(obs))
                if te or tr:
                    break
        X = np.array(X, np.float32)
        os.makedirs(W, exist_ok=True)
        np.savez(STATS, mean=X.mean(0), std=X.std(0) + 1e-3)
        print(f"관측 분포 저장: {STATS} ({len(X)}개)")
        raise SystemExit
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    start = 1
    if os.path.exists(LB_CSV):
        with open(LB_CSV, encoding="utf-8") as f:
            start = 1 + sum(1 for r in csv.DictReader(f) if r["model"].startswith("cycle"))
    for k in range(start, start + a.cycles):
        cycle(a, k)
