"""SUMO + PPO: 한국 규칙을 지키며 실제 강남역·대치동 교통 속을 달리는 법 배우기 (CPU).

    python -m sumo_kr.train --budget-minutes 8         # BC(DAgger) → PPO → 평가 → 승격 판단
    python -m sumo_kr.train --eval-only                # 기준 운전자/현재 최고 모델 점수표만
가중치: weights/sumo_best.zip (git 무시). 점수: results/sumo_leaderboard.md
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

from .env import OBS_DIM, SCENARIOS, SumoKREnv

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    raise SystemExit("pip install gymnasium stable-baselines3")

W, R = "weights", "results"
BEST = os.path.join(W, "sumo_best.zip")
BEST_CARD = os.path.join(R, "sumo_best_scorecard.json")
LB = os.path.join(R, "sumo_leaderboard.csv")
EVAL_SEEDS = (101, 202)


class MultiScenario(gym.Env):
    """reset마다 시나리오를 바꿔 가며 학습 (프로세스 하나에 SUMO 하나)."""

    def __init__(self, seed=0, scenarios=None):
        self.rng = np.random.default_rng(seed)
        self.names = list(scenarios or SCENARIOS)
        self.envs = {}
        self.cur = None
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        if self.cur is not None:
            self.cur.close()
        n = self.names[int(self.rng.integers(len(self.names)))]
        if n not in self.envs:
            self.envs[n] = SumoKREnv(n, max_steps=500)
        self.cur = self.envs[n]
        return self.cur.reset(seed=int(self.rng.integers(1e6)))

    def step(self, a):
        return self.cur.step(a)


def run(env, act, seed):
    obs, _ = env.reset(seed=seed)
    done, n, speeds = False, 0, []
    while not done:
        obs, r, te, tr, info = env.step(act(env, obs))
        n += 1
        if env._ego_ok():
            speeds.append(env._cache["v"])
        done = te or tr
    st = dict(info["stats"])
    st.update(completion=round(min(1.0, st["dist_m"] / env.route_len), 3), arrived=info["reason"] == "arrived",
              time_s=round(n * 0.5, 1), mean_speed_kmh=round(float(np.mean(speeds or [0])) * 3.6, 1))
    return st


REF_KMH = {"gangnam": 25, "gangnam_reverse": 25, "daechi_school": 20}


def score(r, ref):
    eff = min(1.0, r["mean_speed_kmh"] / ref)
    return round(70 * r["completion"] + 15 * eff * r["completion"] - 40 * (r["collisions"] > 0) - 15 * r["red_runs"]
                 - 10 * r["right_turn_no_stop"] - 1.0 * r["speeding_s"] - 3.0 * r["school_speeding_s"]
                 - 0.5 * r["harsh_brake"] + 15, 2)


def evaluate(policy):
    if policy == "expert":
        act = lambda env, obs: env.expert_action()
    else:
        from stable_baselines3 import PPO
        m = PPO.load(policy, device="cpu") if isinstance(policy, str) else policy
        act = lambda env, obs: m.predict(obs, deterministic=True)[0]
    card = {}
    for n in SCENARIOS:
        env = SumoKREnv(n)
        for sd in EVAL_SEEDS:
            r = run(env, act, sd)
            r["score"] = score(r, REF_KMH[n])
            card[f"{n}@{sd}"] = r
        env.close()
    return {"total": round(float(np.mean([v["score"] for v in card.values()])), 2), "scenarios": card}


def compare(new, best):
    bad = []
    for k, b in best["scenarios"].items():
        n = new["scenarios"][k]
        if n["collisions"] > b["collisions"]:
            bad.append(f"{k}: 충돌 증가")
        if n["red_runs"] + n["right_turn_no_stop"] > b["red_runs"] + b["right_turn_no_stop"]:
            bad.append(f"{k}: 신호/우회전 위반 증가")
        if n["school_speeding_s"] > b["school_speeding_s"] + 1.0:
            bad.append(f"{k}: 어린이보호구역 과속 증가")
        if n["completion"] < b["completion"] - 0.05:
            bad.append(f"{k}: 완주율 하락")
    if new["total"] < best["total"] + 0.5:
        bad.append(f"총점 부족 {best['total']}→{new['total']}")
    return not bad, bad


def summarize(card):
    sc = card["scenarios"].values()
    return {"total": card["total"], "arrived": f"{sum(v['arrived'] for v in sc)}/{len(card['scenarios'])}",
            "collisions": sum(v["collisions"] for v in sc), "red_runs": sum(v["red_runs"] for v in sc),
            "right_turn_no_stop": sum(v["right_turn_no_stop"] for v in sc),
            "speeding_s": round(sum(v["speeding_s"] for v in sc), 1),
            "school_speeding_s": round(sum(v["school_speeding_s"] for v in sc), 1),
            "harsh_brake": sum(v["harsh_brake"] for v in sc),
            "mean_speed_kmh": round(float(np.mean([v["mean_speed_kmh"] for v in sc])), 1)}


def log_row(model, card, promoted, note):
    os.makedirs(R, exist_ok=True)
    row = {"time": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "model": model, **summarize(card),
           "promoted": promoted, "note": note[:120]}
    new = not os.path.exists(LB)
    with open(LB, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    rows = list(csv.DictReader(open(LB, encoding="utf-8")))
    cols = list(rows[0])
    md = "# SUMO + PPO 리더보드 (강남역·역방향 우회전·대치동 어린이보호구역, 시드 2개씩)\n\n"
    md += "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    md += "".join("| " + " | ".join(r[c] for c in cols) + " |\n" for r in rows)
    open(os.path.join(R, "sumo_leaderboard.md"), "w", encoding="utf-8").write(md)


def bc_dagger(model, minutes, rng, log):
    import torch
    t_end = time.time() + minutes * 60
    X, Y = [], []
    pol = model.policy
    opt = torch.optim.Adam(list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters()), lr=1e-3)
    envs = {n: SumoKREnv(n, max_steps=500) for n in SCENARIOS}
    rnd = 0
    while time.time() < t_end:
        beta = 0.5 ** rnd
        for _ in range(4):
            n = list(SCENARIOS)[int(rng.integers(len(SCENARIOS)))]
            env = envs[n]
            obs, _ = env.reset(seed=int(rng.integers(1e6)))
            done = False
            while not done:
                ae = env.expert_action()
                X.append(obs); Y.append(ae)
                a = ae + rng.normal(0, 0.15, 2) if rng.random() < beta else model.predict(obs, deterministic=True)[0]
                obs, _, te, tr, _ = env.step(a)
                done = te or tr
            env.close()
        Xa = torch.as_tensor(np.array(X, np.float32)); Ya = torch.as_tensor(np.array(Y, np.float32))
        for _ in range(30):
            perm = torch.randperm(len(Xa))
            for i in range(0, len(Xa), 256):
                b = perm[i:i + 256]
                feats = pol.extract_features(Xa[b], pol.pi_features_extractor)
                loss = ((pol.action_net(pol.mlp_extractor.forward_actor(feats)) - Ya[b]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        rnd += 1
        log(f"  DAgger {rnd}: 데이터 {len(Xa)}, MSE {float(loss.detach()):.4f}, β={beta:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-minutes", type=float, default=8.0)
    ap.add_argument("--bc-frac", type=float, default=0.4)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    log = lambda *x: print(*x, flush=True)
    os.makedirs(W, exist_ok=True); os.makedirs(R, exist_ok=True)
    if not os.path.exists(os.path.join(R, "sumo_expert_scorecard.json")):
        ex = evaluate("expert")
        json.dump(ex, open(os.path.join(R, "sumo_expert_scorecard.json"), "w"), ensure_ascii=False, indent=1)
        log_row("기준 운전자(규칙)", ex, "-", "제한속도 미리 보기·신호·우회전 일시정지")
        log("기준 운전자:", summarize(ex))
    if a.eval_only:
        if os.path.exists(BEST):
            log("최고 모델:", summarize(evaluate(BEST)))
        return
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv
    venv = SubprocVecEnv([lambda i=i: MultiScenario(1000 + i + int(time.time()) % 1000) for i in range(a.n_envs)])
    rng = np.random.default_rng(int(time.time()))
    if os.path.exists(BEST):
        model = PPO.load(BEST, env=venv, device="cpu")
        best = json.load(open(BEST_CARD, encoding="utf-8"))
        log(f"최고 모델에서 이어 학습 (총점 {best['total']})")
    else:
        model = PPO("MlpPolicy", venv, n_steps=256, batch_size=256, learning_rate=1e-4, clip_range=0.1, n_epochs=5,
                    gamma=0.98, policy_kwargs=dict(net_arch=[64, 64], log_std_init=-1.2), device="cpu", verbose=0)
        best = None
        log("새 모델: 기준 운전자 흉내(BC/DAgger)로 시작")
    bc_dagger(model, a.budget_minutes * a.bc_frac, rng, log)
    model.save(os.path.join(W, "sumo_cand_bc.zip"))
    c_bc = evaluate(os.path.join(W, "sumo_cand_bc.zip"))
    log("후보(BC):", summarize(c_bc))
    t0 = time.time()
    total = a.budget_minutes * (1 - a.bc_frac) * 60

    class Clock(BaseCallback):
        def _on_step(self):
            return time.time() - t0 < total
    model.learn(10_000_000, callback=Clock(), reset_num_timesteps=False)
    venv.close()
    model.save(os.path.join(W, "sumo_cand_ppo.zip"))
    c_ppo = evaluate(os.path.join(W, "sumo_cand_ppo.zip"))
    log(f"후보(BC+PPO, {model.num_timesteps} 스텝):", summarize(c_ppo))
    for tag, card in (("BC(DAgger)", c_bc), ("BC+PPO", c_ppo)):
        if best is None:
            ok, why = True, ["첫 모델"]
        else:
            ok, why = compare(card, best)
        log_row(tag, card, "yes" if ok else "no", "; ".join(why) if why else "통과")
        if ok:
            src = os.path.join(W, "sumo_cand_bc.zip" if tag.startswith("BC(") else "sumo_cand_ppo.zip")
            shutil.copy(src, BEST)
            json.dump(card, open(BEST_CARD, "w"), ensure_ascii=False, indent=1)
            best = card
        log(f"  {tag} 승격: {'예' if ok else '아니오'} ({'; '.join(why)})")


if __name__ == "__main__":
    main()
