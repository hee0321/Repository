"""시골길 사이클에서 도시 점수가 왜 떨어졌는지 숫자로 확인한다.

    python tools/diagnose_forgetting.py [모델.zip ...]

① 학습 데이터 구성(에피소드·스텝 비율)  ② 도시/시골 상태에서 기준 제어기 대비 흉내 오차와 최고 모델 대비 정책 변화
③ 도시 vs 시골길 관측 분포 차이(어떤 특징이 다른지)  → results/diagnosis_rural.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autodrive.bench import BENCHMARK, RURAL_BENCHMARK, ALL_BENCH  # noqa: E402
from autodrive.controllers import PurePursuitController  # noqa: E402
from autodrive.env import OBS_NAMES  # noqa: E402
from autodrive.scenarios import make_env, sample_train_spec  # noqa: E402


def suite_of(spec):
    from autodrive.scenarios import RURAL_OSM
    if spec.get("kind") == "rural" or spec.get("korea") in RURAL_OSM:
        return "rural"
    return "urban_korea" if "korea" in spec else "urban_synthetic"


def composition(sampler, stage, failures, n=120, seed=0, **kw):
    """sampler 로 n개 뽑아 기준 제어기로 달려 '에피소드 비율'과 '스텝 비율'을 잰다(학습 데이터는 스텝 단위)."""
    rng = np.random.default_rng(seed)
    ep, st = {}, {}
    for _ in range(n):
        spec = sampler(rng, stage, failures, **kw)
        env = make_env(spec, seed=int(rng.integers(1e9)), random_start=spec.get("start_s") is None)
        obs, _ = env.reset(seed=int(rng.integers(1e9)))
        ex = PurePursuitController(env)
        k, done = 0, False
        while not done and k < 900:      # dagger() 의 에피소드 상한과 같게
            obs, _, te, tr, _ = env.step(ex.act(obs)); done = te or tr; k += 1
        s = suite_of(spec)
        ep[s] = ep.get(s, 0) + 1
        st[s] = st.get(s, 0) + k
    tot_e, tot_s = sum(ep.values()), sum(st.values())
    return {s: {"episodes": round(ep[s] / tot_e, 3), "steps": round(st.get(s, 0) / tot_s, 3)} for s in sorted(ep)}


def expert_states(names, max_steps=900):
    X, Y = [], []
    for n in names:
        env = make_env(ALL_BENCH[n], seed=0)
        obs, _ = env.reset(seed=0)
        ex = PurePursuitController(env)
        done, k = False, 0
        while not done and k < max_steps:
            a = ex.act(obs); X.append(obs); Y.append(np.clip(a, -1, 1))
            obs, _, te, tr, _ = env.step(a); done = te or tr; k += 1
    return np.array(X, np.float32), np.array(Y, np.float32)


def means(model, X):
    import torch
    from improve import policy_mean
    with torch.no_grad():
        return policy_mean(model, torch.as_tensor(X)).numpy().clip(-1, 1)


def main(models):
    from stable_baselines3 import PPO
    out = {}
    fails = []
    if os.path.exists("results/failures.jsonl"):
        fails = [json.loads(x)["spec"] for x in open("results/failures.jsonl", encoding="utf-8") if x.strip()][-200:]
    out["composition_old_stage3"] = composition(sample_train_spec, 3, fails)
    out["composition_old_stage4"] = composition(sample_train_spec, 4, fails)
    try:
        from autodrive.scenarios import sample_mixed_spec
        out["composition_new_mixed"] = composition(lambda r, s, f: sample_mixed_spec(r, f, urban_ratio=0.5), 4, fails)
    except ImportError:
        pass
    Xu, Yu = expert_states(list(BENCHMARK))
    Xr, Yr = expert_states(list(RURAL_BENCHMARK))
    pols = {m: PPO.load(m, device="cpu") for m in models}
    ref = means(pols[models[0]], Xu)
    out["policy"] = {}
    for m, p in pols.items():
        mu, mr = means(p, Xu), means(p, Xr)
        out["policy"][os.path.basename(m)] = {
            "urban_bc_mse": round(float(((mu - Yu) ** 2).mean()), 4),
            "urban_steer_mse": round(float(((mu[:, 0] - Yu[:, 0]) ** 2).mean()), 4),
            "urban_accel_mse": round(float(((mu[:, 1] - Yu[:, 1]) ** 2).mean()), 4),
            "rural_bc_mse": round(float(((mr - Yr) ** 2).mean()), 4),
            "urban_shift_vs_first": round(float(((mu - ref) ** 2).mean()), 4),
        }
    # 관측 분포: 도시 vs 시골길, 표준화한 평균 차이가 큰 특징
    nl = Xu.shape[1] - len(OBS_NAMES)
    names = [f"lidar{i}" for i in range(nl)] + OBS_NAMES
    sd = np.concatenate([Xu, Xr]).std(0) + 1e-3
    gap = (Xr.mean(0) - Xu.mean(0)) / sd
    order = np.argsort(-np.abs(gap))[:8]
    out["obs_shift_rural_vs_urban"] = {names[i]: {"urban_mean": round(float(Xu[:, i].mean()), 3),
                                                  "rural_mean": round(float(Xr[:, i].mean()), 3),
                                                  "gap_sd": round(float(gap[i]), 2)} for i in order}
    if os.path.exists("weights/obs_stats.npz"):
        st = np.load("weights/obs_stats.npz")
        z = np.abs((Xr - st["mean"]) / st["std"]).max(1)
        zu = np.abs((Xu - st["mean"]) / st["std"]).max(1)
        out["ood_stats_file"] = {"rural_states_flagged(z>6)": round(float((z > 6).mean()), 3),
                                 "urban_states_flagged(z>6)": round(float((zu > 6).mean()), 3)}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/diagnosis_rural.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["weights/best.zip", "weights/cand_4_bc.zip", "weights/cand_5_bc.zip", "weights/cand_5_ppo.zip"])
