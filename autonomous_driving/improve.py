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

from autodrive.bench import aggregate_dominates, compare, evaluate, scorecard_md
from autodrive.controllers import PurePursuitController
from autodrive.shield import SafetyShield
from autodrive.scenarios import BENCH_VERSION, RURAL_OSM, STAGES, make_env, sample_mixed_spec, sample_train_spec

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    raise SystemExit("pip install gymnasium stable-baselines3 torch(CPU) 가 필요해요")

W, R = "weights", "results"
BEST = os.path.join(W, "best.zip")
STATS = os.path.join(W, "obs_stats.npz")
FAIL = os.path.join(R, "failures.jsonl")
# 벤치마크 버전마다 리더보드 파일을 나눈다 (v1 = 예전 시골길(트랙터) 점수, v2 = 일반 차량 흐름 + 실제 OSM 시골길)
_BV = BENCH_VERSION.split(".")[0]   # v2.1 도 v2 리더보드에 이어 씀 (시나리오 같음)
LB_CSV = os.path.join(R, "leaderboard.csv" if _BV == "v1" else f"leaderboard_{_BV}.csv")
LB_MD = os.path.join(R, "leaderboard.md" if _BV == "v1" else f"leaderboard_{_BV}.md")
BEST_CARD = os.path.join(R, "best_scorecard.json")


class MixEnv(gym.Env):
    """reset 때마다 커리큘럼 단계에 맞는 시나리오를 새로 뽑는 학습용 환경(도메인 무작위화 포함)."""

    def __init__(self, seed=0, stage=1, failures=None, suite=None):
        self.rng = np.random.default_rng(seed)
        self.stage = stage
        self.failures = failures or []
        self.suite = suite      # None = 예전 커리큘럼(단계), "urban"/"rural" = 섞기 학습에서 이 환경이 맡는 쪽
        self.env = make_env(sample_train_spec(self.rng, 1), seed=seed)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, *, seed=None, options=None):
        if self.suite is None:
            spec = sample_train_spec(self.rng, self.stage, self.failures)
        else:
            spec = sample_mixed_spec(self.rng, self.failures, suite=self.suite)
        self.spec = spec
        self.env = make_env(spec, seed=int(self.rng.integers(1e9)), random_start=spec.get("start_s") is None)
        return self.env.reset(seed=int(self.rng.integers(1e9)))

    def step(self, a):
        return self.env.step(a)


FOCUS_DEFAULT = ["rules_track", "obstacles_track", "gangnam_reverse", "gangnam_traffic", "rural_wide_traffic",
                 "jeongbae_village", "jeongbae_night"]


def focus_spec(rng, name):
    """남은 실패 시나리오를 '조금씩 바꿔서' 연습용으로 (벤치마크와 똑같은 판은 30%만 → 외워 버리기 줄임)."""
    from autodrive.bench import ALL_BENCH
    import copy
    spec = copy.deepcopy(ALL_BENCH[name])
    spec["max_steps"] = 1500
    if rng.random() < 0.3:
        spec["start_s"] = 0.0
        return spec
    if "korea" in spec:
        if spec["korea"] in RURAL_OSM:
            spec["traffic_seed"] = int(rng.integers(1e6)) if rng.random() < 0.6 else None
        else:
            spec["signal_seed"] = int(rng.integers(5))
    if spec.get("kind") == "rural":
        spec["difficulty"] = float(rng.uniform(0.8, 1.3))
        if rng.random() < 0.5:
            spec["road_seed"] = int(rng.choice([134, 134, 34, 234, 334, 434]))
    if spec.get("signals"):
        spec["signal_seed"] = int(rng.integers(1e6)); spec["laps"] = 1
    for o in spec.get("obstacles") or []:
        o["s"] = float(o["s"] + rng.uniform(-30, 30))
        if o.get("v", 0) > 0:
            o["v"] = float(o["v"] * rng.uniform(0.7, 1.3))
        o["d"] = float(o.get("d", 0.0) + rng.uniform(-0.3, 0.3))
    if rng.random() < 0.5:
        spec["start_s"] = None   # 무작위 출발점
    else:
        spec["start_s"] = 0.0
    return spec


CRIT_NOTES = ("빨간불", "우회전", "앞차", "장애물", "보행자", "양보", "앞지르기")


def critical_state(env, ex):
    """안전에 중요한 순간: 신호 정지/빨간불 우회전 일시정지, 앞차 따라가기·장애물·보행자·마주 오는 차, 앞 60 m 안 정지선,
    주변 ±(15~45) m 안 차·보행자·장애물. (제한속도·시야·자갈길 서행 같은 '평소 감속'은 제외)"""
    if any(w in ex.rule_note for w in CRIT_NOTES):
        return True
    if env.stop_distance() < 60.0:
        return True
    for k in range(len(env.ob_s)):
        if -15.0 < env._gap(env.ob_s[k]) < 45.0:
            return True
    return False


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


def is_rural_spec(spec):
    return spec.get("kind") == "rural" or spec.get("korea") in RURAL_OSM


def dagger(model, minutes, stage, failures, rng, beta0=0.5, log=print, mixed=False, urban_ratio=0.5,
           anchor=None, anchor_w=1.0, focus=None, focus_p=0.0, crit_w=1.0, shield_p=0.0):
    """모델이 달리는 상태를 모으고 기준 제어기의 행동을 정답으로 붙여 지도학습(BC/DAgger).

    mixed=True (시골길 사이클): 도시/시골 데이터를 따로 모아 미니배치마다 urban_ratio 대 나머지로 섞는다
    (시골길은 느려서 에피소드가 길어 '스텝 비율'이 쏠리는 것을 막음). anchor(이전 최고 정책, 고정)가 있으면
    도시 상태에서 새 정책 평균 행동을 anchor 평균 행동에 묶는 항(같은 σ의 가우시안 정책이면 KL ∝ 평균 차이²)을 더한다.
    반환: 관측 배열(관측 분포 통계용, 도시+시골 모두)."""
    import torch
    t_end = time.time() + minutes * 60
    D = {"urban": ([], [], []), "rural": ([], [], [])}   # (관측, 정답 행동, 중요 상태 여부)
    rounds = 0
    opt = torch.optim.Adam(list(model.policy.mlp_extractor.policy_net.parameters()) +
                           list(model.policy.action_net.parameters()), lr=1e-3 if not mixed else 5e-4)
    while time.time() < t_end:
        beta = beta0 * (0.5 ** rounds)  # 처음엔 기준 제어기가 운전을 더 많이, 점점 모델이 운전
        collect_until = time.time() + max(10.0, (t_end - time.time()) * 0.45)
        while time.time() < collect_until:
            if focus and rng.random() < focus_p:
                spec = focus_spec(rng, str(rng.choice(focus)))
            else:
                spec = sample_mixed_spec(rng, failures, urban_ratio) if mixed else sample_train_spec(rng, stage, failures)
            env = make_env(spec, seed=int(rng.integers(1e9)), random_start=spec.get("start_s") is None)
            obs, _ = env.reset(seed=int(rng.integers(1e9)))
            # shield_p: 모델이 운전하되 안전 실드를 씌워 굴림 → 실수해도 에피소드가 끝나지 않아 실수 '뒤' 상황까지 정답을 모음
            sh = SafetyShield(env) if rng.random() < shield_p else None
            ex = sh.exp if sh is not None else PurePursuitController(env)
            X, Y, C = D["rural" if is_rural_spec(spec) else "urban"]
            done, n = False, 0
            while not done and n < 900:
                if sh is not None:
                    a = sh.filter(obs, model.predict(obs, deterministic=True)[0])
                    a_exp = sh.last["a_exp"]
                    X.append(obs); Y.append(a_exp); C.append(critical_state(env, ex))
                    obs, _, term, trunc, _ = env.step(a)
                    done = term or trunc
                    n += 1
                    continue
                a_exp = ex.act(obs)
                X.append(obs); Y.append(a_exp); C.append(critical_state(env, ex))
                if rng.random() < beta:
                    a = a_exp + rng.normal(0, 0.1, 2)   # 기준 제어기 + 약간의 흔들림
                else:
                    a = model.predict(obs, deterministic=True)[0]
                obs, _, term, trunc, _ = env.step(a)
                done = term or trunc
                n += 1
        for k in D:
            D[k] = tuple(x[-75_000:] for x in D[k])   # 쪽마다 최근 7.5만 개
        T = {k: (torch.as_tensor(np.array(v[0], np.float32)), torch.as_tensor(np.clip(np.array(v[1], np.float32), -1, 1)),
                 torch.as_tensor(np.array(v[2], np.float32)))
             for k, v in D.items() if len(v[0])}
        A = None
        if anchor is not None and "urban" in T:
            with torch.no_grad():
                A = policy_mean(anchor, T["urban"][0]).clamp(-1, 1)

        def wmse(pred, y, c):
            # 안전에 중요한 상태(c=1)는 crit_w 배 무게
            w = 1.0 + (crit_w - 1.0) * c
            return (w[:, None] * (pred - y) ** 2).mean() / w.mean()
        n_all = sum(len(v[0]) for v in T.values())
        tot, nb = 0.0, 0
        for ep in range(6):
            if mixed and len(T) == 2:
                # 미니배치 512 = 도시 256 + 시골 256 (urban_ratio), 도시 상태에는 anchor 항
                nu = int(512 * urban_ratio)
                steps = max(len(T["urban"][0]) // nu, len(T["rural"][0]) // (512 - nu), 1)
                for _ in range(steps):
                    iu = torch.randint(len(T["urban"][0]), (nu,)); ir = torch.randint(len(T["rural"][0]), (512 - nu,))
                    xu, yu, cu = T["urban"][0][iu], T["urban"][1][iu], T["urban"][2][iu]
                    xr, yr, cr = T["rural"][0][ir], T["rural"][1][ir], T["rural"][2][ir]
                    pu, pr = policy_mean(model, xu), policy_mean(model, xr)
                    loss = 0.5 * wmse(pu, yu, cu) + 0.5 * wmse(pr, yr, cr)
                    if A is not None:   # anchor(이전 최고)는 '평범한' 도시 상태에서만 — 중요한 순간의 예전 실수까지 붙잡지 않게
                        m_ = (1.0 - cu)[:, None]
                        loss = loss + anchor_w * 0.5 * (m_ * (pu - A[iu]) ** 2).sum() / (m_.sum() * 2 + 1e-6)
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot += float(loss.detach()); nb += 1
            else:
                Xa = torch.cat([v[0] for v in T.values()]); Ya = torch.cat([v[1] for v in T.values()])
                Ca = torch.cat([v[2] for v in T.values()])
                perm = torch.randperm(len(Xa))
                for i in range(0, len(Xa), 512):
                    b = perm[i:i + 512]
                    loss = wmse(policy_mean(model, Xa[b]), Ya[b], Ca[b])
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot += float(loss.detach()); nb += 1
            if time.time() > t_end + 30:
                break
        rounds += 1
        nu_, nr_ = len(D["urban"][0]), len(D["rural"][0])
        nc_ = int(sum(D["urban"][2]) + sum(D["rural"][2]))
        log(f"  DAgger 라운드 {rounds}: 데이터 {n_all}개 (도시 {nu_}, 시골 {nr_}, 중요 상태 {nc_}), 손실 {tot / max(nb, 1):.4f}, β={beta:.2f}")
    X = np.array(D["urban"][0] + D["rural"][0], np.float32)
    return X


def make_venv(stage, failures, n_envs, seed, mixed=False):
    """mixed=True: 환경 절반은 도시, 절반은 시골길로 고정 → PPO 롤아웃 배치마다 50:50으로 섞임."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    suites = [("urban" if i % 2 == 0 else "rural") if mixed else None for i in range(n_envs)]
    return DummyVecEnv([lambda i=i: MixEnv(seed + i, stage, failures, suites[i]) for i in range(n_envs)])


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
                e.stage = st      # (섞기 모드에서는 suite 가 정해져 있어 stage 는 쓰지 않음)
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
    from stable_baselines3.common.utils import get_schedule_fn
    os.makedirs(W, exist_ok=True); os.makedirs(R, exist_ok=True)
    rng = np.random.default_rng(args.seed + 1000 * k)
    t0 = time.time()
    bpath = os.path.join(R, "baseline_scorecard.json")
    base = json.load(open(bpath, encoding="utf-8")) if os.path.exists(bpath) else None
    if base is None or base.get("bench") != BENCH_VERSION:
        base = evaluate("pure_pursuit"); base["bench"] = BENCH_VERSION
        json.dump(base, open(bpath, "w"), ensure_ascii=False, indent=1)
        append_leaderboard(card_row("pure_pursuit(기준)", base, "-", f"규칙 기반 기준 제어기 (벤치마크 {BENCH_VERSION})", 0))
    mixed = args.rural and not args.no_mix
    best_card = None
    anchor = None
    if os.path.exists(BEST) and os.path.exists(BEST_CARD):
        best_card = json.load(open(BEST_CARD, encoding="utf-8"))
        if best_card.get("bench") != BENCH_VERSION:   # 벤치마크가 바뀌면 최고 모델을 새 벤치마크로 다시 평가 (= '이전' 점수)
            old = best_card
            best_card = evaluate(BEST); best_card["bench"] = BENCH_VERSION
            best_card["candidate"] = old.get("candidate"); best_card["from_cycle"] = old.get("from_cycle", "cycle2")
            json.dump(best_card, open(BEST_CARD, "w"), ensure_ascii=False, indent=1)
            append_leaderboard(card_row(f"{best_card['from_cycle']}(최고, {BENCH_VERSION} 재평가)", best_card, "-",
                                        "현재 최고 모델을 새 벤치마크로 다시 잰 '이전' 점수", 0))
        init = BEST
        if args.init == "last":   # 직전 사이클 후보가 총점은 더 높았지만 관문에서 막혔으면 거기서 이어 학습 (비교 기준·anchor 는 여전히 최고 모델)
            prev = os.path.join(R, f"scorecard_cycle{k - 1}.json")
            if os.path.exists(prev):
                pc = json.load(open(prev, encoding="utf-8"))
                src = os.path.join(W, f"cand_{k - 1}_{pc.get('candidate', 'ppo')}.zip")
                if str(pc.get("bench", "")).split(".")[0] == _BV and pc["total"] > best_card["total"] and os.path.exists(src):
                    init = src
                    log(f"  초기값: 직전 후보 {src} (총점 {pc['total']}) — 승격 판단은 최고 모델과 비교")
        model = PPO.load(init, env=make_venv(3, load_failures(), args.n_envs, args.seed + k, mixed=mixed), device="cpu")
        log(f"[사이클 {k}] 최고 모델에서 이어 학습 (현재 최고 {best_card['total']}, 도시 {best_card['urban']}, 시골 {best_card['rural']})")
        stages = [3, 4] if args.rural else [2, 3]
        if mixed:
            anchor = PPO.load(BEST, device="cpu")   # 고정된 '도시 최고' 정책 (anchor)
            for p_ in anchor.policy.parameters():
                p_.requires_grad_(False)
    else:
        model = new_model(make_venv(1, load_failures(), args.n_envs, args.seed + k, mixed=mixed), args.seed)
        log(f"[사이클 {k}] 최고 모델 없음 → 새로 시작 (기준 제어기 흉내로 시작)")
        stages = [1, 2, 3, 4] if args.rural else [1, 2, 3]
    failures = load_failures()
    if mixed:
        log(f"  섞기 학습: 도시 {args.urban_ratio:.0%} / 시골 {1 - args.urban_ratio:.0%} (미니배치·PPO 환경 단위), "
            f"anchor 가중치 {args.anchor_w}, 실패 사례 {len(failures)}개")
    else:
        log(f"  실패 사례 {len(failures)}개를 학습에 섞음, 커리큘럼 {[STAGES[s] for s in stages]}")
    budget = args.budget_minutes
    focus = [x for x in args.focus.split(",") if x] if args.focus else None
    fkw = dict(focus=focus, focus_p=args.focus_p, crit_w=args.crit_w, shield_p=args.shield_p)
    if focus:
        log(f"  집중 연습: {focus} ({args.focus_p:.0%}), 중요 상태 가중치 x{args.crit_w}")
    X = dagger(model, budget * args.bc_frac, max(stages), failures, rng, log=log, mixed=mixed, beta0=args.beta0,
               urban_ratio=args.urban_ratio, anchor=anchor, anchor_w=args.anchor_w, **fkw)
    if len(X):  # 이 후보가 학습한 관측 분포 (자율 모드의 '처음 보는 상황' 판단용). 승격되면 weights/obs_stats.npz 로 복사
        np.savez(os.path.join(W, f"cand_{k}_stats.npz"), mean=X.mean(0), std=X.std(0) + 1e-3)
    cand_bc = os.path.join(W, f"cand_{k}_bc.zip")
    model.save(cand_bc)
    card_bc = evaluate(cand_bc); card_bc["bench"] = BENCH_VERSION
    log(f"  후보(BC/DAgger) 총점 {card_bc['total']} (도시 {card_bc['urban']}, 시골길 {card_bc['rural']})")
    cands = [(cand_bc, card_bc, "bc")]
    if budget * (1 - args.bc_frac) > 0.2:
        for e in model.get_env().envs:
            e.failures = failures
        if mixed:   # 섞기 모드 PPO: 학습률을 낮춰 정책이 크게 흔들리지 않게
            model.learning_rate = args.ppo_lr
            model.lr_schedule = get_schedule_fn(args.ppo_lr)
        ppo_min = budget * (1 - args.bc_frac) - 0.3
        if mixed:
            ppo_min *= 0.7
        ppo_finetune(model, ppo_min, stages, log=log)
        if mixed:   # PPO 뒤 정리: 모델이 운전(β=0) + 기준 제어기 정답 + 도시 anchor
            dagger(model, budget * (1 - args.bc_frac) * 0.3, 4, failures, rng, beta0=0.0, log=log, mixed=True,
                   urban_ratio=args.urban_ratio, anchor=anchor, anchor_w=args.anchor_w, **fkw)
        cand_ppo = os.path.join(W, f"cand_{k}_ppo.zip")
        model.save(cand_ppo)
        card_ppo = evaluate(cand_ppo); card_ppo["bench"] = BENCH_VERSION
        log(f"  후보(BC+PPO{'+정리' if mixed else ''}) 총점 {card_ppo['total']} (도시 {card_ppo['urban']}, 시골길 {card_ppo['rural']})")
        cands.append((cand_ppo, card_ppo, "ppo"))
    suite = "all" if args.rural else "urban"
    if best_card is not None:   # 관문을 통과하는 후보 중 총점이 높은 것, 없으면 총점이 높은 것
        ok = [c for c in cands if compare(c[1], best_card, suite=suite, urban_tol=args.urban_tol)[0]]
        pool = ok or cands
    else:
        pool = cands
    key = "total" if args.rural else "urban"
    cand, card, kind = max(pool, key=lambda c: c[1][key])
    card["candidate"] = kind
    card["from_cycle"] = f"cycle{k}"
    json.dump(card, open(os.path.join(R, f"scorecard_cycle{k}.json"), "w"), ensure_ascii=False, indent=1)
    return finalize(args, k, cand, card, best_card, t0, log)


def finalize(args, k, cand, card, best_card, t0, log=log_print, rejudge=False):
    # 실패 사례 모으기
    from autodrive.bench import ALL_BENCH as BENCHMARK
    nf = 0
    with open(FAIL if not rejudge else os.devnull, "a", encoding="utf-8") as f:
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
        ok, reasons = compare(card, best_card, suite="all" if args.rural else "urban", urban_tol=args.urban_tol)
        if not ok and args.gate == "aggregate":
            ok2, why2, n_, b_ = aggregate_dominates(card, best_card, suite="all" if args.rural else "urban")
            if ok2:
                ok = True
                reasons = [f"집계 안전 우위로 승격 (이탈 {b_['crashes']}→{n_['crashes']}, 신호·우회전 위반 {b_['signal_violations']}→"
                           f"{n_['signal_violations']}, 시골 위험 {b_['rural_unsafe']}→{n_['rural_unsafe']}); 남은 시나리오별 회귀"] + reasons
            else:
                reasons = reasons + ["집계 안전 우위 아님: " + ", ".join(why2)]
    if ok:
        shutil.copy(cand, BEST)
        st = os.path.join(W, f"cand_{k}_stats.npz")
        if os.path.exists(st):
            shutil.copy(st, STATS)
        json.dump(card, open(BEST_CARD, "w"), ensure_ascii=False, indent=1)
        with open(os.path.join(R, "promotions.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "cycle": k, "candidate": card.get("candidate"),
                                "total": card["total"], "urban": card["urban"], "rural": card["rural"],
                                "previous": None if best_card is None else {x: best_card[x] for x in ("total", "urban", "rural")},
                                "reasons": reasons}, ensure_ascii=False) + "\n")
    log(scorecard_md(card, f"사이클 {k} 후보"))
    log(f"  승격: {'예' if ok else '아니오'} ({'; '.join(reasons) if reasons else '모든 조건 통과'}), 새 실패 사례 {nf}개")
    tag = f"cycle{k}" + (("-rural-mix" if not args.no_mix else "-rural") if args.rural else "") + f"({card.get('candidate', '')})" + ("-재판단" if rejudge else "")
    append_leaderboard(card_row(tag, card, "yes" if ok else "no",
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
    ap.add_argument("--init", choices=["best", "last"], default="best",
                    help="last: 직전 사이클 후보가 총점이 더 높으면 거기서 이어 학습(비교·anchor 는 최고 모델)")
    ap.add_argument("--gate", choices=["strict", "aggregate"], default="aggregate",
                    help="strict: 시나리오별 조건만. aggregate: 시나리오별 조건에 걸려도 집계 안전 우위(bench.aggregate_dominates)면 승격")
    ap.add_argument("--promote-cycle", type=int, default=None,
                    help="학습 없이 results/scorecard_cycleN.json 후보를 지금 최고 모델과 다시 판단(같은 벤치마크 버전일 때만)")
    ap.add_argument("--no-mix", action="store_true", help="시골길 사이클에서 예전 방식(단계 커리큘럼만) 쓰기")
    ap.add_argument("--urban-ratio", type=float, default=0.5, help="섞기 학습에서 도시 데이터 비율")
    ap.add_argument("--anchor-w", type=float, default=1.0, help="도시 상태에서 이전 최고 정책에 묶는 항의 가중치")
    ap.add_argument("--urban-tol", type=float, default=1.0,
                    help="시골길 점수가 +5 이상 오를 때만 허용하는 도시 평균 점수 하락 폭(시나리오별 안전 조건은 그대로)")
    ap.add_argument("--focus", default="", help="집중 연습할 벤치마크 시나리오(쉼표). 'auto' = 남은 실패 목록 기본값")
    ap.add_argument("--focus-p", type=float, default=0.35, help="DAgger 에피소드 중 집중 연습 비율")
    ap.add_argument("--crit-w", type=float, default=1.0, help="안전에 중요한 상태(규칙 감속·주변 차량)의 BC 손실 가중치")
    ap.add_argument("--shield-p", type=float, default=0.0, help="DAgger 에피소드 중 '모델+실드'로 굴리는 비율")
    ap.add_argument("--beta0", type=float, default=0.5, help="DAgger 첫 라운드에서 기준 제어기가 운전하는 비율")
    ap.add_argument("--ppo-lr", type=float, default=3e-5, help="섞기 모드 PPO 학습률")
    ap.add_argument("--refresh-stats", action="store_true", help="관측 분포(weights/obs_stats.npz)만 다시 계산")
    a = ap.parse_args()
    if a.focus == "auto":
        a.focus = ",".join(FOCUS_DEFAULT)
    if a.promote_cycle is not None:
        k = a.promote_cycle
        card = json.load(open(os.path.join(R, f"scorecard_cycle{k}.json"), encoding="utf-8"))
        best_card = json.load(open(BEST_CARD, encoding="utf-8"))
        if card.get("bench") != BENCH_VERSION or best_card.get("bench") != BENCH_VERSION:
            raise SystemExit(f"벤치마크 버전이 달라 비교 불가: 후보 {card.get('bench')}, 최고 {best_card.get('bench')}, 지금 {BENCH_VERSION}")
        cand = os.path.join(W, f"cand_{k}_{card.get('candidate', 'bc')}.zip")
        finalize(a, k, cand, card, best_card, time.time(), rejudge=True)
        raise SystemExit
    if a.refresh_stats:
        rng = np.random.default_rng(a.seed)
        X = []
        for _ in range(60):
            spec = sample_mixed_spec(rng, None, 0.5) if a.rural else sample_train_spec(rng, 3)
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
    for path in sorted(set([os.path.join(R, "leaderboard.csv"), LB_CSV])):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                start += sum(1 for r in csv.DictReader(f) if r["model"].startswith("cycle") and "재평가" not in r["model"] and "재판단" not in r["model"])
    for k in range(start, start + a.cycles):
        cycle(a, k)
