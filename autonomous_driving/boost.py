"""병렬 DAgger 부스트 사이클 (CPU 여러 개 사용). improve.py 사이클과 같은 관문·리더보드를 쓴다.

    python boost.py --minutes 5 --focus auto

① 최고 모델에서 시작 → ② 일꾼 프로세스 여러 개가 '모델이 운전(일부는 안전 실드 씌움)'하며 기준 제어기 정답을 모음
   (남은 실패 시나리오를 바꿔 가며 집중 연습 + 도시/시골 섞기) → ③ 데이터를 weights/dagger_buffer.npz 에 계속 쌓음
   (진짜 DAgger처럼 사이클이 지나도 모은 데이터 유지, git 무시) → ④ 중요 상태(신호·앞차·보행자·장애물) 가중 BC,
   평범한 도시 상태는 이전 최고 정책에 묶음(anchor) → ⑤ 짧은 PPO(낮은 학습률) + BC 정리 후보도 만듦 →
   ⑥ 병렬 벤치마크 → ⑦ improve.finalize (엄격한 안전 관문: 이탈·충돌·위반·위험 행동이 최고보다 많으면 절대 승격 안 함).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

W, R = "weights", "results"
BUF = os.path.join(W, "dagger_buffer.npz")


def _collect(args):
    model_path, seconds, seed, focus, focus_p, shield_p, beta = args
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from autodrive.controllers import PurePursuitController
    from autodrive.scenarios import make_env, sample_mixed_spec
    from autodrive.shield import SafetyShield
    from improve import critical_state, focus_spec, is_rural_spec, load_failures
    model = PPO.load(model_path, device="cpu")
    rng = np.random.default_rng(seed)
    failures = load_failures()
    X, Y, C, S = [], [], [], []
    t_end = time.time() + seconds
    while time.time() < t_end:
        if focus and rng.random() < focus_p:
            spec = focus_spec(rng, str(rng.choice(focus)))
        else:
            spec = sample_mixed_spec(rng, failures, 0.5)
        env = make_env(spec, seed=int(rng.integers(1e9)), random_start=spec.get("start_s") is None)
        obs, _ = env.reset(seed=int(rng.integers(1e9)))
        sh = SafetyShield(env) if rng.random() < shield_p else None
        ex = sh.exp if sh is not None else PurePursuitController(env)
        side = int(is_rural_spec(spec))
        done, n = False, 0
        while not done and n < 1500:
            a_m = model.predict(obs, deterministic=True)[0]
            if sh is not None:
                a = sh.filter(obs, a_m); a_exp = sh.last["a_exp"]
            else:
                a_exp = ex.act(obs)
                a = a_exp + rng.normal(0, 0.1, 2) if rng.random() < beta else a_m
            X.append(obs); Y.append(a_exp); C.append(critical_state(env, ex)); S.append(side)
            obs, _, te, tr, _ = env.step(a)
            done = te or tr
            n += 1
    return (np.array(X, np.float32), np.clip(np.array(Y, np.float32), -1, 1), np.array(C, np.float32), np.array(S, np.int8))


def _eval_one(args):
    path, name, shield = args
    import torch
    torch.set_num_threads(1)
    from autodrive.bench import evaluate
    return name, evaluate(path, scenarios=[name], shield=shield)["scenarios"][name]


def par_evaluate(path, pool, shield=False):
    from autodrive.bench import ALL_BENCH, BENCHMARK, RURAL_BENCHMARK
    res = dict(pool.map(_eval_one, [(path, n, shield) for n in ALL_BENCH]))
    card = {n: res[n] for n in ALL_BENCH}
    u = [card[k]["score"] for k in BENCHMARK]; r = [card[k]["score"] for k in RURAL_BENCHMARK]
    return {"total": round(float(np.mean([v["score"] for v in card.values()])), 2), "urban": round(float(np.mean(u)), 2),
            "rural": round(float(np.mean(r)), 2), "scenarios": card}


def train_bc(model, D, epochs, lr, crit_w, anchor=None, anchor_w=1.0, log=print):
    import torch
    from improve import policy_mean
    X, Y, C, S = (torch.as_tensor(D[k]) for k in ("X", "Y", "C", "S"))
    A = None
    if anchor is not None:
        with torch.no_grad():
            A = policy_mean(anchor, X).clamp(-1, 1)
    opt = torch.optim.Adam(list(model.policy.mlp_extractor.policy_net.parameters()) +
                           list(model.policy.action_net.parameters()), lr=lr)
    iu, ir = torch.nonzero(S == 0).flatten(), torch.nonzero(S == 1).flatten()
    steps = max(len(X) // 512, 1)
    for ep in range(epochs):
        tot = 0.0
        for _ in range(steps):
            b = torch.cat([iu[torch.randint(len(iu), (256,))], ir[torch.randint(len(ir), (256,))]])
            p = policy_mean(model, X[b])
            w = 1.0 + (crit_w - 1.0) * C[b]
            loss = (w[:, None] * (p - Y[b]) ** 2).mean() / w.mean()
            if A is not None:   # 평범한 '도시' 상태만 이전 최고 정책에 묶음
                m = ((1.0 - C[b]) * (S[b] == 0).float())[:, None]
                loss = loss + anchor_w * (m * (p - A[b]) ** 2).sum() / (m.sum() * 2 + 1e-6)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        log(f"    BC epoch {ep + 1}/{epochs}: 손실 {tot / steps:.4f} (데이터 {len(X)}개)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5.0, help="모으기+학습 시간 (평가 제외)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--focus", default="auto")
    ap.add_argument("--focus-p", type=float, default=0.4)
    ap.add_argument("--shield-p", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--crit-w", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--anchor-w", type=float, default=0.5)
    ap.add_argument("--buffer", type=int, default=900_000)
    ap.add_argument("--ppo-minutes", type=float, default=0.0, help=">0 이면 BC 뒤 짧은 PPO(낮은 학습률) + BC 정리 후보도 평가")
    ap.add_argument("--ppo-lr", type=float, default=1e-5)
    ap.add_argument("--init", default="best", help="best | 가중치 경로 (승격 판단은 언제나 최고 모델과 비교)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    from stable_baselines3 import PPO
    import improve
    from autodrive.scenarios import BENCH_VERSION
    focus = improve.FOCUS_DEFAULT if a.focus == "auto" else [x for x in a.focus.split(",") if x]
    best_card = json.load(open(improve.BEST_CARD, encoding="utf-8"))
    import csv
    k = 1
    for path in sorted(set([os.path.join(R, "leaderboard.csv"), improve.LB_CSV])):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                k += sum(1 for r in csv.DictReader(f) if r["model"].startswith("cycle") and "재평가" not in r["model"] and "재판단" not in r["model"])
    t0 = time.time()
    log = improve.log_print
    init = improve.BEST if a.init == "best" else a.init
    model = PPO.load(init, device="cpu")
    anchor = PPO.load(improve.BEST, device="cpu")
    for p_ in anchor.policy.parameters():
        p_.requires_grad_(False)
    log(f"[사이클 {k} 부스트] 시작 {init} (최고 {best_card['total']}/{best_card['urban']}/{best_card['rural']}), 일꾼 {a.workers}, 집중 {focus}")
    D = dict(np.load(BUF)) if os.path.exists(BUF) else None
    if D is not None:
        log(f"  쌓인 DAgger 데이터 {len(D['X'])}개 불러옴")
    t_end = t0 + a.minutes * 60
    rnd = 0
    tmp = os.path.join(W, f"_boost_cur.zip")
    with ProcessPoolExecutor(a.workers) as pool:
        while time.time() < t_end - 30:
            rnd += 1
            model.save(tmp)
            sec = max(20.0, min(90.0, (t_end - time.time()) * 0.5))
            outs = list(pool.map(_collect, [(tmp, sec, a.seed * 1000 + k * 100 + rnd * 10 + i, focus, a.focus_p, a.shield_p,
                                             a.beta if rnd == 1 else a.beta * 0.3) for i in range(a.workers)]))
            new = {key: np.concatenate([o[j] for o in outs]) for j, key in enumerate("XYCS")}
            D = new if D is None else {key: np.concatenate([D[key], new[key]])[-a.buffer:] for key in "XYCS"}
            log(f"  라운드 {rnd}: 새 데이터 {len(new['X'])}개 (중요 {int(new['C'].sum())}, 시골 {int(new['S'].sum())}) → 버퍼 {len(D['X'])}")
            train_bc(model, D, a.epochs, a.lr, a.crit_w, anchor, a.anchor_w, log)
        np.savez(BUF, **D)
        np.savez(os.path.join(W, f"cand_{k}_stats.npz"), mean=D["X"].mean(0), std=D["X"].std(0) + 1e-3)
        cands = []
        cand_bc = os.path.join(W, f"cand_{k}_bc.zip"); model.save(cand_bc)
        c = par_evaluate(cand_bc, pool); c["bench"] = BENCH_VERSION
        log(f"  후보(DAgger 부스트) 총점 {c['total']} (도시 {c['urban']}, 시골 {c['rural']})")
        cands.append((cand_bc, c, "bc"))
        if a.ppo_minutes > 0:
            from stable_baselines3.common.utils import get_schedule_fn
            model = PPO.load(cand_bc, env=improve.make_venv(3, improve.load_failures(), 4, a.seed + k, mixed=True), device="cpu")
            model.learning_rate = a.ppo_lr; model.lr_schedule = get_schedule_fn(a.ppo_lr)
            improve.ppo_finetune(model, a.ppo_minutes, [3, 4], log=log)
            train_bc(model, D, 1, a.lr * 0.5, a.crit_w, anchor, a.anchor_w, log)   # PPO 뒤 정리
            cand_ppo = os.path.join(W, f"cand_{k}_ppo.zip"); model.save(cand_ppo)
            c2 = par_evaluate(cand_ppo, pool); c2["bench"] = BENCH_VERSION
            log(f"  후보(부스트+PPO+정리) 총점 {c2['total']} (도시 {c2['urban']}, 시골 {c2['rural']})")
            cands.append((cand_ppo, c2, "ppo"))
    from autodrive.bench import compare, aggregate_dominates, safety_totals
    for p_, c_, kind in cands:
        log(f"   {kind}: {safety_totals(c_)}")
    ok = [x for x in cands if compare(x[1], best_card, suite="all")[0] or aggregate_dominates(x[1], best_card)[0]]
    cand, card, kind = max(ok or cands, key=lambda x: x[1]["total"])
    card["candidate"] = kind; card["from_cycle"] = f"cycle{k}"
    json.dump(card, open(os.path.join(R, f"scorecard_cycle{k}.json"), "w"), ensure_ascii=False, indent=1, default=float)

    class A_:
        rural, no_mix, urban_tol, gate = True, False, 1.0, "aggregate"
    improve.finalize(A_, k, cand, card, best_card, t0, log)


if __name__ == "__main__":
    main()
