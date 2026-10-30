"""벤치마크 v2.2 (18개 시나리오) 비교: 비전 정책(카메라+내비만, 실드 없음) vs 규칙 기반 expert vs rl-daily 최고 모델(라이다·특징 관측).

사용: python eval_vision.py --policy weights/vision_policy.pt --rl weights/rl_daily_best.zip --out results/vision_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _one(job):
    import torch
    torch.set_num_threads(1)
    kind, path, name = job
    from autodrive.bench import ALL_BENCH, REF_SPEED, make_policy, run_episode, scenario_score
    from autodrive.scenarios import make_env
    env = make_env(ALL_BENCH[name], seed=0)
    if kind == "vision":
        from autodrive.vision.drive import drive_episode
        from autodrive.vision.policy import load_policy
        r, _, _ = drive_episode(env, load_policy(path), seed=0)
    else:
        r = run_episode(env, make_policy(path), seed=0)
    r["score"] = scenario_score(r, REF_SPEED.get(name, 30))
    return kind, name, r


def summarize(card):
    from autodrive.bench import BENCHMARK, RURAL_BENCHMARK, safety_totals
    sc = card["scenarios"]
    card["total"] = round(float(np.mean([v["score"] for v in sc.values()])), 2)
    card["urban"] = round(float(np.mean([v["score"] for k, v in sc.items() if k in BENCHMARK])), 2)
    card["rural"] = round(float(np.mean([v["score"] for k, v in sc.items() if k in RURAL_BENCHMARK])), 2)
    card["safety"] = safety_totals(card)
    card["finished"] = sum(v["finished"] for v in sc.values())
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="weights/vision_policy.pt")
    ap.add_argument("--rl", default="weights/rl_daily_best.zip")
    ap.add_argument("--only", default="vision,expert,rl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/vision_benchmark.json")
    a = ap.parse_args()
    from multiprocessing import get_context
    from autodrive.bench import ALL_BENCH
    from autodrive.scenarios import BENCH_VERSION
    kinds = a.only.split(",")
    paths = {"vision": a.policy, "expert": "pure_pursuit", "rl": a.rl}
    jobs = [(k, paths[k], n) for k in kinds for n in ALL_BENCH]
    cards = {k: {"scenarios": {}} for k in kinds}
    with get_context("spawn").Pool(a.workers) as pool:
        for k, n, r in pool.imap_unordered(_one, jobs):
            cards[k]["scenarios"][n] = r
            print(f"{k:7s} {n:20s} score {r['score']:6.2f} crash {r['crashed']} red {r['red_runs']} rt {r.get('right_turn_no_stop', 0)}", flush=True)
    for k in kinds:
        cards[k]["scenarios"] = {n: cards[k]["scenarios"][n] for n in ALL_BENCH}
        summarize(cards[k])
    old = json.load(open(a.out)) if os.path.exists(a.out) else {}
    old.update(cards)
    old["bench"] = BENCH_VERSION
    json.dump(old, open(a.out, "w"), ensure_ascii=False, indent=1)
    for k in kinds:
        c = cards[k]
        print(k, c["total"], c["urban"], c["rural"], c["finished"], c["safety"])


if __name__ == "__main__":
    main()
