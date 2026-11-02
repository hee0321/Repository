"""벤치마크 v2.2 (18개 시나리오) 비교: 비전 정책(카메라+내비만, 실드 없음) vs 규칙 기반 expert vs rl-daily 최고 모델(라이다·특징 관측).

사용: python eval_vision.py --policy weights/vision_policy.pt --rl weights/rl_daily_best.zip --out results/vision_benchmark.json
      python eval_vision.py --suite dcas --out results/dcas_benchmark.json   # DCAS 01 (UN R171 01) 20개
자율 완주(autonomous) = 도착 + 충돌·이탈 0 + 개입(takeover) 0. 실드를 안 쓰므로 개입은 늘 0, 도착·무사고가 관건.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _one(job):
    import torch
    torch.set_num_threads(1)
    kind, path, name, suite = job
    from autodrive.bench import ALL_BENCH, REF_SPEED, make_policy, run_episode, scenario_score
    from autodrive import dcas
    if suite == "dcas":
        env = dcas.make_env(name, seed=0)
        if kind == "vision":
            from autodrive.vision.drive import drive_episode
            from autodrive.vision.policy import load_policy
            env.monitor = dcas.DcasMonitor()
            r, _, _ = drive_episode(env, load_policy(path), seed=0)
            r.update(env.monitor.results(env, r, env.dcas))
        else:
            r = dcas.run(env, make_policy(path), seed=0)
        r["takeovers"] = r.get("takeovers", 0)
        r["autonomous"] = dcas.autonomous_completion(r)
        r["score"] = scenario_score(r, dcas.REF_SPEED.get(name, 40))
        return kind, name, _slim(r)
    from autodrive.scenarios import make_env
    env = make_env(ALL_BENCH[name], seed=0)
    if kind == "vision":
        from autodrive.vision.drive import drive_episode
        from autodrive.vision.policy import load_policy
        r, _, _ = drive_episode(env, load_policy(path), seed=0)
    else:
        r = run_episode(env, make_policy(path), seed=0)
    r["score"] = scenario_score(r, REF_SPEED.get(name, 30))
    r["takeovers"] = r.get("takeovers", 0)
    r["autonomous"] = dcas.autonomous_completion(r)
    return kind, name, _slim(r)


def _slim(r):
    return {k: v for k, v in r.items() if isinstance(v, (int, float, bool, str, dict, type(None)))
            or (isinstance(v, list) and len(v) < 50)}


def summarize(card):
    from autodrive.bench import BENCHMARK, RURAL_BENCHMARK, safety_totals
    sc = card["scenarios"]
    card["total"] = round(float(np.mean([v["score"] for v in sc.values()])), 2)
    card["urban"] = round(float(np.mean([v["score"] for k, v in sc.items() if k in BENCHMARK])), 2)
    card["rural"] = round(float(np.mean([v["score"] for k, v in sc.items() if k in RURAL_BENCHMARK])), 2)
    card["safety"] = safety_totals(card)
    card["finished"] = sum(v["finished"] for v in sc.values())
    card["autonomous"] = sum(bool(v.get("autonomous")) for v in sc.values())
    card["n"] = len(sc)
    return card


def table_v22(cards):
    names = {"expert": "규칙 기반 expert", "rl": "rl-daily 최고", "vision": "비전(카메라 1대+위성+GPS)"}
    out = "| 정책 | 총점 | 도시 | 시골 | 완주 | 자율 완주(개입 0) | 사고 | 충돌 | 신호 위반 |\n|---|---|---|---|---|---|---|---|---|\n"
    for k, c in cards.items():
        s = c["safety"]
        out += (f"| {names.get(k, k)} | {c['total']} | {c['urban']} | {c['rural']} | {c['finished']}/{c['n']} | "
                f"{c['autonomous']}/{c['n']} | {s.get('crashes', '-')} | {s.get('collisions', '-')} | "
                f"{s.get('signal_violations', s.get('red_runs', '-'))} |\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="weights/vision_policy.pt")
    ap.add_argument("--rl", default="weights/rl_daily_best.zip")
    ap.add_argument("--only", default="vision,expert,rl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/vision_benchmark.json")
    ap.add_argument("--suite", default="v22", choices=["v22", "dcas"])
    ap.add_argument("--md", default=None, help="점수표 마크다운도 저장")
    a = ap.parse_args()
    from multiprocessing import get_context
    from autodrive import dcas
    from autodrive.bench import ALL_BENCH
    from autodrive.scenarios import BENCH_VERSION
    kinds = a.only.split(",")
    paths = {"vision": a.policy, "expert": "pure_pursuit", "rl": a.rl}
    names = dcas.DCAS_NAMES if a.suite == "dcas" else list(ALL_BENCH)
    jobs = [(k, paths[k], n, a.suite) for k in kinds for n in names]
    cards = {k: {"scenarios": {}} for k in kinds}
    with get_context("spawn").Pool(a.workers) as pool:
        for k, n, r in pool.imap_unordered(_one, jobs):
            cards[k]["scenarios"][n] = r
            print(f"{k:7s} {n:24s} score {r['score']:6.2f} crash {r['crashed']} auto {r['autonomous']}"
                  + (f" pass {r['dcas_pass']}" if a.suite == "dcas" else f" red {r.get('red_runs')}"), flush=True)
    for k in kinds:
        cards[k]["scenarios"] = {n: cards[k]["scenarios"][n] for n in names}
        if a.suite == "dcas":
            cards[k] = dcas.summarize(cards[k]["scenarios"])
        else:
            summarize(cards[k])
    old = json.load(open(a.out)) if os.path.exists(a.out) else {}
    old.update(cards)
    old["bench"] = dcas.SUITE if a.suite == "dcas" else BENCH_VERSION
    json.dump(old, open(a.out, "w"), ensure_ascii=False, indent=1)
    allc = {k: old[k] for k in ("expert", "rl", "vision") if k in old}
    md = dcas.table_md(allc) if a.suite == "dcas" else table_v22(allc)
    if a.md:
        open(a.md, "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
