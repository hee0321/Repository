"""인식·예측·안전 계획 시제품 평가.
1) 인식 오차(held-out 시드 90_000대 시나리오에서 측정, 정답은 측정용만)
2) 운전: 비전만 vs 비전+인식 계획기 — v2.2 18개와 DCAS 20개, 자율 완주·사고·위반

python eval_perception.py --suite dcas --workers 4 --out results/perception_eval.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _drive(job):
    import torch
    torch.set_num_threads(1)
    kind, pol, seg, name, suite, measure = job
    from autodrive import dcas
    from autodrive.bench import ALL_BENCH, REF_SPEED, scenario_score
    from autodrive.scenarios import make_env
    from autodrive.perception.drive import drive_perception
    from autodrive.perception.seg import load_seg
    from autodrive.vision.drive import drive_episode
    from autodrive.vision.policy import load_policy
    if suite == "dcas":
        env = dcas.make_env(name, seed=0)
        env.monitor = dcas.DcasMonitor()
    else:
        env = make_env(ALL_BENCH[name], seed=0)
    if kind == "vision":
        r, _, _ = drive_episode(env, load_policy(pol), seed=0)
        metrics = None
    else:
        r, metrics, _ = drive_perception(env, load_policy(pol), load_seg(seg), seed=0,
                                         use_planner=True, measure=measure)
    if suite == "dcas":
        r.update(env.monitor.results(env, r, env.dcas))
        r["score"] = scenario_score(r, dcas.REF_SPEED.get(name, 40))
        r["autonomous"] = dcas.autonomous_completion(r)
    else:
        r["score"] = scenario_score(r, REF_SPEED.get(name, 30))
        r["autonomous"] = dcas.autonomous_completion(r)
    slim = {k: v for k, v in r.items() if isinstance(v, (int, float, bool, str, type(None)))}
    return kind, name, slim, _sum_m(metrics) if metrics else None


def _sum_m(m):
    rows = [x for block in m["det"] for x in block["rows"]]
    hits = [r for r in rows if r["hit"]]
    by = {}
    for cls in ("car", "ped", "object"):
        g = [r for r in rows if r["cls"] == cls]
        h = [r for r in g if r["hit"]]
        by[cls] = dict(n=len(g), recall=round(len(h) / max(len(g), 1), 3),
                       mae_m=round(float(np.mean([abs(r["err"]) for r in h])), 2) if h else None)
    sig = [s for s in m["sig"] if s[0] is not None]
    ok = sum(1 for g, d in sig if g == d)
    return dict(classes=by, signal_acc=round(ok / max(len(sig), 1), 3) if sig else None,
                fp_per_frame=round(float(np.mean([b["fp"] for b in m["det"]])), 2) if m["det"] else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="dcas", choices=["v22", "dcas", "both"])
    ap.add_argument("--policy", default="weights/vision_best.pt")
    ap.add_argument("--seg", default="weights/perception_seg.pt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="vision,perc")
    ap.add_argument("--out", default="results/perception_eval.json")
    a = ap.parse_args()
    from multiprocessing import get_context
    from autodrive import dcas
    from autodrive.bench import ALL_BENCH
    suites = ["v22", "dcas"] if a.suite == "both" else [a.suite]
    kinds = a.only.split(",")
    jobs = []
    for suite in suites:
        names = dcas.DCAS_NAMES if suite == "dcas" else list(ALL_BENCH)
        for k in kinds:
            for n in names:
                jobs.append((k, a.policy, a.seg, n, suite, suite == "dcas" and k == "perc"))
    cards = {s: {k: {"scenarios": {}} for k in kinds} for s in suites}
    mets = []
    with get_context("spawn").Pool(a.workers) as pool:
        for k, n, r, m in pool.imap_unordered(_drive, jobs):
            suite = "dcas" if n.startswith("dcas_") else "v22"
            cards[suite][k]["scenarios"][n] = r
            if m:
                mets.append(m)
            print(f"{k:6s} {n:24s} score {r['score']:6.2f} crash {r['crashed']} auto {r['autonomous']}"
                  + (f" brake {r.get('planner_brake_share', 0)}" if k == "perc" else ""), flush=True)
    out = {}
    for suite in suites:
        names = dcas.DCAS_NAMES if suite == "dcas" else list(ALL_BENCH)
        for k in kinds:
            sc = {n: cards[suite][k]["scenarios"][n] for n in names}
            if suite == "dcas":
                out.setdefault(suite, {})[k] = dcas.summarize(sc)
            else:
                from autodrive.bench import safety_totals
                tot = round(float(np.mean([v["score"] for v in sc.values()])), 2)
                out.setdefault(suite, {})[k] = dict(total=tot, n=len(sc),
                                                    finished=sum(v["finished"] for v in sc.values()),
                                                    autonomous=sum(bool(v["autonomous"]) for v in sc.values()),
                                                    safety=safety_totals({"scenarios": sc}), scenarios=sc)
    if mets:
        # 평균 인식 오차
        out["perception"] = dict(
            car_recall=round(float(np.mean([m["classes"]["car"]["recall"] for m in mets])), 3),
            car_mae_m=round(float(np.nanmean([m["classes"]["car"]["mae_m"] or np.nan for m in mets])), 2),
            ped_recall=round(float(np.mean([m["classes"]["ped"]["recall"] for m in mets])), 3),
            signal_acc=round(float(np.mean([m["signal_acc"] for m in mets if m["signal_acc"] is not None] or [0])), 3),
            fp_per_frame=round(float(np.mean([m["fp_per_frame"] for m in mets])), 2),
            n_episodes=len(mets))
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: {kk: {a: b for a, b in vv.items() if a != "scenarios"} for kk, vv in v.items()}
                      if isinstance(v, dict) else v for k, v in out.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
