"""비전 정책 매일 라운드 관문 + 점수표(leaderboard) 쓰기.

python tools/vision_gate.py --best-v22 B.json --best-dcas B.json --cand-v22 C.json --cand-dcas C.json \
       --round 5 --log results/vision_daily_log.jsonl --board results/vision_leaderboard.md
종료 코드 0 = 승격, 1 = 유지(후보 버림), 2 = 입력 오류.

승격 조건(모두):
  1) 안전 합계 안 나빠짐 — v2.2: 사고, 충돌, 신호·우회전 위반, 시골 위험 행동 / DCAS: 사고, 충돌
  2) 자율 완주 수(v2.2 + DCAS, 개입 0·충돌 0·도착) 안 줄어듦, DCAS 합격 수 안 줄어듦
  3) 평균 점수((v2.2 총점 + DCAS 총점)/2) ≥ 최고 + min_gain  또는  자율 완주 수가 늘어남
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys


def card(path):
    return json.load(open(path))["vision"]


def metrics(v22, dc):
    s = v22["safety"]
    return dict(v22_total=v22["total"], dcas_total=dc["total"], mean=round((v22["total"] + dc["total"]) / 2, 2),
                v22_auto=v22["autonomous"], dcas_auto=dc["autonomous"], auto=v22["autonomous"] + dc["autonomous"],
                n_auto=v22["n"] + dc["n"], dcas_pass=dc["dcas_pass"], v22_finished=v22["finished"],
                crashes=s["crashes"], collisions=s["collisions"], signal=s["signal_violations"],
                rural_unsafe=s["rural_unsafe"], dcas_crashes=dc["crashes"], dcas_collisions=dc["collisions"])


SAFETY = ("crashes", "collisions", "signal", "rural_unsafe", "dcas_crashes", "dcas_collisions")


def decide(c, b, min_gain=0.3):
    why = []
    worse = [k for k in SAFETY if c[k] > b[k]]
    if worse:
        why.append("안전 나빠짐: " + ", ".join(f"{k} {b[k]}→{c[k]}" for k in worse))
    if c["auto"] < b["auto"]:
        why.append(f"자율 완주 줄어듦 {b['auto']}→{c['auto']}")
    if c["dcas_pass"] < b["dcas_pass"]:
        why.append(f"DCAS 합격 줄어듦 {b['dcas_pass']}→{c['dcas_pass']}")
    gain = c["mean"] >= b["mean"] + min_gain or c["auto"] > b["auto"]
    if not gain:
        why.append(f"향상 부족: 평균 {b['mean']}→{c['mean']} (+{min_gain} 필요), 자율 완주 {b['auto']}→{c['auto']}")
    return (not why), why


def board(rows):
    h = ("| 날짜(KST) | 라운드 | 결과 | v2.2 총점 | DCAS 총점 | 평균 | 자율 완주 v2.2 | 자율 완주 DCAS | DCAS 합격 | "
         "사고 | 충돌 | 신호 위반 | 시골 위험 | DCAS 사고 |\n|" + "---|" * 14 + "\n")
    out = []
    for r in rows:
        m = r["cand"]
        out.append(f"| {r['date']} | r{r['round']} | {'✅ 승격' if r['promoted'] else '❌ 유지'} | {m['v22_total']} | "
                   f"{m['dcas_total']} | {m['mean']} | {m['v22_auto']}/18 | {m['dcas_auto']}/20 | {m['dcas_pass']}/20 | "
                   f"{m['crashes']} | {m['collisions']} | {m['signal']} | {m['rural_unsafe']} | {m['dcas_crashes']} |")
    best = rows[-1]["best_after"]
    return ("# 비전 정책 매일 라운드 점수표\n\n카메라 1대(앞유리 가운데 위) + 위성 지도 + GPS 만으로 운전. 실드 없음.\n"
            "자율 완주 = 도착 + 충돌·이탈 0 + 개입 0.\n\n" + h + "\n".join(out) +
            f"\n\n**현재 최고(weights/vision_best.pt)**: v2.2 {best['v22_total']}, DCAS {best['dcas_total']}, "
            f"자율 완주 {best['auto']}/{best['n_auto']}, DCAS 합격 {best['dcas_pass']}/20\n")


def main():
    ap = argparse.ArgumentParser()
    for k in ("best-v22", "best-dcas", "cand-v22", "cand-dcas", "log", "board"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--min-gain", type=float, default=0.3)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    try:
        b = metrics(card(a.best_v22), card(a.best_dcas))
        c = metrics(card(a.cand_v22), card(a.cand_dcas))
    except Exception as e:  # noqa: BLE001
        print("gate input error:", e)
        return 2
    ok, why = decide(c, b, a.min_gain)
    row = dict(date=dt.datetime.now().strftime("%Y-%m-%d %H:%M"), round=a.round, promoted=ok, why=why,
               cand=c, best_before=b, best_after=c if ok else b, note=a.note)
    with open(a.log, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    rows = [json.loads(x) for x in open(a.log) if x.strip()]
    open(a.board, "w").write(board(rows))
    print(json.dumps(dict(promoted=ok, why=why, cand=c, best=b), ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
