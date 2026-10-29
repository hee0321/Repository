"""SUMO 자율 주행 모드: 학습한 PPO 모델이 실제 강남역·대치동 교통 속을 운전하고, 2D 자율 주행 모드와 **같은 안전 감시기**
(`autopilot.Autopilot` + `rules/autopilot.yaml`)가 위험하면 '인간 개입 요청'을 띄우고 기준 운전자(규칙)에게 3초 넘기거나 안전 정지해요.

    python -m sumo_kr.autopilot --scenario gangnam --model best          # 점수표 기준 못 넘으면 기준 운전자가 운전
    python -m sumo_kr.autopilot --scenario gangnam --model best --force  # 모델이 직접 운전(시험용)
    python -m sumo_kr.autopilot --eval --force                           # 경로 3개 × 시드 2개 개입 통계
    python autopilot.py --scenario sumo_gangnam --force                  # 같은 것 (2D 진입점에서 넘겨줌)
출력: results/sumo_autopilot_<경로>@<시드>_events.jsonl, results/sumo_autopilot_scorecard.json / .md

감시기 로직은 복사하지 않고 `autopilot.Autopilot`을 그대로 써요. SUMO 쪽은 감시기가 읽는 신호만 계산하는 얇은 어댑터예요:
- TTC: min(traci getLeader(앞차 간격, minGap 제외 = SUMO 충돌 경계) / (내 속도 − 앞차 속도), 교차로 충돌 지점(getJunctionFoes) 도착 시간)
- 과속: 차로 제한속도(OSM maxspeed·5030·어린이보호구역 30 km/h로 다시 칠한 값) + 5 km/h
- 신호 정지 못 할 위험: 다음 신호가 빨강/노랑일 때 정지선까지 거리
- 길 가장자리: 차로 폭/2 − |횡방향 위치| − 차폭/2 (도로 끝 차로만)
- 모델 자신 없음: 가치(value) 0.5초 하락, 모델 행동 vs 기준 운전자 행동 차이, 관측 OOD(z) — weights/sumo_obs_stats.npz
행동 순서: 감시기(2D)는 [조향, 가감속], SUMO는 [가감속, 차로 변경] → 어댑터에서 [차로 변경, 가감속]으로 바꿔 넘기고 되돌려요.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopilot import Autopilot, load_yaml  # noqa: E402  (같은 감시기·같은 설정)

from .env import DT, SCENARIOS, SumoKREnv  # noqa: E402

W, R = os.path.join(ROOT, "weights"), os.path.join(ROOT, "results")
BEST = os.path.join(W, "sumo_best.zip")
BEST_CARD = os.path.join(R, "sumo_best_scorecard.json")
STATS = os.path.join(W, "sumo_obs_stats.npz")
EVAL_SEEDS = (101, 202)
TAKEOVER, SAFE_STOP = "기준 제어기로 전환", "안전 정지"


# ---------------- 어댑터: SUMO 환경을 감시기가 읽는 모양으로 ----------------
class _Car:
    def __init__(self, view):
        self._view = view

    @property
    def v(self):
        return float(self._view.sim._cache["v"])


class _Rules:
    rural = False

    def __init__(self, view):
        self._view = view

    def limit_at(self, s):  # 현재 차로 제한속도(어린이보호구역이면 30 km/h)
        return float(self._view.sim._cache["lim"])


class SumoMonitorView:
    """Autopilot(감시기)이 쓰는 env 인터페이스(t, dt, _s, car.v, rules.limit_at, ttc, stop_distance, edge_clearance,
    lead_obstacle)를 SumoKREnv 위에 구현."""

    sight = np.inf

    def __init__(self, sim: SumoKREnv):
        self.sim = sim
        self.dt = DT
        self.car = _Car(self)
        self.rules = _Rules(self)

    @property
    def t(self):
        return self.sim.steps * DT

    @property
    def _s(self):
        return float(self.sim.stats["dist_m"])

    def leader_ttc(self):
        tr = self.sim.traci
        if not self.sim._ego_ok():
            return np.inf
        lead = tr.vehicle.getLeader("ego", 100.0)
        if not lead or not lead[0]:
            return np.inf
        gap = max(0.0, lead[1])  # getLeader 간격은 minGap을 뺀 값 = SUMO가 충돌로 판정하는 경계(collision mingap-factor 1)까지 거리
        closing = tr.vehicle.getSpeed("ego") - tr.vehicle.getSpeed(lead[0])
        return gap / closing if closing > 0.1 else np.inf

    def ttc(self):
        """앞차 TTC와 교차로 충돌 지점 TTC 중 작은 값 (getLeader는 옆에서 들어오는 차를 못 봄)."""
        return min(self.leader_ttc(), self.junction_ttc())

    def junction_ttc(self, margin=0.5):
        """지금 속도로 가면 내가 충돌 지점에 있는 동안 foe가 들어오는 경우의 '충돌까지 시간'."""
        v = self.sim.traci.vehicle.getSpeed("ego") if self.sim._ego_ok() else 0.0
        if v < 0.5:
            return np.inf
        best = np.inf
        for _, ed, ex, fa, fe in self.sim.junction_conflicts():
            ea, ee = max(ed, 0.0) / v, ex / v
            if fa <= ee + margin and fe >= ea - margin:   # 두 차가 충돌 지점을 차지하는 시간이 겹침
                best = min(best, max(ea, fa))
        return best

    def stop_distance(self):
        c = self.sim._cache
        return float(c["tdist"]) if c["st"] in "rRyY" and c["tdist"] < 150 else np.inf

    def edge_clearance(self):
        tr = self.sim.traci
        if not self.sim._ego_ok():
            return np.inf
        edge = tr.vehicle.getRoadID("ego")
        if edge.startswith(":"):
            return np.inf  # 교차로 안 내부 차로는 '도로 끝'이 아님
        lane = tr.vehicle.getLaneID("ego")
        ln, nl = tr.vehicle.getLaneIndex("ego"), tr.edge.getLaneNumber(edge)
        lat = tr.vehicle.getLateralLanePosition("ego")  # +면 왼쪽
        free = tr.lane.getWidth(lane) / 2 - tr.vehicle.getWidth("ego") / 2
        out = np.inf
        if ln == 0:
            out = min(out, free + lat)        # 오른쪽 도로 끝(보도·갓길)
        if ln == nl - 1:
            out = min(out, free - lat)        # 왼쪽 끝(중앙선)
        return float(out)

    def lead_obstacle(self):
        return None  # 2D의 '서 있는 장애물 비켜 가기' 예외는 SUMO에 없음 → TTC<1s면 항상 안전 정지


def _to_mon(a):   # SUMO [가감속, 차로] → 감시기 [차로(=조향 자리), 가감속]
    return np.array([a[1], a[0]], np.float32)


def _to_sumo(a):
    return np.array([a[1], a[0]], np.float32)


def _lane_cmd(x):  # SUMO는 |차로 값|>0.5 일 때만 차로 변경 → 실제로 하는 행동으로 맞춰 비교
    return float(np.sign(x)) if abs(x) > 0.5 else 0.0


class _ExpertBase:
    """감시기의 '기준 제어기' 자리: SUMO 기준 운전자(규칙: 제한속도·앞차·빨간불·우회전 일시정지)."""

    def __init__(self, sim):
        self.sim = sim
        self.rule_note = ""

    def act(self, obs):
        a = self.sim.expert_action()
        self.rule_note = getattr(self.sim, "rule_note", "")
        return _to_mon(a)

    def avoid_offset(self, s):
        return (None,)


class SumoAutopilot(Autopilot):
    """autopilot.Autopilot 그대로 (risks·alert·step·모드 전환·안전 정지·설정 값 공유), 입출력만 SUMO에 맞춤."""

    def __init__(self, sim, model=None, cfg=None, stats=None, alert_cmd=None, quiet=False):
        self.sim = sim
        super().__init__(SumoMonitorView(sim), model, cfg, stats, alert_cmd, quiet)
        self.base = _ExpertBase(sim)

    def model_eval(self, obs):
        mean, std, value, ent = super().model_eval(obs)
        mean = np.array([mean[0], _lane_cmd(mean[1])], np.float32)
        return _to_mon(mean), _to_mon(std), value, ent

    def alert(self, t, reasons, action):
        super().alert(t, reasons, action)
        self.events[-1]["edge"] = self.sim.traci.vehicle.getRoadID("ego") if self.sim._ego_ok() else ""

    def act(self, obs):
        a, why = self.step(obs)
        return _to_sumo(a), why


# ---------------- 켜기 조건(점수표) · OOD 통계 ----------------
def sumo_engage_check(card_path, cfg):
    """rules/autopilot.yaml의 engage 기준을 SUMO 점수표(results/sumo_best_scorecard.json)에 적용."""
    if not os.path.exists(card_path):
        return False, "점수표 없음(results/sumo_best_scorecard.json)"
    card = json.load(open(card_path, encoding="utf-8"))
    sc, e = card["scenarios"], cfg["engage"]
    probs = []
    low = [k for k, v in sc.items() if v["completion"] < e["min_completion"]]
    if low:
        probs.append(f"완주율 {e['min_completion']} 미만: {', '.join(low)}")
    crashes = sum(v["collisions"] for v in sc.values())
    if crashes > e["max_crashes"]:
        probs.append(f"충돌 {crashes}건")
    viol = sum(v["red_runs"] + v["right_turn_no_stop"] for v in sc.values())
    if viol > e["max_violations"]:
        probs.append(f"신호·우회전 위반 {viol}건")
    ss = sum(v["school_speeding_s"] for v in sc.values())
    if ss > e["max_school_speeding_s"]:
        probs.append(f"어린이보호구역 과속 {ss:.1f}s")
    return (not probs), ("; ".join(probs) if probs else f"점수표 통과 (총점 {card['total']})")


def build_obs_stats(path=STATS, seeds=(1, 2, 3), noise=0.15):
    """기준 운전자(+BC 때처럼 약간의 잡음)로 달린 관측 분포 → '처음 보는 상황'(OOD) 판단용. 평가 시드(101, 202)와 겹치지 않게."""
    rng = np.random.default_rng(0)
    X = []
    for n in SCENARIOS:
        env = SumoKREnv(n, max_steps=500)
        for sd in seeds:
            obs, _ = env.reset(seed=sd)
            done = False
            while not done:
                X.append(obs)
                a = env.expert_action() + rng.normal(0, noise, 2)
                obs, _, te, tr, _ = env.step(a)
                done = te or tr
        env.close()
    X = np.array(X, np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, mean=X.mean(0), std=X.std(0) + 1e-3)  # improve.py 와 같은 식
    return dict(np.load(path))


def load_model(model, force, cfg, log=print):
    """(모델 또는 None, OOD 통계, 켜기 판단 문구)."""
    path = BEST if model == "best" else model
    if model == "none" or not path or not os.path.exists(path):
        return None, None, f"모델 없음({path}) → 기준 운전자"
    ok, why = sumo_engage_check(BEST_CARD, cfg) if model == "best" else (False, "점수표 없는 가중치")
    if not (ok or force):
        return None, None, why
    from stable_baselines3 import PPO
    m = PPO.load(path, device="cpu")
    stats = dict(np.load(STATS)) if os.path.exists(STATS) else build_obs_stats()
    return m, stats, why + (" (--force)" if force and not ok else "")


# ---------------- 실행 ----------------
REASON_KEYS = (("비상", "비상 정지(TTC<1s)"), ("TTC", "충돌 위험(TTC<2s)"), ("과속", "과속(+5km/h)"),
               ("신호 정지", "신호 정지 못 할 위험"), ("가장자리", "길 가장자리"), ("기준과 차이", "모델·기준 운전자 차이"),
               ("가치", "가치(value) 급락"), ("OOD", "처음 보는 상황(OOD)"), ("시야", "시야보다 빠름"))


def reason_key(r):
    """'충돌 위험 TTC 1.6s' → '충돌 위험(TTC<2s)' 처럼 숫자를 뺀 이유 종류."""
    for k, v in REASON_KEYS:
        if k in r:
            return v
    return re.sub(r"[-\d.]+", "", r).strip()


def run_episode(name, seed, model=None, stats=None, cfg=None, quiet=True, alert_cmd=None, max_steps=700, trace=None):
    cfg = cfg or load_yaml(os.path.join(ROOT, "rules", "autopilot.yaml"))
    sim = SumoKREnv(name, max_steps=max_steps)
    ap = SumoAutopilot(sim, model, cfg, stats, alert_cmd=alert_cmd, quiet=quiet)
    obs, _ = sim.reset(seed=seed)
    done, steps, auto, modes = False, 0, 0, Counter()
    info = {"stats": sim.stats, "reason": ""}
    while not done:
        a, why = ap.act(obs)
        modes[why["mode"]] += 1
        auto += why["mode"] == "AUTONOMOUS"
        if trace is not None:
            trace.append({"t": round(sim.steps * DT, 1), "v_kmh": round(sim._cache["v"] * 3.6, 1),
                          "lim_kmh": round(sim._cache["lim"] * 3.6), "mode": why["mode"], "rule": why["rule"],
                          "ttc": None if not np.isfinite(why["ttc"]) else round(why["ttc"], 2), "risks": why["risks"],
                          "action": np.round(_to_sumo(why["action"]), 3).tolist()})
        obs, _, te, tr, info = sim.step(a)
        steps += 1
        done = te or tr
    sim.close()
    st = dict(info["stats"])
    reasons = Counter(reason_key(r) for e in ap.events if e["action"] == TAKEOVER for r in e["reasons"])
    out = {"scenario": name, "seed": seed, "model": "ppo" if model is not None else "expert",
           "arrived": info["reason"] == "arrived", "end": info["reason"],
           "completion": round(min(1.0, st["dist_m"] / sim.route_len), 3), "time_s": round(steps * DT, 1),
           "autonomous_share": round(auto / max(steps, 1), 3), "modes": dict(modes),
           "takeovers": sum(1 for e in ap.events if e["action"] == TAKEOVER),
           "safe_stops": sum(1 for e in ap.events if e["action"] == SAFE_STOP),
           "takeover_reasons": dict(reasons), "collisions": st["collisions"], "red_runs": st["red_runs"],
           "right_turn_no_stop": st["right_turn_no_stop"], "right_turn_ok": st["right_turn_ok"],
           "speeding_s": st["speeding_s"], "school_speeding_s": st["school_speeding_s"], "harsh_brake": st["harsh_brake"]}
    os.makedirs(R, exist_ok=True)
    with open(os.path.join(R, f"sumo_autopilot_{name}@{seed}_events.jsonl"), "w", encoding="utf-8") as f:
        for e in ap.events:
            f.write(json.dumps(e, ensure_ascii=False, default=float) + "\n")
    return out


def summarize(rows):
    n = len(rows)
    reasons = Counter()
    for r in rows:
        reasons.update(r["takeover_reasons"])
    tot_steps = sum(sum(r["modes"].values()) for r in rows)
    return {"runs": n, "arrived": sum(r["arrived"] for r in rows),
            "autonomous_share": round(sum(r["autonomous_share"] * sum(r["modes"].values()) for r in rows) / max(tot_steps, 1), 3),
            "takeovers": sum(r["takeovers"] for r in rows), "safe_stops": sum(r["safe_stops"] for r in rows),
            "takeover_reasons": dict(reasons.most_common()), "collisions": sum(r["collisions"] for r in rows),
            "red_runs": sum(r["red_runs"] for r in rows), "right_turn_no_stop": sum(r["right_turn_no_stop"] for r in rows),
            "speeding_s": sum(r["speeding_s"] for r in rows), "school_speeding_s": sum(r["school_speeding_s"] for r in rows)}


def evaluate_autopilot(model="best", force=False, quiet=True, log=print):
    """SUMO 점수표와 같은 경로 3개 × 시드 2개를 자율 주행 모드(감시기 켬)로 달리고 개입 통계 저장."""
    cfg = load_yaml(os.path.join(ROOT, "rules", "autopilot.yaml"))
    m, stats, why = load_model(model, force, cfg)
    log(f"자율 모드 {'켜짐' if m is not None else '꺼짐(기준 운전자)'}: {why}")
    rows = [run_episode(n, sd, m, stats, cfg, quiet=quiet) for n in SCENARIOS for sd in EVAL_SEEDS]
    card = {"engage": why, "model": "ppo" if m is not None else "expert", "summary": summarize(rows), "runs": rows}
    json.dump(card, open(os.path.join(R, "sumo_autopilot_scorecard.json"), "w", encoding="utf-8"), ensure_ascii=False,
              indent=1, default=float)
    md = [f"# SUMO 자율 주행 모드 개입 통계 ({card['model']}, {why})\n",
          "| 경로@시드 | 도착 | 자율 비율 | 개입 요청 | 안전 정지 | 이유(개입) | 충돌 | 신호 위반 | 우회전 일시정지 위반 | 과속 s | 보호구역 과속 s |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['scenario']}@{r['seed']} | {'예' if r['arrived'] else r['end']} | {r['autonomous_share'] * 100:.0f}% | "
                  f"{r['takeovers']} | {r['safe_stops']} | {', '.join(f'{k} {v}' for k, v in r['takeover_reasons'].items()) or '-'} | "
                  f"{r['collisions']} | {r['red_runs']} | {r['right_turn_no_stop']} | {r['speeding_s']:.1f} | {r['school_speeding_s']:.1f} |")
    open(os.path.join(R, "sumo_autopilot_scorecard.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    s = card["summary"]
    log(f"자율 비율 {s['autonomous_share'] * 100:.0f}%, 도착 {s['arrived']}/{s['runs']}, 개입 요청 {s['takeovers']}회, "
        f"안전 정지 {s['safe_stops']}회, 충돌 {s['collisions']}, 신호 위반 {s['red_runs']}, 우회전 일시정지 위반 {s['right_turn_no_stop']}, "
        f"과속 {s['speeding_s']:.1f}s (보호구역 {s['school_speeding_s']:.1f}s)")
    log("  개입 이유:", s["takeover_reasons"])
    return card


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="gangnam", help=" | ".join(SCENARIOS) + " (sumo_ 접두어 허용)")
    ap.add_argument("--model", default="best", help="best | 가중치 경로 | none")
    ap.add_argument("--force", action="store_true", help="점수표 기준을 못 넘어도 자율 모드 켜기(시험용)")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--eval", action="store_true", help="경로 3개 × 시드 2개 개입 통계")
    ap.add_argument("--refresh-stats", action="store_true", help="OOD용 관측 분포(weights/sumo_obs_stats.npz) 다시 계산")
    ap.add_argument("--alert-cmd", default=None, help="개입 요청 때 실행할 명령 (예: 'notify-send 개입')")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.refresh_stats:
        build_obs_stats()
    if a.eval:
        return evaluate_autopilot(a.model, a.force, quiet=True)
    name = a.scenario.removeprefix("sumo_")
    cfg = load_yaml(os.path.join(ROOT, "rules", "autopilot.yaml"))
    m, stats, why = load_model(a.model, a.force, cfg)
    print(f"자율 모드 {'켜짐' if m is not None else '꺼짐(기준 운전자)'}: {why}")
    trace = []
    r = run_episode(name, a.seed, m, stats, cfg, quiet=a.quiet, alert_cmd=a.alert_cmd, trace=trace)
    with open(os.path.join(R, f"sumo_autopilot_{name}@{a.seed}_trace.jsonl"), "w", encoding="utf-8") as f:
        for x in trace:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"\n[SUMO {name}@{a.seed}] {'도착' if r['arrived'] else r['end']}, 완주 {r['completion'] * 100:.0f}%, {r['time_s']} s, "
          f"자율 비율 {r['autonomous_share'] * 100:.0f}%, 개입 요청 {r['takeovers']}회 {r['takeover_reasons']}, 안전 정지 {r['safe_stops']}회")
    print(f"  충돌 {r['collisions']}, 신호 위반 {r['red_runs']}, 우회전 일시정지 위반 {r['right_turn_no_stop']}, "
          f"과속 {r['speeding_s']:.1f}s (보호구역 {r['school_speeding_s']:.1f}s)")
    return r


if __name__ == "__main__":
    main()
