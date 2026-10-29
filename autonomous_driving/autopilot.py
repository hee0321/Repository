"""자율 주행 모드: 학습한 PPO 모델이 스스로 운전하고, 안전 감시기가 위험하면 사람에게 알리고(인간 개입 요청)
기준 제어기/안전 정지로 넘긴다.

    python autopilot.py --scenario gangnam --model best          # 강남역 (실제 OSM 도로)
    python autopilot.py --scenario track --model best            # 합성 서킷
    python autopilot.py --scenario rural_night --model best      # 시골길 밤
    python autopilot.py --scenario gangnam_traffic --model none  # 모델 없이 기준 제어기
    python autopilot.py --scenario sumo_gangnam --force          # SUMO 실제 교통 (sumo_kr/autopilot.py, 같은 감시기)
출력: media/autopilot_<시나리오>.gif / .mp4 / .html(재생기), results/autopilot_<시나리오>_events.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from autodrive.bench import ALL_BENCH
from autodrive.controllers import PurePursuitController
from autodrive.krrules import load as load_kr
from autodrive.scenarios import make_env

ALIASES = {"gangnam": "gangnam_station", "track": "default_2laps", "obstacles": "obstacles_track",
           "school": "daechi_school_zone", "rural": "rural_day_narrow", "rural_night": "rural_night_gravel",
           "rural_curvy": "rural_curvy", "village": "rural_village"}
HERE = os.path.dirname(os.path.abspath(__file__))


def load_yaml(path):
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        return {"engage": {"min_completion": 0.95, "max_crashes": 0, "max_violations": 0, "max_school_speeding_s": 1.0},
                "monitor": {"ttc_s": 2.0, "value_drop": 4.0, "disagreement": 0.6, "ood_z": 6.0, "over_limit_kmh": 5.0,
                            "min_edge_clearance_m": 0.15, "hold_s": 3.0, "safe_stop_ttc_s": 1.0}}


def engage_check(card_path, cfg, scenario):
    """점수표 기준을 넘으면 (True, 이유)."""
    if not os.path.exists(card_path):
        return False, "점수표 없음(results/best_scorecard.json)"
    card = json.load(open(card_path, encoding="utf-8"))
    sc = card["scenarios"]
    e = cfg["engage"]
    probs = []
    if scenario in sc and sc[scenario]["completion"] < e["min_completion"]:
        probs.append(f"{scenario} 완주율 {sc[scenario]['completion']}")
    low = [k for k, v in sc.items() if v["completion"] < e["min_completion"]]
    if low:
        probs.append(f"완주율 {e['min_completion']} 미만: {', '.join(low)}")
    crashes = sum(v["crashed"] for v in sc.values())
    if crashes > e["max_crashes"]:
        probs.append(f"이탈·충돌 {crashes}건")
    viol = sum(v["red_runs"] + v.get("right_turn_no_stop", 0) + v.get("unsafe_meet", 0) + v.get("ped_unsafe", 0)
               + v.get("unsafe_overtake", 0) for v in sc.values())
    if viol > e["max_violations"]:
        probs.append(f"위반 {viol}건")
    ss = sum(v["school_speeding_s"] for v in sc.values())
    if ss > e["max_school_speeding_s"]:
        probs.append(f"어린이보호구역 과속 {ss:.1f}s")
    return (not probs), ("; ".join(probs) if probs else f"점수표 통과 (총점 {card['total']})")


class Autopilot:
    def __init__(self, env, model=None, cfg=None, stats=None, alert_cmd=None, quiet=False):
        self.env, self.model, self.cfg = env, model, cfg
        self.m = cfg["monitor"]
        self.base = PurePursuitController(env)
        self.stats = stats
        self.alert_cmd = alert_cmd
        self.quiet = quiet
        self.mode = "AUTONOMOUS" if model is not None else "BASELINE"
        self.hold_until = -1.0
        self.events = []
        self.v_hist = []

    def model_eval(self, obs):
        import torch
        pol = self.model.policy
        with torch.no_grad():
            ot = torch.as_tensor(obs[None])
            dist = pol.get_distribution(ot)
            mean = dist.distribution.mean.numpy()[0]
            std = dist.distribution.stddev.numpy()[0]
            value = float(pol.predict_values(ot).numpy()[0, 0])
            ent = float(dist.entropy().numpy()[0])
        return np.clip(mean, -1, 1), std, value, ent

    def risks(self, obs, a_model, a_base, value):
        env, m = self.env, self.m
        r = []
        ttc = env.ttc()
        if ttc < m["ttc_s"]:
            r.append(f"충돌 위험 TTC {ttc:.1f}s")
        lim = env.rules.limit_at(env._s)
        if env.car.v * 3.6 > lim * 3.6 + m["over_limit_kmh"]:
            r.append(f"과속 {env.car.v * 3.6:.0f}/{lim * 3.6:.0f} km/h")
        sd = env.stop_distance()
        if np.isfinite(sd) and env.car.v ** 2 / (2 * 4.0) > sd + 1.0:
            r.append(f"신호 정지 못 할 위험 (정지선 {sd:.0f} m)")
        if env.edge_clearance() < m["min_edge_clearance_m"]:
            r.append("길 가장자리(농수로) 너무 가까움")
        if env.rules.rural:
            v = env.car.v
            if v * 1.0 + v * v / 8.0 > env.sight + 3:
                r.append(f"시야 {env.sight:.0f} m보다 빠름")
        if a_model is not None:
            dis = float(np.abs(a_model - a_base).sum())
            if dis > m["disagreement"]:
                r.append(f"자신 없음(기준과 차이 {dis:.2f})")
            self.v_hist.append(value)
            k = int(0.5 / env.dt)
            if len(self.v_hist) > k and self.v_hist[-k - 1] - value > m["value_drop"]:
                r.append(f"자신 없음(가치 {self.v_hist[-k - 1]:.1f}→{value:.1f})")
            if self.stats is not None:
                z = np.abs((obs - self.stats["mean"]) / self.stats["std"])
                if z.max() > m["ood_z"]:
                    r.append(f"처음 보는 상황(OOD z={z.max():.1f})")
        return r, ttc

    def alert(self, t, reasons, action):
        ev = {"t": round(t, 2), "s": round(float(self.env._s), 1), "speed_kmh": round(self.env.car.v * 3.6, 1),
              "reasons": reasons, "action": action}
        self.events.append(ev)
        if not self.quiet:
            print(f"\n\033[1;41m ⚠ 인간 개입 요청 (t={t:.1f}s) \033[0m {', '.join(reasons)} → {action}\a", flush=True)
        if self.alert_cmd:
            try:
                subprocess.Popen(self.alert_cmd, shell=True)
            except Exception:
                pass

    def step(self, obs):
        env = self.env
        a_base = self.base.act(obs)
        a_model, std, value, ent = (None, None, 0.0, 0.0)
        if self.model is not None:
            a_model, std, value, ent = self.model_eval(obs)
        reasons, ttc = self.risks(obs, a_model, a_base, value)
        t = env.t
        prev = self.mode
        if self.model is not None:
            if reasons and self.mode == "AUTONOMOUS":
                self.mode = "TAKEOVER"
                self.hold_until = t + self.m["hold_s"]
                self.alert(t, reasons, "기준 제어기로 전환")
            elif self.mode == "TAKEOVER" and t >= self.hold_until and not reasons:
                self.mode = "AUTONOMOUS"
                self.events.append({"t": round(t, 2), "s": round(float(env._s), 1), "reasons": [], "action": "자율 모드 복귀"})
        # 비상 정지는 '실제로 빠르게 다가가는' 경우만 (서 있는 장애물을 기준 제어기가 옆으로 비켜 가는 중이면 제외)
        lo = env.lead_obstacle()
        bypass = lo is not None and abs(lo[1]) < 1.0 and self.base.avoid_offset(env._s)[0] is not None
        if ttc < self.m["safe_stop_ttc_s"] and env.car.v > 3.0 and not bypass:
            if prev != "SAFE_STOP":
                self.alert(t, [f"비상: TTC {ttc:.1f}s"], "안전 정지")
            self.mode = "SAFE_STOP"
            self.hold_until = t + self.m["hold_s"]
        elif self.mode == "SAFE_STOP" and t >= self.hold_until:
            self.mode = "TAKEOVER" if self.model is not None else "BASELINE"
        if self.mode == "AUTONOMOUS":
            a = a_model
        elif self.mode == "SAFE_STOP":
            a = np.array([a_base[0], -1.0], np.float32)
        else:
            a = a_base
        conf = None if std is None else float(np.exp(-np.abs(a_model - a_base).sum()))
        why = {"mode": self.mode, "rule": getattr(self.base, "rule_note", ""), "ttc": ttc, "risks": reasons,
               "a_model": None if a_model is None else a_model.round(3).tolist(),
               "std": None if std is None else std.round(3).tolist(), "entropy": round(ent, 3),
               "value": round(value, 2), "confidence": conf, "a_base": np.round(a_base, 3).tolist(),
               "action": np.round(a, 3).tolist()}
        return a, why


def signal_state(env):
    nx = env.rules.next_signal(env._s, 150.0)
    if nx is None:
        return "-", None
    return nx[1].state(env.t), nx[0]


def run(args):
    if args.scenario.startswith("sumo_"):  # SUMO 실제 교통 (같은 감시기·같은 rules/autopilot.yaml) → sumo_kr/autopilot.py
        from sumo_kr.autopilot import main as sumo_main
        argv = ["--scenario", args.scenario, "--model", args.model, "--seed", str(args.seed or 101)]
        argv += ["--force"] * args.force + ["--quiet"] * args.quiet + (["--alert-cmd", args.alert_cmd] if args.alert_cmd else [])
        return sumo_main(argv)
    name = ALIASES.get(args.scenario, args.scenario)
    spec = ALL_BENCH[name]
    cfg = load_yaml(os.path.join(HERE, "rules", "autopilot.yaml"))
    env = make_env(spec, seed=args.seed)
    model, stats, why_mode = None, None, ""
    path = os.path.join("weights", "best.zip") if args.model == "best" else args.model
    if args.model != "none" and path and os.path.exists(path):
        ok, why_mode = engage_check(os.path.join("results", "best_scorecard.json"), cfg, name)
        if ok or args.force:
            from stable_baselines3 import PPO
            model = PPO.load(path, device="cpu")
            if os.path.exists(os.path.join("weights", "obs_stats.npz")):
                stats = dict(np.load(os.path.join("weights", "obs_stats.npz")))
        print(f"자율 모드 {'켜짐' if model is not None else '꺼짐(기준 제어기)'}: {why_mode}{' (--force)' if args.force and not ok else ''}")
    else:
        print(f"모델 없음({path}) → 기준 제어기(Pure Pursuit)로 달려요")
    ap = Autopilot(env, model, cfg, stats, alert_cmd=args.alert_cmd, quiet=args.quiet)
    obs, info = env.reset(seed=args.seed)
    frames, done, steps_auto, steps = [], False, 0, 0
    t0 = time.time()
    while not done:
        a, why = ap.step(obs)
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        steps_auto += why["mode"] == "AUTONOMOUS"
        st, sd = signal_state(env)
        if steps % args.every == 0 or term or trunc:
            frames.append({
                "t": round(env.t, 2), "x": round(env.car.x, 2), "y": round(env.car.y, 2), "yaw": round(env.car.yaw, 3),
                "v": round(env.car.v * 3.6, 1), "limit": round(env.rules.limit_at(env._s) * 3.6), "s": round(env._s, 1),
                "light": st, "light_d": None if sd is None else round(sd, 1),
                "school": env.rules.school_at(env._s), "sight": round(env.sight, 1),
                "obs": [[round(float(c[0]), 2), round(float(c[1]), 2), round(float(c[2]), 2), env.ob_kind[k]]
                        for c, k in zip(env.circles, env._vis)],
                "rays": np.round(env.lidar_ranges, 1).tolist(),
                "ttc": None if not np.isfinite(why["ttc"]) else round(why["ttc"], 1),
                "mode": why["mode"], "rule": why["rule"], "risks": why["risks"], "action": why["action"],
                "a_model": why["a_model"], "std": why["std"], "conf": None if why["confidence"] is None else round(why["confidence"], 2),
                "value": why["value"], "n_events": len(ap.events)})
        done = term or trunc
        if args.realtime:
            time.sleep(max(0.0, env.dt - (time.time() - t0) / steps))
    v = info["violations"]
    summary = {"scenario": name, "model": path if model is not None else "pure_pursuit", "engage": why_mode,
               "completion": round(info["completion"], 3), "crashed": info["crashed"], "time_s": round(env.t, 1),
               "autonomous_share": round(steps_auto / max(steps, 1), 3),
               "takeovers": sum(1 for e in ap.events if e["action"] == "기준 제어기로 전환"),
               "safe_stops": sum(1 for e in ap.events if e["action"] == "안전 정지"),
               "violations": v, "events": ap.events[:200]}
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", f"autopilot_{name}_events.jsonl"), "w", encoding="utf-8") as f:
        for e in ap.events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(os.path.join("results", f"autopilot_{name}_summary.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in summary.items() if k != "events"}, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n[{name}] 완주 {summary['completion'] * 100:.0f}%, 이탈 {summary['crashed']}, {summary['time_s']} s, "
          f"자율 비율 {summary['autonomous_share'] * 100:.0f}%, 개입 요청 {summary['takeovers']}회, 안전 정지 {summary['safe_stops']}회")
    print("  위반:", {k: x for k, x in v.items() if x})
    os.makedirs(args.out, exist_ok=True)
    from autodrive.viz import render_autopilot
    outs = render_autopilot(env, frames, name, args.out, gif=not args.no_gif, mp4=args.mp4, html=not args.no_html)
    for o in outs:
        print("  저장:", o, f"({os.path.getsize(o) / 1e6:.2f} MB)")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="gangnam", help="gangnam | track | obstacles | school | rural | rural_night | 벤치마크 이름 | sumo_gangnam | sumo_gangnam_reverse | sumo_daechi_school")
    ap.add_argument("--model", default="best", help="best | 가중치 경로 | none")
    ap.add_argument("--force", action="store_true", help="점수표 기준을 못 넘어도 자율 모드 켜기(시험용)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=4, help="그림용으로 몇 스텝마다 기록할지")
    ap.add_argument("--out", default="media")
    ap.add_argument("--mp4", action="store_true")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("--realtime", action="store_true", help="실제 시간 속도로 돌기(스텝마다 dt 기다림)")
    ap.add_argument("--alert-cmd", default=None, help="개입 요청 때 실행할 명령 (예: 'notify-send 개입')")
    ap.add_argument("--quiet", action="store_true")
    run(ap.parse_args())
