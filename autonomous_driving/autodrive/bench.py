"""평가 도구: 고정 벤치마크를 돌려 점수표(scorecard)를 만든다."""
from __future__ import annotations

import json

import numpy as np

from .controllers import PurePursuitController
from .scenarios import BENCHMARK, RURAL_BENCHMARK, make_env

ALL_BENCH = {**BENCHMARK, **RURAL_BENCHMARK}


def make_policy(policy, shield=False):
    """'pure_pursuit' 또는 PPO 가중치 경로/모델 → (env, obs) -> action 함수.
    shield=True 면 모델 행동에 안전 실드(autodrive/shield.py)를 씌운다 (점수표에 따로 표시)."""
    if policy in (None, "pure_pursuit"):
        cache = {}

        def act(env, obs):
            c = cache.get(id(env))
            if c is None or c.env is not env:
                c = cache[id(env)] = PurePursuitController(env)
            return c.act(obs)
        return act
    if isinstance(policy, str):
        from stable_baselines3 import PPO
        policy = PPO.load(policy, device="cpu")
    if shield:
        from .shield import shielded_policy
        return shielded_policy(policy)
    return lambda env, obs: policy.predict(obs, deterministic=True)[0]


def run_episode(env, act, seed=0, record=False):
    obs, info = env.reset(seed=seed)
    devs, speeds, dsteer, acc, frames = [], [], [], [], []
    prev = np.zeros(2)
    done = False
    fail_s = None
    while not done:
        a = np.clip(np.asarray(act(env, obs), dtype=float).reshape(-1)[:2], -1, 1)
        obs, r, term, trunc, info = env.step(a)
        devs.append(abs(info["lane_dev"]))
        speeds.append(info["speed"])
        dsteer.append(abs(a[0] - prev[0]))
        acc.append(abs(a[1] - prev[1]))
        prev = a
        if record:
            c = env.car
            frames.append((c.x, c.y, c.yaw, c.v, env.lidar_ranges.copy(), info["completion"], info["t"],
                           info["crashed"], info["limit"], info["s"]))
        if fail_s is None and (info["crashed"] or info["violations"]["red_runs"] or
                               info["violations"]["school_speeding_s"] > 0.5):
            fail_s = info["s"]
        done = term or trunc
    v = info["violations"]
    res = {
        "completion": round(info["completion"], 3), "finished": bool(info["finished"] or info["completion"] >= 0.999),
        "crashed": bool(info["crashed"]), "time_s": round(info["t"], 1),
        "mean_speed_kmh": round(float(np.mean(speeds)) * 3.6, 1),
        "mean_lane_dev_m": round(float(np.mean(devs)), 3), "max_lane_dev_m": round(float(np.max(devs)), 3),
        "smooth_steer": round(float(np.mean(dsteer)), 4), "smooth_accel": round(float(np.mean(acc)), 4),
        **v,
    }
    if fail_s is not None:
        res["fail_s"] = round(float(fail_s), 1)
    return (res, frames) if record else res


def scenario_score(r: dict, ref_speed_kmh: float) -> float:
    """높을수록 좋은 점수(최대 약 100)."""
    eff = min(1.0, r["mean_speed_kmh"] / max(ref_speed_kmh, 1.0))
    return round(70 * r["completion"] + 15 * eff * r["completion"] - 40 * r["crashed"]
                 - 5 * r["mean_lane_dev_m"] - 1.0 * r["speeding_s"] - 3.0 * r["school_speeding_s"]
                 - 15 * r["red_runs"] - 10 * r.get("right_turn_no_stop", 0)
                 - 1.0 * r.get("edge_close_s", 0) - 1.0 * r.get("sight_overspeed_s", 0)
                 - 10 * (r.get("unsafe_overtake", 0) + r.get("unsafe_meet", 0) + r.get("ped_unsafe", 0)) - 1.0 * r["centerline_s"] - 50 * r["smooth_steer"] + 15, 2)


# 효율 기준 속도(km/h): 규칙을 지키며 낼 수 있는 대략의 평균 속도(기준 제어기 수준)
REF_SPEED = {"rural_day_narrow": 25, "rural_night_gravel": 24, "rural_curvy": 29, "rural_village": 28,
             "rural_wide_traffic": 30, "jeongbae_village": 30, "jeongbae_night": 30, "jeongbae_lane": 22, "obstacles_track": 25, "gangnam_reverse": 25, "gangnam_traffic": 22, "oval_easy": 45, "default_2laps": 40, "hard_2laps": 35, "hard_wet": 28, "rules_track": 30,
             "gangnam_station": 25, "daechi_school_zone": 22}


def evaluate(policy, scenarios=None, seed=0, rural=True, shield=False):
    act = make_policy(policy, shield=shield)
    card = {}
    names = scenarios or (list(BENCHMARK) + (list(RURAL_BENCHMARK) if rural else []))
    for name in names:
        env = make_env(ALL_BENCH[name], seed=seed)
        r = run_episode(env, act, seed=seed)
        r["score"] = scenario_score(r, REF_SPEED.get(name, 30))
        sh = getattr(act, "cache", {}).get(id(env))
        if sh is not None:
            r.update(sh.stats())
        card[name] = r
    total = round(float(np.mean([r["score"] for r in card.values()])), 2)
    urban = [r["score"] for k, r in card.items() if k in BENCHMARK]
    ru = [r["score"] for k, r in card.items() if k in RURAL_BENCHMARK]
    out = {"total": total, "urban": round(float(np.mean(urban)), 2) if urban else None,
           "rural": round(float(np.mean(ru)), 2) if ru else None, "scenarios": card}
    if shield:
        out["shielded"] = True
        n = sum(r["steps"] for r in card.values())
        out["long_override_share"] = round(sum(r["long_override_share"] * r["steps"] for r in card.values()) / max(n, 1), 3)
        out["lat_override_share"] = round(sum(r["lat_override_share"] * r["steps"] for r in card.values()) / max(n, 1), 3)
    return out


def compare(new: dict, best: dict, min_gain=0.5, suite="all", urban_tol=1.0, rural_gain_for_tol=5.0):
    """새 모델이 '점수가 오르고 + 어느 시나리오에서도 나빠지지 않을 때'만 승격.
    suite="urban"이면 도시 벤치마크만으로 판단(아직 시골길을 배우기 전 단계),
    "all"이면 도시+시골길 모두 보고 도시 점수 하락도 막는다.

    시나리오별 조건(하나라도 걸리면 거절): 새로 이탈 / 충돌 증가 / 신호·우회전 위반 증가 /
    어린이보호구역 과속 +1초 초과 / 완주율 -5%p 초과 / 시골길 위험 행동 증가 / 시나리오 점수 -5 초과.
    도시 평균 조건: 기본은 -0.5 까지만 허용. 단 suite="all"에서 시골길 평균이 +rural_gain_for_tol(5) 이상 오를 때는
    -urban_tol(1.0) 까지 허용. 1점 = 도시 10개 합으로 10점 = 예) 모든 도시 시나리오에서 차선 오차 +0.2 m
    또는 일반 과속 합 10초 수준의 '승차감·효율' 차이. 안전 항목(이탈·충돌·위반·보호구역 과속)은 시나리오별 조건이
    그대로 막으므로, 이 허용치로 안전이 나빠진 모델이 통과하지는 못한다."""
    reasons = []
    names = list(BENCHMARK) if suite == "urban" else list(ALL_BENCH)
    for k, b in best["scenarios"].items():
        if k not in names:
            continue
        n = new["scenarios"].get(k)
        if n is None:
            continue
        if n["crashed"] and not b["crashed"]:
            reasons.append(f"{k}: 새로 이탈")
        if n.get("collisions", 0) > b.get("collisions", 0):
            reasons.append(f"{k}: 충돌 증가")
        if n["red_runs"] + n.get("right_turn_no_stop", 0) > b["red_runs"] + b.get("right_turn_no_stop", 0):
            reasons.append(f"{k}: 신호·우회전 위반 증가 {b['red_runs'] + b.get('right_turn_no_stop', 0)}"
                           f"→{n['red_runs'] + n.get('right_turn_no_stop', 0)}")
        if n["school_speeding_s"] > b["school_speeding_s"] + 1.0:
            reasons.append(f"{k}: 어린이보호구역 과속 증가 {b['school_speeding_s']}→{n['school_speeding_s']}s")
        if n["completion"] < b["completion"] - 0.05:
            reasons.append(f"{k}: 완주율 하락 {b['completion']}→{n['completion']}")
        if n.get("unsafe_meet", 0) + n.get("ped_unsafe", 0) + n.get("unsafe_overtake", 0) > \
                b.get("unsafe_meet", 0) + b.get("ped_unsafe", 0) + b.get("unsafe_overtake", 0):
            reasons.append(f"{k}: 시골길 위험 행동 증가")
        if n["score"] < b["score"] - 5.0:
            reasons.append(f"{k}: 점수 크게 하락 {b['score']}→{n['score']}")
    if best.get("urban") is not None and new.get("urban") is not None:
        tol = 0.5
        if suite == "all" and best.get("rural") is not None and new.get("rural") is not None \
                and new["rural"] >= best["rural"] + rural_gain_for_tol:
            tol = max(tol, urban_tol)
        if new["urban"] < best["urban"] - tol:
            reasons.append(f"도시 점수 하락 {best['urban']}→{new['urban']} (허용 -{tol})")
    key = "urban" if suite == "urban" else "total"
    if new[key] < best[key] + min_gain:
        reasons.append(f"{'도시' if key == 'urban' else '전체'} 점수 부족 {best[key]}→{new[key]} (필요 +{min_gain})")
    return len(reasons) == 0, reasons


def safety_totals(card: dict, names=None) -> dict:
    """점수표 전체(또는 names)의 안전 항목 합계."""
    sc = {k: v for k, v in card["scenarios"].items() if names is None or k in names}
    return {"crashes": sum(bool(v["crashed"]) for v in sc.values()),
            "collisions": sum(v.get("collisions", 0) for v in sc.values()),
            "signal_violations": sum(v["red_runs"] + v.get("right_turn_no_stop", 0) for v in sc.values()),
            "rural_unsafe": sum(v.get("unsafe_meet", 0) + v.get("ped_unsafe", 0) + v.get("unsafe_overtake", 0) for v in sc.values()),
            "school_speeding_s": round(sum(v["school_speeding_s"] for v in sc.values()), 2)}


def aggregate_dominates(new: dict, best: dict, suite="all", min_gain=0.5):
    """관문 보완 규칙(집계 안전 우위). 시나리오별 조건(compare)에 걸려도 아래를 '모두' 만족하면 승격:
      - 안전 항목 합계가 하나도 나빠지지 않음: 이탈 수, 충돌 수, 신호·우회전 위반 수, 시골길 위험 행동 수
        (각각 ≤ 최고), 어린이보호구역 과속 합 ≤ 최고 + 1.0 s
      - 도시 평균 ≥ 최고 (허용치 없음)
      - 전체 평균 ≥ 최고 + min_gain, (suite="all"이면) 시골길 평균 ≥ 최고 + min_gain
    즉 '어떤 안전 항목도 더 많지 않고 도시도 낮지 않은데 전체·시골이 더 좋은' 모델만. 남은 시나리오별 회귀는
    승격 기록(note)과 점수표에 그대로 남긴다."""
    names = list(BENCHMARK) if suite == "urban" else list(ALL_BENCH)
    n, b = safety_totals(new, names), safety_totals(best, names)
    why = []
    for k in ("crashes", "collisions", "signal_violations", "rural_unsafe"):
        if n[k] > b[k]:
            why.append(f"{k} {b[k]}→{n[k]}")
    if n["school_speeding_s"] > b["school_speeding_s"] + 1.0:
        why.append(f"school_speeding_s {b['school_speeding_s']}→{n['school_speeding_s']}")
    if new["urban"] < best["urban"]:
        why.append(f"도시 {best['urban']}→{new['urban']}")
    if new["total"] < best["total"] + min_gain:
        why.append(f"전체 {best['total']}→{new['total']}")
    if suite == "all" and (new.get("rural") or -1e9) < (best.get("rural") or -1e9) + min_gain:
        why.append(f"시골 {best.get('rural')}→{new.get('rural')}")
    return len(why) == 0, why, n, b


def scorecard_md(card: dict, title: str) -> str:
    cols = ["completion", "crashed", "time_s", "mean_speed_kmh", "mean_lane_dev_m", "speeding_s",
            "school_speeding_s", "red_runs", "right_turn_no_stop", "collisions", "centerline_s",
            "edge_close_s", "sight_overspeed_s", "unsafe_overtake", "unsafe_meet", "yield_ok", "ped_unsafe",
            "smooth_steer", "score"]
    head = "| 시나리오 | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n"
    rows = "".join(f"| {k} | " + " | ".join(str(v.get(c, "")) for c in cols) + " |\n" for k, v in card["scenarios"].items())
    return (f"### {title} (총점 {card['total']}, 도시 {card.get('urban')}, 시골길 {card.get('rural')})\n\n"
            + head + rows)


if __name__ == "__main__":
    import sys
    pol = sys.argv[1] if len(sys.argv) > 1 else "pure_pursuit"
    c = evaluate(pol)
    print(scorecard_md(c, pol))
    print(json.dumps(c, ensure_ascii=False)[:200])
