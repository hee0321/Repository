"""시나리오 모음: 고정 벤치마크(평가용)와 학습용 무작위 시나리오 생성기."""
from __future__ import annotations

import functools

import numpy as np

from .car import CarParams
from .env import AutoDriveEnv
from .rules import uniform_rules
from .track import make_default_track, make_hard_track, make_oval_track, make_random_track


@functools.lru_cache(maxsize=None)
def korea_route(name: str, signal_seed: int = 0):
    from korea_roads.scenarios import SCENARIOS
    from .roads import build_route
    track, rules, meta = build_route(name, SCENARIOS[name], signal_seed=signal_seed)
    return track, rules, {k: v for k, v in meta.items() if k not in ("graph", "data")}


def _synthetic(kind):
    return {"oval_easy": make_oval_track, "default": make_default_track, "hard": make_hard_track}[kind]()


# 고정 벤치마크. 바꾸면 리더보드 점수가 예전 것과 비교가 안 되므로 버전을 올린다.
BENCH_VERSION = "v2.1"  # v2.1: 관측의 제한속도를 앞 구간 미리 보기 값으로 (시나리오는 v2와 같음, 최고 모델 다시 평가)
_V2_NOTE = "v2"  # v2: 시골길 농기계 → 일반 차량 흐름, 실제 OSM 시골길(양평 정배리) 3개 + 넓은 시골길 앞지르기 추가
BENCHMARK = {
    "oval_easy":      dict(kind="oval_easy", laps=1),
    "default_2laps":  dict(kind="default", laps=2),
    "hard_2laps":     dict(kind="hard", laps=2),
    "hard_wet":       dict(kind="hard", laps=1, mu=0.5),
    "rules_track":    dict(kind="default", laps=2, limit_kmh=40, signals=2, lane_ref=-1.6, twoway=True, width=6.4),
    "obstacles_track": dict(kind="default", laps=1, width=7.0, obstacles=[
        dict(s=60, d=0.0, v=0.0, r=1.0), dict(s=150, d=1.5, v=0.0, r=1.0), dict(s=200, d=-0.5, v=5.0, r=1.2)]),
    "gangnam_station": dict(korea="gangnam_station"),
    "gangnam_reverse": dict(korea="gangnam_reverse", signal_seed=1),
    "gangnam_traffic": dict(korea="gangnam_station", obstacles=[
        dict(s=700, d=-1.6, v=6.0, r=1.2), dict(s=400, d=-1.6, v=0.0, r=1.1)]),
    "daechi_school_zone": dict(korea="daechi_school_zone"),
}


def make_env(spec: dict, seed: int = 0, **kw) -> AutoDriveEnv:
    """spec(dict) → 환경. 벤치마크/학습/실패 사례 재현이 모두 이 함수를 쓴다."""
    params = CarParams(mu=spec.get("mu", 0.9))
    noise = spec.get("lidar_noise", 0.0)
    if "korea" in spec:
        track, rules, meta = korea_route(spec["korea"], spec.get("signal_seed", 0))
        max_steps = spec.get("max_steps", int(track.length / 2.0 / 0.05))
        obstacles = spec.get("obstacles")
        if rules.rural:   # 실제 OSM 시골길: 밤 여부·차량 흐름을 바꿔 가며 연습 (캐시된 규칙은 복사해서 수정)
            import dataclasses
            if "night" in spec:
                rules = dataclasses.replace(rules, light=0.15 if spec["night"] else 1.0)
            if obstacles is None:
                if spec.get("traffic_seed") is not None:
                    from .rural import rural_traffic
                    narrow = track.width < 6.0
                    obstacles = rural_traffic(np.random.default_rng(spec["traffic_seed"]), track, track.width, narrow,
                                              rules.lane_ref_at(0.0), spec.get("difficulty", 1.0))
                else:
                    obstacles = meta.get("traffic")
        return AutoDriveEnv(track=track, rules=rules, params=params, lidar_noise=noise, seed=seed,
                            max_steps=max_steps, start_s=spec.get("start_s"), name=spec["korea"],
                            obstacles=obstacles, **kw)
    if spec.get("kind") == "rural":
        from .rural import make_rural_road
        track, rules, obs = make_rural_road(spec.get("road_seed", seed), spec.get("difficulty", 1.0),
                                            spec.get("night"), spec.get("length", 900.0), spec.get("traffic", True))
        max_steps = spec.get("max_steps", int(track.length / 1.5 / 0.05))
        return AutoDriveEnv(track=track, rules=rules, params=params, lidar_noise=noise, seed=seed,
                            max_steps=max_steps, start_s=spec.get("start_s"), name=f"rural_{spec.get('road_seed')}",
                            obstacles=spec.get("obstacles", obs), **kw)
    if spec.get("kind") == "random":
        track = make_random_track(spec.get("track_seed", seed), width=spec.get("width"),
                                  difficulty=spec.get("difficulty", 1.0))
    else:
        track = _synthetic(spec.get("kind", "default"))
        if spec.get("width"):
            from .track import Track
            track = Track(track.center, width=spec["width"], name=track.name)
    rules = uniform_rules(track.length, spec.get("limit_kmh"), spec.get("signals", 0), closed=True,
                          seed=spec.get("signal_seed", 0), lane_ref=spec.get("lane_ref", 0.0),
                          twoway=spec.get("twoway", False))
    laps = spec.get("laps", 1)
    max_steps = spec.get("max_steps", int(laps * track.length / 2.0 / 0.05) + 400)
    return AutoDriveEnv(track=track, rules=rules, params=params, lidar_noise=noise, seed=seed, target_laps=laps,
                        max_steps=max_steps, start_s=spec.get("start_s"), name=spec.get("kind"),
                        obstacles=spec.get("obstacles"), **kw)


# --- 학습용 커리큘럼 + 도메인 무작위화 ---
STAGES = {
    1: "쉬운 트랙(타원·완만한 무작위)",
    2: "센 굽이·좁은 트랙·젖은 노면",
    3: "실제 한국 도로 + 규칙(제한속도·어린이보호구역·신호·우측통행) + 장애물",
    4: "시골길(좁은 길·비포장·밤·보행자·일반 차량 흐름: 마주 오는 차·속도 제각각 앞차·옆길 진입 차·허용 구간 앞지르기)",
}

# 시골길 벤치마크 (도시 벤치마크와 따로 점수 냄)
RURAL_BENCHMARK = {
    "rural_day_narrow":   dict(kind="rural", road_seed=11, difficulty=1.0, night=False),
    "rural_night_gravel": dict(kind="rural", road_seed=23, difficulty=1.2, night=True),
    "rural_curvy":        dict(kind="rural", road_seed=37, difficulty=1.6, night=False),
    "rural_village":      dict(kind="rural", road_seed=48, difficulty=1.0, night=True),
    "rural_wide_traffic": dict(kind="rural", road_seed=134, difficulty=1.0, night=False),   # 중앙선 있는 2차로: 앞지르기·옆길 진입
    # 실제 한국 시골길 (OpenStreetMap, 경기 양평군 서종면 정배리)
    "jeongbae_village":   dict(korea="jeongbae_village"),
    "jeongbae_night":     dict(korea="jeongbae_night"),
    "jeongbae_lane":      dict(korea="jeongbae_lane"),
}
RURAL_OSM = ["jeongbae_village", "jeongbae_night", "jeongbae_lane"]


def sample_train_spec(rng: np.random.Generator, stage: int, failures: list | None = None) -> dict:
    """stage까지의 시나리오를 섞어서 하나 뽑는다(앞 단계도 조금씩 계속 연습 = 잊어버림 방지)."""
    if failures and rng.random() < 0.25:
        f = dict(failures[rng.integers(len(failures))])
        f["start_s"] = max(0.0, f.get("fail_s", 0.0) - 40.0) if "korea" in f else None
        f.pop("fail_s", None)
        f["max_steps"] = 1200
        return f
    # 앞 단계 복습 비율: 보통 30%, 시골길 단계에서는 도시 실력을 잊지 않도록 50% (그중 대부분 실제 도시 도로)
    review = 0.5 if stage >= 4 else 0.3
    if rng.random() < review:
        st = 3 if (stage >= 4 and rng.random() < 0.7) else int(rng.integers(1, stage + 1))
    else:
        st = stage
    spec = {"mu": float(rng.uniform(0.5, 1.0)), "lidar_noise": float(rng.uniform(0.0, 0.15)), "max_steps": 1200}
    if st == 1:
        spec.update(kind="random", track_seed=int(rng.integers(1e6)), difficulty=0.5, width=float(rng.uniform(6.5, 8.0)))
        spec["mu"] = max(spec["mu"], 0.8)
    elif st == 2:
        if rng.random() < 0.3:
            spec.update(kind="hard")
        else:
            spec.update(kind="random", track_seed=int(rng.integers(1e6)), difficulty=1.6)
    elif st == 4:
        spec.update(_rural_spec(rng))
    else:
        r = rng.random()
        if r < 0.15:
            spec.update(kind="default", width=7.0, obstacles=[
                dict(s=float(rng.uniform(30, 220)), d=float(rng.uniform(-1.5, 1.5)), v=0.0, r=1.0),
                dict(s=float(rng.uniform(30, 220)), d=float(rng.uniform(-1, 1)), v=float(rng.uniform(3, 7)), r=1.0)])
        elif r < 0.6:
            # 역방향(강남역 우회전 일시정지)도 학습에 넣음 — 예전엔 실패 사례로만 들어가서 우회전 일시정지를 거의 못 배웠음
            spec.update(korea=str(rng.choice(["gangnam_station", "gangnam_reverse", "daechi_school_zone"])),
                        signal_seed=int(rng.integers(5)))
            spec["max_steps"] = 1500
        else:
            spec.update(kind="random", track_seed=int(rng.integers(1e6)), difficulty=1.2,
                        limit_kmh=float(rng.choice([30, 40, 50, 60])), signals=int(rng.integers(0, 4)),
                        signal_seed=int(rng.integers(1e6)), lane_ref=-1.6, twoway=True, width=6.4)
    return spec


def _rural_spec(rng):
    """시골길 하나: 35% 실제 OSM 시골길(양평 정배리, 밤·차량 흐름 무작위, 중간 출발), 65% 절차 생성."""
    if rng.random() < 0.35:
        return dict(korea=str(rng.choice(RURAL_OSM)), traffic_seed=int(rng.integers(1e6)),
                    night=bool(rng.random() < 0.4), difficulty=float(rng.uniform(0.6, 1.6)), max_steps=1500)
    return dict(kind="rural", road_seed=int(rng.integers(1000, 1e6)), difficulty=float(rng.uniform(0.6, 1.8)),
                length=600.0, max_steps=1500)


def sample_mixed_spec(rng: np.random.Generator, failures: list | None = None, urban_ratio: float = 0.5,
                      suite: str | None = None) -> dict:
    """시골길 사이클용 '섞기' 샘플러: 도시(1~3단계, 실제 한국 도시 도로 위주)와 시골길을 urban_ratio 대 나머지로.
    suite="urban"/"rural" 이면 그쪽만 (환경 절반은 도시, 절반은 시골로 고정해서 PPO 배치마다 섞이게 할 때 사용).
    실패 사례는 같은 쪽(도시/시골) 것만 20% 섞는다."""
    if suite is None:
        suite = "urban" if rng.random() < urban_ratio else "rural"
    if failures:
        pool = [f for f in failures if (f.get("kind") == "rural" or f.get("korea") in RURAL_OSM) == (suite == "rural")]
        if pool and rng.random() < 0.2:
            f = dict(pool[rng.integers(len(pool))])
            f["start_s"] = max(0.0, f.get("fail_s", 0.0) - 40.0) if "korea" in f else None
            f.pop("fail_s", None)
            f["max_steps"] = 1200
            return f
    if suite == "rural":
        spec = {"mu": float(rng.uniform(0.7, 1.0)), "lidar_noise": float(rng.uniform(0.0, 0.15)), "max_steps": 1500}
        spec.update(_rural_spec(rng))
        return spec
    st = 3 if rng.random() < 0.7 else int(rng.integers(1, 3))
    return sample_train_spec(rng, st, None) if st != 3 else _stage3(rng)


def _stage3(rng):
    rng2 = rng
    while True:
        sp = sample_train_spec(rng2, 3, None)
        if sp.get("kind") != "rural" and sp.get("korea") not in RURAL_OSM:
            return sp
