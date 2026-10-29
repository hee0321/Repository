"""시골길 수정 테스트: 농기계 없음, 일반 차량 흐름, 실제 OSM 시골길, 승격 관문 허용치."""
import numpy as np

from autodrive.bench import compare, evaluate
from autodrive.rural import make_rural_road
from autodrive.scenarios import RURAL_BENCHMARK, make_env, sample_mixed_spec


def test_no_farm_machinery_and_car_traffic():
    kinds = set()
    for sd in range(40):
        _, _, obs = make_rural_road(sd, 1.2, None, 900.0, True)
        kinds |= {o["kind"] for o in obs}
    assert "tractor" not in kinds
    assert {"car", "oncoming", "entering", "ped"} <= kinds


def test_real_osm_rural_routes_load_from_cache():
    for name in ("jeongbae_village", "jeongbae_night", "jeongbae_lane"):
        env = make_env(RURAL_BENCHMARK[name], seed=0)
        obs, _ = env.reset(seed=0)
        assert env.rules.rural and env.track.length > 500 and np.all(np.isfinite(obs))
    assert make_env(RURAL_BENCHMARK["jeongbae_lane"]).track.width < 6.0      # 마을 안길 = 좁은 1차로
    assert make_env(RURAL_BENCHMARK["jeongbae_village"]).rules.school.sum() > 100  # 정배초 어린이보호구역


def test_baseline_legal_overtake_and_entering_car():
    c = evaluate("pure_pursuit", scenarios=["rural_wide_traffic", "jeongbae_village"])
    for r in c["scenarios"].values():
        assert r["finished"] and not r["crashed"]
        assert r["unsafe_overtake"] == 0 and r["centerline_s"] == 0
    assert sum(r["legal_overtake"] for r in c["scenarios"].values()) >= 1


def test_mixed_sampler_balances_suites():
    rng = np.random.default_rng(0)
    specs = [sample_mixed_spec(rng, None, 0.5) for _ in range(400)]
    rural = sum(s.get("kind") == "rural" or str(s.get("korea", "")).startswith("jeongbae") for s in specs)
    assert 150 < rural < 250


def _card(urban, rural, score_drop=0.0):
    sc = {"a": dict(crashed=False, red_runs=0, school_speeding_s=0.0, completion=1.0, score=90.0 - score_drop)}
    return {"urban": urban, "rural": rural, "total": (urban + rural) / 2, "scenarios": sc}


def test_gate_urban_tolerance_only_with_rural_gain():
    best = _card(74.0, 40.0)
    assert compare(_card(73.2, 46.0), best, suite="all")[0]          # 도시 -0.8, 시골 +6 → 허용
    assert not compare(_card(73.2, 43.0), best, suite="all")[0]      # 시골 +3 뿐이면 -0.5 까지만
    assert not compare(_card(72.5, 60.0), best, suite="all")[0]      # 도시 -1.5 는 항상 거절
