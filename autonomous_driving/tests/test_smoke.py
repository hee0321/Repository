"""빠른 점검: python -m pytest tests -q"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autodrive import AutoDriveEnv, BicycleCar, Lidar, PurePursuitController, make_default_track  # noqa: E402
from autodrive.bench import make_policy, run_episode  # noqa: E402
from autodrive.rules import Signal, ViolationLog, uniform_rules  # noqa: E402
from autodrive.scenarios import BENCHMARK, RURAL_BENCHMARK, make_env  # noqa: E402
from autodrive.track import Track  # noqa: E402


def test_bicycle_straight_and_turn():
    car = BicycleCar(v=10.0)
    for _ in range(20):
        car.step(0.0, 0.0, 0.05)
    assert abs(car.x - 10.0) < 1e-6 and abs(car.y) < 1e-9
    car2 = BicycleCar(v=5.0)
    for _ in range(100):
        car2.step(0.3, 0.0, 0.05)
    assert car2.y > 0  # 왼쪽(+) 조향이면 왼쪽으로 돈다


def test_friction_limits_lateral_accel():
    car = BicycleCar(v=15.0)
    car.p.mu = 0.5
    car.steer = 0.5
    car.step(0.5, 0.0, 0.05)
    assert car.slipping and abs(car.lat_acc) <= 0.5 * 9.81 + 1e-6


def test_track_projection_square_and_open():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    tr = Track(sq, width=2.0)
    assert abs(tr.length - 40) < 1e-9
    s, d, _ = tr.project(np.array([5.0, 0.5]))
    assert abs(s - 5) < 1e-9 and abs(d - 0.5) < 1e-9
    op = Track(np.array([[0, 0], [10, 0], [20, 0]], float), closed=False)
    assert abs(op.length - 20) < 1e-9 and op.walls.shape[0] == 4


def test_lidar_hits_wall_and_obstacle():
    walls = np.array([[[5.0, -10.0], [5.0, 10.0]]])
    lid = Lidar(n_rays=3, fov_deg=90, max_range=30)
    r = lid.scan(np.zeros(2), 0.0, walls)
    assert abs(r[1] - 5.0) < 1e-9 and abs(r[0] - 5 * np.sqrt(2)) < 1e-6
    r2 = lid.scan(np.zeros(2), 0.0, walls, np.array([[3.0, 0.0, 1.0]]))
    assert abs(r2[1] - 2.0) < 1e-9


def test_signal_cycle_and_red_run():
    sg = Signal(s=10.0, green=5, yellow=1, red=4)
    assert sg.state(0) == "green" and sg.state(5.5) == "yellow" and sg.state(7) == "red"
    rules = uniform_rules(100, 50, 0, closed=False)
    rules.signals = [sg]
    log = ViolationLog()
    log.update(rules, 9.0, 11.0, 7.0, 5.0, 0.0, 0.05, 0.9)
    assert log.red_runs == 1


def test_env_api_shapes():
    env = AutoDriveEnv(seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (env.obs_dim,)
    obs, r, term, trunc, info = env.step(np.zeros(2))
    assert obs.shape == (env.obs_dim,) and isinstance(r, float)


def test_pure_pursuit_completes_lap_without_crash():
    env = AutoDriveEnv(target_laps=1, max_steps=5000, seed=0)
    ctrl = PurePursuitController(env)
    obs, info = env.reset(seed=0)
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(ctrl.act(obs))
        done = term or trunc
    assert not info["crashed"] and info["laps"] >= 1.0


def test_default_track_boundaries_do_not_self_intersect():
    tr = make_default_track()
    assert 1 / np.abs(tr.curvature).max() > tr.width / 2


def test_baseline_obstacles_and_rules_track():
    act = make_policy("pure_pursuit")
    for name in ("obstacles_track", "rules_track"):
        r = run_episode(make_env(BENCHMARK[name]), act)
        assert r["finished"] and not r["crashed"] and r["red_runs"] == 0, (name, r)


def test_korea_route_from_cache_runs():
    """캐시된 OSM 데이터(korea_roads/data)로 강남역 경로를 만들고 기준 제어기가 규칙을 지키며 완주."""
    r = run_episode(make_env(BENCHMARK["gangnam_reverse"]), make_policy("pure_pursuit"))
    assert r["finished"] and not r["crashed"]
    assert r["red_runs"] == 0 and r["right_turn_no_stop"] == 0 and r["school_speeding_s"] < 1.0


def test_rural_road_baseline_safe():
    r = run_episode(make_env(RURAL_BENCHMARK["rural_night_gravel"]), make_policy("pure_pursuit"))
    assert r["finished"] and not r["crashed"]
    assert r["unsafe_meet"] == 0 and r["ped_unsafe"] == 0 and r["unsafe_overtake"] == 0


def test_autopilot_engage_gate(tmp_path):
    import json
    from autopilot import engage_check, load_yaml
    cfg = load_yaml(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rules", "autopilot.yaml"))
    good = {"total": 90, "scenarios": {"a": {"completion": 1.0, "crashed": False, "red_runs": 0, "school_speeding_s": 0.0}}}
    bad = {"total": 50, "scenarios": {"a": {"completion": 0.5, "crashed": True, "red_runs": 1, "school_speeding_s": 0.0}}}
    p1, p2 = tmp_path / "g.json", tmp_path / "b.json"
    p1.write_text(json.dumps(good)); p2.write_text(json.dumps(bad))
    assert engage_check(str(p1), cfg, "a")[0] is True
    assert engage_check(str(p2), cfg, "a")[0] is False
