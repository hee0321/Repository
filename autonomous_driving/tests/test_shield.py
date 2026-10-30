"""안전 실드: 늘 '풀 가속'만 하는 정책도 실드를 씌우면 빨간불·우회전 일시정지·앞차를 지키는지."""
import numpy as np

from autodrive.bench import ALL_BENCH, run_episode
from autodrive.controllers import PurePursuitController
from autodrive.scenarios import make_env
from autodrive.shield import SafetyShield


def _reckless(env):
    steer = PurePursuitController(env)   # 조향만 기준 제어기, 가속은 늘 최대

    def act(obs):
        a = steer.act(obs)
        return np.array([a[0], 1.0], np.float32)
    return act


def _run(name, shield):
    env = make_env(ALL_BENCH[name], seed=0)
    pol = _reckless(env)
    sh = SafetyShield(env) if shield else None
    return run_episode(env, lambda e, o: sh.filter(o, pol(o)) if sh else pol(o), seed=0), sh


def test_shield_stops_for_right_turn_on_red():
    raw, _ = _run("gangnam_reverse", False)
    r, sh = _run("gangnam_reverse", True)
    assert raw["red_runs"] + raw["right_turn_no_stop"] > 0 or raw["crashed"]   # 실드 없으면 위반 또는 이탈
    assert r["red_runs"] == 0 and r["right_turn_no_stop"] == 0 and not r["crashed"]
    assert sh.stats()["long_override_share"] > 0


def test_shield_keeps_distance_in_traffic():
    r, _ = _run("gangnam_traffic", True)
    assert not r["crashed"] and r["collisions"] == 0 and r["unsafe_overtake"] == 0
