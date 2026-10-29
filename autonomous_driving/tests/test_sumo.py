"""SUMO 스모크 테스트 (SUMO가 설치되어 있지 않으면 건너뜀)."""
import pytest

pytest.importorskip("sumolib")
pytest.importorskip("pyproj")
try:
    import libsumo  # noqa: F401
except ImportError:
    pytest.importorskip("traci")


def test_expert_arrives_without_violations():
    from sumo_kr.env import SumoKREnv
    env = SumoKREnv("gangnam_reverse")
    env.reset(seed=202)
    done = False
    while not done:
        _, _, te, tr, info = env.step(env.expert_action())
        done = te or tr
    st = info["stats"]
    env.close()
    assert info["reason"] == "arrived"
    assert st.get("collisions", 0) == 0
    assert st.get("red_runs", 0) == 0
    assert st.get("right_turn_no_stop", 0) == 0
    assert st.get("right_turn_ok", 0) >= 1   # 빨간불 우회전 전 일시정지가 실제로 일어남
