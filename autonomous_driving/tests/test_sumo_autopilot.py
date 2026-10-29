"""SUMO 자율 주행 모드(안전 감시기·인간 개입 요청) 스모크 테스트. SUMO 없으면 건너뜀. 가중치 없이 돌아감."""
import copy
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytest.importorskip("sumolib")
pytest.importorskip("pyproj")
try:
    import libsumo  # noqa: F401
except ImportError:
    pytest.importorskip("traci")

import autopilot as AP2D  # noqa: E402
from sumo_kr import autopilot as A  # noqa: E402


@pytest.fixture(autouse=True)
def _tmp_results(tmp_path, monkeypatch):  # 저장소 results/ 를 덮어쓰지 않게
    monkeypatch.setattr(A, "R", str(tmp_path))


class Reckless(A.SumoAutopilot):
    """일부러 위험한 '모델': 항상 최대 가속(감시기가 잡아야 함)."""

    def model_eval(self, obs):
        return np.array([0.0, 1.0], np.float32), np.array([0.1, 0.1], np.float32), 0.0, 0.0


def _cfg(**monitor):
    cfg = copy.deepcopy(AP2D.load_yaml(os.path.join(A.ROOT, "rules", "autopilot.yaml")))
    cfg["monitor"].update(monitor)
    return cfg


def _run(name, seed, cfg, monkeypatch, model=object()):
    monkeypatch.setattr(A, "SumoAutopilot", Reckless)
    return A.run_episode(name, seed, model, None, cfg)


def test_same_monitor_and_config_as_2d():
    # 감시 로직(위험 판단·모드 전환·안전 정지)은 2D Autopilot 것을 그대로 씀 (복사본 아님)
    assert issubclass(A.SumoAutopilot, AP2D.Autopilot)
    for fn in ("risks", "step"):
        assert getattr(A.SumoAutopilot, fn) is getattr(AP2D.Autopilot, fn)
    assert A.load_yaml is AP2D.load_yaml


def test_expert_only_no_takeovers():
    r = A.run_episode("gangnam_reverse", 202, None, None, _cfg())
    assert r["arrived"] and r["collisions"] == 0 and r["red_runs"] == 0
    assert r["autonomous_share"] == 0 and r["takeovers"] == 0


def test_reckless_model_is_taken_over(monkeypatch):
    r = _run("gangnam_reverse", 202, _cfg(), monkeypatch)
    assert r["takeovers"] > 0 and "모델·기준 운전자 차이" in r["takeover_reasons"]
    assert r["collisions"] == 0 and r["arrived"]
    assert r["autonomous_share"] < 0.5
    ev = [json.loads(x) for x in open(os.path.join(A.R, "sumo_autopilot_gangnam_reverse@202_events.jsonl"), encoding="utf-8")]
    assert any(e["action"] == A.TAKEOVER and e["reasons"] for e in ev)


def test_ttc_speeding_and_safe_stop(monkeypatch):
    # '자신 없음' 감시를 끄고 나머지(TTC·과속·안전 정지)만으로도 충돌 없이 막는지
    r = _run("daechi_school", 101, _cfg(disagreement=99.0), monkeypatch)
    assert r["takeover_reasons"].get("과속(+5km/h)", 0) > 0
    assert r["takeover_reasons"].get("충돌 위험(TTC<2s)", 0) > 0
    assert r["safe_stops"] >= 1
    assert r["collisions"] == 0


def test_2d_entry_point_delegates(monkeypatch):
    import argparse
    args = argparse.Namespace(scenario="sumo_gangnam_reverse", model="none", force=False, seed=202, quiet=True, alert_cmd=None)
    r = AP2D.run(args)
    assert r["scenario"] == "gangnam_reverse" and r["arrived"]
