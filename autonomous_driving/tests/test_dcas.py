"""DCAS 01 (UN R171 01) 시나리오·판정·자율 완주 지표 시험."""
import json
import os
import subprocess
import sys

import numpy as np

from autodrive import bench, dcas

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_suite_has_refs_and_desc():
    assert len(dcas.DCAS_NAMES) == 20
    for n, sc in dcas.DCAS01.items():
        assert sc["ref"] and sc["desc"], n
        env = dcas.make_env(n)
        assert env.track.length > 100


def test_expert_passes_key_scenarios():
    act = bench.make_policy("pure_pursuit")
    for n in ("dcas_lk_curve_85", "dcas_lc_request", "dcas_lead_brake_6", "dcas_ped_crossing", "dcas_lc_oncoming_refuse"):
        r = dcas.run(dcas.make_env(n), act)
        assert r["dcas_pass"], (n, r["checks"])
        assert r["autonomous"], n


def test_lane_change_request_needs_indicator_3s():
    r = dcas.run(dcas.make_env("dcas_lc_request"), bench.make_policy("pure_pursuit"))
    assert r["lane_changes"] >= 1 and all(x is not None and x >= 3.0 for x in r["lcp_lead_s"])


def test_straight_only_fails_lane_change_and_collision():
    zero = lambda env, obs: np.array([0.0, 0.3], np.float32)  # noqa: E731  (직진·가속만)
    r = dcas.run(dcas.make_env("dcas_lc_request"), zero)
    assert not r["dcas_pass"] and not r["checks"]["lane_change_completed (4.2.5.1.3)"]
    r = dcas.run(dcas.make_env("dcas_ped_standing"), zero)   # 서 있는 보행자를 그대로 들이받음
    assert not r["dcas_pass"] and not r["autonomous"]


def test_autonomous_completion_metric():
    ok = dict(finished=True, crashed=False, collisions=0, takeovers=0)
    assert dcas.autonomous_completion(ok)
    for k, v in (("finished", False), ("crashed", True), ("collisions", 1), ("takeovers", 2)):
        assert not dcas.autonomous_completion({**ok, k: v})


def test_gate_rules():
    sys.path.insert(0, os.path.join(HERE, "tools"))
    import vision_gate as g
    b = dict(v22_total=70, dcas_total=60, mean=65, v22_auto=10, dcas_auto=10, auto=20, n_auto=38, dcas_pass=5,
             v22_finished=12, crashes=3, collisions=1, signal=1, rural_unsafe=3, dcas_crashes=7, dcas_collisions=7)
    assert g.decide({**b, "mean": 66}, b)[0]
    assert not g.decide({**b, "mean": 66, "crashes": 4}, b)[0]           # 안전 나빠짐
    assert not g.decide({**b, "mean": 70, "auto": 19}, b)[0]             # 자율 완주 줄어듦
    assert g.decide({**b, "auto": 21}, b)[0]                             # 점수 같아도 자율 완주 늘면 승격
    assert not g.decide({**b, "mean": 65.1}, b)[0]                       # 향상 부족
