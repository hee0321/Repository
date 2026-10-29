"""시점 렌더러 빠른 점검: python -m pytest tests/test_views.py -q"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autodrive.controllers import PurePursuitController  # noqa: E402
from autodrive.scenarios import RURAL_BENCHMARK, make_env  # noqa: E402
from autodrive.views import StateRecorder, interpolate, pick_window, render_views  # noqa: E402
from autodrive.views.driver import DriverView, clip_near  # noqa: E402
from autodrive.views.engine import EngineView  # noqa: E402
from autodrive.views.satellite_panel import SatelliteView  # noqa: E402


def _rollout(name="rural_night_gravel", steps=60):
    env = make_env(RURAL_BENCHMARK[name], seed=0)
    obs, _ = env.reset(seed=0)
    ctrl = PurePursuitController(env)
    rec = StateRecorder(env)
    for _ in range(steps):
        a = ctrl.act(obs)
        obs, *_ = env.step(a)
        rec.record(env, a, {"mode": "TAKEOVER", "risks": ["시험"]})
    return env, rec


def test_camera_left_right_and_behind():
    env, rec = _rollout(steps=5)
    dv = DriverView(env, kinds=rec.kinds)
    f = interpolate(rec.rows, 25)[-1]
    dv.cam.set_pose(f["x"], f["y"], f["yaw"])
    s = f["s"] + 20
    left = np.r_[dv.geo.at(s, 2.0), 0.0]   # 법선(+) = 진행 방향 왼쪽
    right = np.r_[dv.geo.at(s, -2.0), 0.0]
    uL = dv.cam.proj(dv.cam.to_cam(left[None]))[0, 0]
    uR = dv.cam.proj(dv.cam.to_cam(right[None]))[0, 0]
    assert uL < uR  # 진행 방향 왼쪽(+법선)은 화면 왼쪽 (좌우 뒤집힘 방지)
    # 카메라 뒤 점은 잘려 나간다
    behind = np.r_[dv.geo.at(f["s"] - 10, 0.0), 0.0]
    assert dv.cam.to_cam(behind[None])[0, 2] < 0
    tri = dv.cam.to_cam(np.array([behind, np.r_[dv.geo.at(s, 0.0), 0.0], np.r_[dv.geo.at(s, 1.0), 0.0]]))
    C = clip_near(tri)
    assert C is not None and (C[:, 2] >= 0.35 - 1e-9).all()


def test_render_panels_and_interpolation():
    env, rec = _rollout()
    fr = interpolate(rec.rows, 25)
    assert len(fr) >= int((rec.rows[-1]["t"] - rec.rows[0]["t"]) * 25)
    assert abs(fr[1]["t"] - fr[0]["t"] - 0.04) < 1e-6
    f = fr[len(fr) // 2]
    for view in (DriverView(env, kinds=rec.kinds), EngineView(env), SatelliteView(env, kinds=rec.kinds)):
        im = np.asarray(view.render(f))
        assert im.shape == (360, 640, 3) and im.std() > 5
    assert "tractor" not in DriverView(env, kinds=rec.kinds).kinds  # 트랙터는 일반 차로 그림
    t0, t1 = pick_window(rec.rows, 1.0)
    assert t1 - t0 <= 1.0 + 1e-6


def test_render_views_writes_files(tmp_path):
    env, rec = _rollout(steps=30)
    outs = render_views(env, rec.rows, rec.kinds, "all", str(tmp_path / "clip"), fps=25, mp4=False, gif=True,
                        still_at=0.5, progress=False)
    assert any(o.endswith(".gif") for o in outs) and any(o.endswith("_still.png") for o in outs)
    for o in outs:
        assert os.path.getsize(o) > 1000
