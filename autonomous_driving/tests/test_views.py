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


# ---------- 전기차(EV) + 운전석 중심 화면 ----------
from autodrive.car import BicycleCar  # noqa: E402
from autodrive.ev import EVPowertrain  # noqa: E402
from autodrive.views import UIState, make_ui  # noqa: E402
from autodrive.views.hud import DriverMainUI  # noqa: E402
from autodrive.views.motor import MotorBayView  # noqa: E402


def test_ev_powertrain_instant_torque_regen_and_soc():
    ev = EVPowertrain()
    soc0 = ev.soc
    # 0 km/h 근처에서도 최대 가속 3 m/s²에 필요한 토크가 바로 나옴(변속 없음)
    ev.step(0.5, 3.0, 0.05)
    assert ev.torque > 100 and ev.p_bat_kw > 0
    for _ in range(100):
        ev.step(15.0, 1.0, 0.05)
    assert ev.soc < soc0 and ev.e_used_wh > 0
    used = ev.soc
    # 완만한 감속은 전부 회생, 배터리가 다시 참
    for _ in range(40):
        ev.step(15.0, -1.5, 0.05)
    assert ev.regen_kw > 5 and ev.fric_frac < 1e-6 and ev.soc > used and ev.e_regen_wh > 0
    # 급제동은 회생 한계를 넘어 마찰 브레이크가 나머지(블렌딩)
    ev.step(15.0, -6.0, 0.05)
    assert 0.3 < ev.fric_frac < 1.0 and ev.regen_kw > 0
    # 정지 직전엔 회생이 줄어듦
    ev.step(0.3, -2.0, 0.05)
    assert ev.fric_frac > 0.8


def test_ev_does_not_change_vehicle_motion():
    a, b = BicycleCar(v=5.0), BicycleCar(v=5.0)
    b.ev = None  # EV 계산 없이 같은 공식(아래)과 비교
    for k in range(50):
        steer, acc = 0.1 * np.sin(k / 5), 3.0 if k < 20 else -4.0
        a.step(steer, acc, 0.05)
        v = float(np.clip(b.v + np.clip(acc, -6, 3) * 0.05, 0, 20))
        b.v = v
    assert abs(a.v - b.v) < 1e-9 and a.ev.e_used_wh > 0


def test_motor_bay_and_alias():
    from autodrive.views.engine import EngineView
    assert EngineView is MotorBayView
    env, rec = _rollout(steps=40)
    fr = interpolate(rec.rows, 25)
    assert all(k in fr[0] for k in ("soc", "p_motor_kw", "regen_kw", "motor_rpm", "torque"))
    im = MotorBayView(env).render(fr[-1])
    assert im.size == (640, 360)
    assert MotorBayView(env, 320, 180).render(fr[-1]).size == (320, 180)


def test_driver_main_ui_layouts():
    env, rec = _rollout(steps=40)
    f = interpolate(rec.rows, 25)[-1]
    for main in ("driver", "top", "motor", "satellite"):
        ui = DriverMainUI(env, rec.kinds, main=main, show=("motor", "satellite"))
        im = np.asarray(ui.render(f))
        assert im.shape == (720, 1280, 3) and im.std() > 5
    f2 = dict(f, mode="TAKEOVER", t=0.1)
    im = np.asarray(DriverMainUI(env, rec.kinds).render(f2))
    assert (im[30:80, 560:720, 0].astype(int) - im[30:80, 560:720, 2]).mean() > 60  # 빨간 개입 요청 띠
    assert make_ui(env, rec.kinds, "all").size == (1280, 720)


def test_ui_state_keys_and_slide():
    s = UIState()
    assert s.main == "driver" and s.show == [] and s.layout == "main"
    s.handle_key("m"); assert s.main == "top"
    s.handle_key("tab"); s.handle_key("tab"); s.handle_key("m"); assert s.main == "driver"
    s.handle_key("1"); assert s.show == ["motor"] and s.slide["motor"] == 0.0
    s.tick(0.1); assert 0 < s.slide["motor"] < 1
    s.tick(1.0); assert s.slide["motor"] == 1.0
    s.handle_key("1"); s.tick(1.0); assert s.show == []
    s.handle_key("g"); assert s.layout == "grid"
    s.handle_key("q"); assert s.quit


def test_render_main_layout_default(tmp_path):
    env, rec = _rollout(steps=30)
    outs = render_views(env, rec.rows, rec.kinds, "main", str(tmp_path / "m"), fps=25, mp4=False, gif=True,
                        still_at=0.5, progress=False, show=("motor",))
    assert any(o.endswith(".gif") for o in outs) and any(o.endswith("_still.png") for o in outs)


def test_autopilot_view_options():
    import argparse
    from autopilot import parse_show, resolve_view
    assert parse_show("engine, satellite,x") == ["motor", "satellite"]
    assert resolve_view(argparse.Namespace(view="main", layout="grid")) == "all"
    assert resolve_view(argparse.Namespace(view="main", layout="main")) == "main"
    assert resolve_view(argparse.Namespace(view="engine", layout="main")) == "engine"
