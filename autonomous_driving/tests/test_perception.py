"""인식 시제품 시험: 라벨 그리기, 거리 추정(정답 라벨로 측정), 추적·계획기 보수 동작, 관측 오염 금지."""
import numpy as np
import pytest

from autodrive import dcas
from autodrive.controllers import PurePursuitController
from autodrive.perception.detect import Geometry, BOTTOM_H, detect
from autodrive.perception.planner import SafetyPlanner, ego_path
from autodrive.perception.track import Odometry, Tracker, Track
from autodrive.vision.camera import CameraMount, LABELS, LBL, camera_for


def _drive(name, steps=80):
    env = dcas.make_env(name)
    obs, _ = env.reset(seed=0)
    ex = PurePursuitController(env)
    for _ in range(steps):
        obs, *_ = env.step(ex.act(obs))
    return env


def test_labels_include_cars_and_lamps():
    env = _drive("dcas_slower_lead", 200)
    m = CameraMount(width=256, height_px=144)
    img, c, i = camera_for(env, mount=m).render_with_labels()
    assert LBL["car"] in c and LBL["road"] in c and LBL["marking"] in c
    assert img.shape == (144, 256, 3)


def test_monocular_distance_on_synthetic_labels():
    """정답 분할로 거리를 재면 앞차 오차가 3 m 안 (장면 기하 가정 아래)."""
    env = _drive("dcas_slower_lead", 200)
    m = CameraMount(width=256, height_px=144)
    img, c, i = camera_for(env, mount=m).render_with_labels()
    dets, _ = detect(c, np.ones_like(c, float), Geometry(m))
    cars = [d for d in dets if d.cls == "car"]
    assert cars
    # 정답 앞차 거리
    k = int(np.unique(i[c == LBL["car"]])[0]) - 1
    p = env.track.point_at(env.ob_s[k]) + env.track.normal_at(env.ob_s[k]) * env.ob_d[k]
    cy, sy = np.cos(env.car.yaw), np.sin(env.car.yaw)
    d = p - (env.car.x, env.car.y)
    gt = float(d[0] * cy + d[1] * sy)
    err = abs(cars[0].fwd + 2.45 - gt)
    assert err < 3.0, (cars[0].fwd + 2.45, gt, err)
    assert cars[0].sigma > 0


def test_tracker_predicts_forward():
    odo = Odometry()
    tr = Tracker()
    from autodrive.perception.detect import Detection
    for t in range(25):
        odo.step(10.0, 0.0, 0.1)
        # 앞에 같은 속도로 달리는 차: 카메라 앞 30 m 고정
        d = Detection("car", (60, 40, 80, 60), 30.0, 0.0, 2.0, 1.9, 0.9)
        tracks = tr.step([d], odo, 0.1)
    t = [x for x in tracks if x.confirmed][0]
    # 측정은 세계좌표로 변환되므로 차도 10 m/s 로 움직임 → vx ≈ 10
    assert abs(t.x[2] - 10) < 3.0
    pred = t.predict_traj(3.0)
    assert pred[-1][0] == 3.0
    assert abs(pred[-1][1][0] - (t.x[0] + 3 * t.x[2])) < 0.1


def test_planner_brakes_for_stationary_ahead():
    geo = Geometry(CameraMount(width=256, height_px=144))
    odo = Odometry()
    pl = SafetyPlanner(geo, odo)
    # 앞에 정지 차 (오도메트리 세계 앞 20 m)
    t = Track(1, "car", np.array([20.0, 0.0, 0.0, 0.0]), np.eye(4) * 0.5, hits=5)
    cls = np.zeros((144, 256), np.uint8)
    cls[80:, :] = LBL["road"]
    conf = np.ones((144, 256))
    a, path = pl.plan(v=15.0, steer=0.0, tracks=[t], lamps=[], cls_map=cls, conf=conf, dt=0.1)
    assert a > 2.0 and "예측 충돌" in pl.reason


def test_perception_does_not_read_gt_objects():
    """인식 경로에 env.ob_* / signal.state 를 쓰지 않음 (소스 검사)."""
    import inspect
    from autodrive.perception import detect, planner, track, seg
    for mod in (detect, planner, track, seg):
        src = inspect.getsource(mod)
        assert "ob_s" not in src and "ob_v" not in src and ".state(" not in src
