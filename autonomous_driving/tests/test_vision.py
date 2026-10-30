"""앞 카메라·내비·비전 정책 기본 확인 (위성 캐시 없이도 돎: 절차 생성 바닥)."""
import numpy as np
import pytest


def _env(spec):
    from autodrive.scenarios import make_env
    env = make_env(spec, seed=0)
    env.reset(seed=0)
    return env


def test_camera_mount_tesla_like():
    from autodrive.vision.camera import CameraMount
    m = CameraMount()
    assert m.lateral == 0.0                       # 차 중심선 (운전석·조수석 사이)
    assert 1.3 <= m.height <= 1.4
    assert 50 <= m.hfov_deg <= 60
    front_bumper = 2.5 + 0.75                     # 뒷바퀴 축 + 축간거리 + 앞 오버행
    assert 0.5 <= front_bumper - m.x_from_rear_axle <= 0.8
    assert 0 < m.pitch_deg < 10


def test_camera_render_road_ahead():
    from autodrive.vision.camera import FrontCamera
    env = _env(dict(kind="oval_easy", laps=1))
    cam = FrontCamera(env)
    img = cam.render()
    assert img.shape == (72, 128, 3) and img.dtype == np.uint8
    # 아래 가운데 = 내 앞 도로(아스팔트 회색), 위 = 하늘(파랑 성분 큼)
    road = img[60:, 54:74].reshape(-1, 3).mean(0)
    sky = img[:5].reshape(-1, 3).mean(0)
    assert abs(road[0] - road[2]) < 25 and road.mean() < 120
    assert sky[2] > sky[0] + 30
    assert np.array_equal(img, cam.render())      # 잡음 없으면 결정적


def _straight_env(obstacles=None, signals=0):
    from autodrive.env import AutoDriveEnv
    from autodrive.rules import uniform_rules
    from autodrive.track import Track
    P = np.c_[np.arange(0, 400.0), np.zeros(400)]
    tr = Track(P, width=6.4, closed=False, name="straight")
    ru = uniform_rules(tr.length, 50, signals, closed=False, seed=0, lane_ref=-1.6, twoway=True)
    env = AutoDriveEnv(track=tr, rules=ru, obstacles=obstacles, seed=0)
    env.reset(seed=0)
    return env


def test_camera_sees_obstacle_and_red_light():
    from autodrive.vision.camera import FrontCamera
    env = _straight_env(obstacles=[dict(s=20, d=-1.6, v=0.0, r=1.0)])
    img = FrontCamera(env).render()
    # 주황 물체(장애물)가 가운데쯤 보임
    orange = (img[..., 0] > 180) & (img[..., 1] > 70) & (img[..., 1] < 160) & (img[..., 2] < 80)
    assert orange[:, 40:88].sum() > 8
    env2 = _straight_env(signals=1)
    sg = env2.rules.signals[0]
    t_red = next(t for t in np.arange(0, 120, 0.5) if sg.state(t) == "red")
    env2.t = float(t_red)
    env2.car.x = sg.s - 25
    env2._s = sg.s - 25
    im2 = FrontCamera(env2).render()
    red = (im2[..., 0] > 200) & (im2[..., 1] < 90) & (im2[..., 2] < 90)
    assert red[:30].sum() >= 2


def test_nav_has_no_object_info():
    """내비 입력은 지도·계기판 정보뿐: 장애물을 빼도 값이 같아야 함."""
    from autodrive.vision.nav import NAV_DIM, Nav
    a = _env(dict(kind="default", laps=1, width=7.0, obstacles=[dict(s=15, d=0.0, v=0.0, r=1.0)]))
    b = _env(dict(kind="default", laps=1, width=7.0))
    fa = Nav(a, rng=np.random.default_rng(1)).features(0.0)
    fb = Nav(b, rng=np.random.default_rng(1)).features(0.0)
    assert fa.shape == (NAV_DIM,)
    assert np.allclose(fa, fb)


def test_policy_forward():
    torch = pytest.importorskip("torch")
    from autodrive.vision.policy import VisionPolicy, stack_frames
    m = VisionPolicy().eval()
    x = stack_frames(np.zeros((72, 128, 3), np.uint8), None)
    out = m(torch.from_numpy(x[None]), torch.zeros(1, 28))
    assert out.shape == (1, 2) and float(out.abs().max()) <= 1.0
