"""내비 안내: '위성지도에 맞춘 도로망 위 경로' + 잡음 있는 GPS 위치 → 사람이 내비에서 얻는 정보만.

정책에 주는 것 (지도·내비가 아는 정적 정보 + 내 차 계기판):
  - 내 속도, 핸들 각, 직전 가속 명령, 마지막 정지 뒤 시간·거리(사람의 기억)
  - 경로 방향: 4·8·15·25·40 m 앞 경로 접선 방향 - 내 방향(GPS/IMU 잡음 포함)  ← '앞에서 오른쪽으로 꺾음'
  - 앞 굽이 곡률(15·30·60 m 안 최대), 남은 거리
  - 제한속도(지금, 앞 구간 미리 보기), 어린이보호구역(지금, 시작까지 거리)  ← 한국 내비 안내와 같음
  - 다음 신호 교차로까지 거리(지도), 그 교차로에서 우회전인지  ← 신호 '색'은 주지 않음(카메라로 봐야 함)
  - 도로 종류: 시골길·좁은 길·중앙선 양방향·비포장 앞·폭·앞지르기 허용(중앙선 점선), 밤 여부, 노면 마찰 추정
주지 않는 것: 차선 안 내 좌우 위치(카메라로), 다른 차·보행자 목록, 신호 색, 라이다.
"""
from __future__ import annotations

import numpy as np

AHEAD = (4.0, 8.0, 15.0, 25.0, 40.0)
NAV_DIM = 28


class GPS:
    """GPS+IMU 위치 잡음: 천천히 흐르는 치우침(σ≈0.8 m, 상관 5 s) + 흰 잡음 0.3 m, 방향 치우침 σ≈1.5°."""

    def __init__(self, rng, pos_sigma=0.8, white=0.3, yaw_sigma_deg=1.5, tau=5.0):
        self.rng = rng
        self.ps, self.wh, self.ys, self.tau = pos_sigma, white, np.radians(yaw_sigma_deg), tau
        self.b = rng.normal(0, pos_sigma, 2)
        self.by = rng.normal(0, self.ys)

    def read(self, x, y, yaw, dt):
        a = np.exp(-dt / self.tau)
        self.b = a * self.b + np.sqrt(1 - a * a) * self.rng.normal(0, self.ps, 2)
        self.by = a * self.by + np.sqrt(1 - a * a) * self.rng.normal(0, self.ys)
        n = self.rng.normal(0, self.wh, 2)
        return x + self.b[0] + n[0], y + self.b[1] + n[1], yaw + self.by


class Nav:
    def __init__(self, env, rng=None, noise=True):
        self.env = env
        self.rng = rng or np.random.default_rng(0)
        self.gps = GPS(self.rng) if noise else None
        self.seg = None
        self.t_stop = 0.0
        self.d_stop = 0.0
        self.last_t = None

    def features(self, prev_accel: float) -> np.ndarray:
        env = self.env
        c, p, tr, ru = env.car, env.params, env.track, env.rules
        t = env.t
        dt = 0.1 if self.last_t is None else max(t - self.last_t, 1e-3)
        self.last_t = t
        if self.gps is not None:
            x, y, yaw = self.gps.read(c.x, c.y, c.yaw, dt)
        else:
            x, y, yaw = c.x, c.y, c.yaw
        s, _, self.seg = tr.project(np.array([x, y]), hint=self.seg)   # 지도 맞춤(map matching)
        # 기억: 마지막으로 멈춘 뒤 시간·거리
        if c.v < 0.3:
            self.t_stop, self.d_stop = 0.0, 0.0
        else:
            self.t_stop += dt
            self.d_stop += c.v * dt
        f = np.zeros(NAV_DIM, np.float32)
        f[0] = c.v / p.max_speed
        f[1] = c.steer / p.max_steer
        f[2] = prev_accel
        for i, a in enumerate(AHEAD):
            f[3 + i] = np.clip(np.angle(np.exp(1j * (tr.heading_at(s + a) - yaw))) / 0.5, -3, 3)
        for i, h in enumerate((15.0, 30.0, 60.0)):
            idx, _ = tr.ahead(s, h)
            f[8 + i] = np.clip(np.max(np.abs(tr.curvature[idx])) * 5.0, 0, 2) if len(idx) else 0.0
        f[11] = ru.limit_at(s) / p.max_speed
        f[12] = env.limit_ahead(s) / p.max_speed
        f[13] = float(ru.school_at(s))
        dz = 1.0
        for k in range(0, 101, 5):
            if ru.school_at(s + k):
                dz = k / 100.0
                break
        f[14] = dz
        nx = ru.next_signal(s, 150.0)
        f[15] = 1.5 if nx is None else min(nx[0], 150.0) / 100.0
        f[16] = 0.0 if nx is None else float(nx[1].right_turn)
        f[17] = float(ru.rural)
        f[18] = float(tr.width < 6.0)
        f[19] = float(ru.twoway_at(s))
        f[20] = float(any(ru.gravel_at(s + k) for k in range(0, 41, 4)))
        f[21] = tr.width / 8.0
        f[22] = ru.light
        f[23] = env.params.mu
        f[24] = min(self.t_stop, 10.0) / 10.0
        f[25] = min(self.d_stop, 50.0) / 50.0
        f[26] = min(max(env.goal_m - env.progress, 0.0), 200.0) / 200.0
        f[27] = float(all(ru.passing_ok_at(s + k) for k in range(0, 121, 10)))
        return f
