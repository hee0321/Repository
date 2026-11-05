"""추적(칼만, 등속 모델) + 2~3초 예측. 좌표는 '주행 기록(오도메트리) 세계': 내 차 속도·요레이트를 적분한 것.
내 차 속도·요레이트 센서는 실제 차에도 있는 것이라 사용해요(잡음 조금 넣음)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CHI2_GATE = 9.21          # 2자유도 99 %
ACC_SIGMA = {"car": 2.5, "ped": 1.2, "object": 0.3}
SIZE = {"car": (4.5, 1.9), "ped": (0.6, 0.6), "object": (1.0, 1.0)}


@dataclass
class Track:
    tid: int
    cls: str
    x: np.ndarray                 # [x, y, vx, vy] 오도메트리 세계
    P: np.ndarray
    hits: int = 1
    misses: int = 0
    age: int = 0
    occluded: bool = False
    hist: list = field(default_factory=list)

    @property
    def confirmed(self):
        return self.hits >= 3

    def predict_traj(self, horizon=3.0, dt=0.25):
        """반환: [(t, 위치(2), 공분산(2×2))] — 등속 가정."""
        out = []
        q = ACC_SIGMA.get(self.cls, 2.0) ** 2
        for k in range(1, int(horizon / dt) + 1):
            t = k * dt
            Fm = np.array([[1, 0, t, 0], [0, 1, 0, t], [0, 0, 1, 0], [0, 0, 0, 1]])
            Q = q * np.array([[t ** 4 / 4, 0, t ** 3 / 2, 0], [0, t ** 4 / 4, 0, t ** 3 / 2],
                              [t ** 3 / 2, 0, t ** 2, 0], [0, t ** 3 / 2, 0, t ** 2]])
            Pt = Fm @ self.P @ Fm.T + Q
            out.append((t, (Fm @ self.x)[:2], Pt[:2, :2]))
        return out


class Odometry:
    """내 차 자세를 속도·요레이트로 적분 (센서 잡음 포함)."""

    def __init__(self, rng=None):
        self.x = self.y = self.yaw = 0.0
        self.rng = rng or np.random.default_rng(0)

    def step(self, v, yaw_rate, dt):
        v = v * (1 + self.rng.normal(0, 0.005))
        yaw_rate = yaw_rate + self.rng.normal(0, np.radians(0.2))
        self.yaw += yaw_rate * dt
        self.x += v * np.cos(self.yaw) * dt
        self.y += v * np.sin(self.yaw) * dt

    def to_world(self, fwd, left):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([self.x + fwd * c - left * s, self.y + fwd * s + left * c])

    def to_ego(self, P):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        d = np.asarray(P) - (self.x, self.y)
        return np.stack([d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c], -1)

    def rot(self):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, -s], [s, c]])


class Tracker:
    def __init__(self, cam_x=2.45):
        self.tracks: list[Track] = []
        self.nid = 1
        self.cam_x = cam_x   # 카메라는 뒷바퀴 축 앞 2.45 m

    def _meas(self, det, odo):
        fwd = det.fwd + self.cam_x
        z = odo.to_world(fwd, det.left)
        rng_ = max(det.dist, 1.0)
        # 거리 방향 σ = det.sigma, 옆 방향 σ = 방위 0.4°·거리 + 0.15 m
        b = np.arctan2(det.left, fwd)
        Rl = np.diag([det.sigma ** 2, (np.radians(0.4) * rng_ + 0.15) ** 2])
        Rot = odo.rot() @ np.array([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]])
        return z, Rot @ Rl @ Rot.T

    def step(self, dets, odo, dt):
        Fm = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        for t in self.tracks:
            q = ACC_SIGMA.get(t.cls, 2.0) ** 2
            Q = q * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0], [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                              [dt ** 3 / 2, 0, dt ** 2, 0], [0, dt ** 3 / 2, 0, dt ** 2]])
            t.x = Fm @ t.x
            t.P = Fm @ t.P @ Fm.T + Q
            t.age += 1
        Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        meas = [self._meas(d, odo) for d in dets]
        pairs = []
        for i, (z, R) in enumerate(meas):
            for j, t in enumerate(self.tracks):
                if t.cls != dets[i].cls:
                    continue
                S = Hm @ t.P @ Hm.T + R
                y = z - Hm @ t.x
                d2 = float(y @ np.linalg.solve(S, y))
                if d2 < CHI2_GATE:
                    pairs.append((d2, i, j))
        used_d, used_t = set(), set()
        for d2, i, j in sorted(pairs):
            if i in used_d or j in used_t:
                continue
            used_d.add(i)
            used_t.add(j)
            t = self.tracks[j]
            z, R = meas[i]
            S = Hm @ t.P @ Hm.T + R
            K = t.P @ Hm.T @ np.linalg.inv(S)
            t.x = t.x + K @ (z - Hm @ t.x)
            t.P = (np.eye(4) - K @ Hm) @ t.P
            t.hits += 1
            t.misses = 0
            t.occluded = dets[i].occluded
            t.hist.append(t.x[:2].copy())
            t.hist = t.hist[-20:]
        for j, t in enumerate(self.tracks):
            if j not in used_t:
                t.misses += 1
        for i, (z, R) in enumerate(meas):
            if i in used_d:
                continue
            P = np.zeros((4, 4))
            P[:2, :2] = R
            P[2:, 2:] = np.eye(2) * (4.0 if dets[i].cls == "car" else 1.5) ** 2
            self.tracks.append(Track(self.nid, dets[i].cls, np.r_[z, 0.0, 0.0], P, hist=[z.copy()]))
            self.nid += 1
        # 놓친 물체는 1.5 s(15번) 동안 불확실성을 키우며 유지 (가려졌을 수 있음 → 보수적)
        self.tracks = [t for t in self.tracks if t.misses <= (15 if t.confirmed else 2)]
        return self.tracks
