"""라이다 비슷한 광선 거리 센서: 차에서 부채꼴로 광선을 쏴서 트랙 벽까지 거리를 잰다."""
from __future__ import annotations

import numpy as np


class Lidar:
    def __init__(self, n_rays: int = 15, fov_deg: float = 180.0, max_range: float = 30.0,
                 noise_std: float = 0.0, rng: np.random.Generator | None = None):
        self.n_rays = n_rays
        self.angles = np.deg2rad(np.linspace(-fov_deg / 2, fov_deg / 2, n_rays))
        self.max_range = float(max_range)
        self.noise_std = float(noise_std)
        self.rng = rng or np.random.default_rng()

    def scan(self, pos: np.ndarray, yaw: float, walls: np.ndarray, circles: np.ndarray | None = None) -> np.ndarray:
        """walls: (S, 2, 2) 선분들, circles: (K, 3) 장애물 원 (x, y, 반지름).
        반환: (n_rays,) 거리 (m), 못 맞히면 max_range."""
        ang = yaw + self.angles
        d = np.stack([np.cos(ang), np.sin(ang)], axis=1)            # (R,2)
        a, b = walls[:, 0], walls[:, 1]                               # (S,2)
        e = b - a                                                     # (S,2)
        w = a - pos                                                   # (S,2)
        # pos + t d = a + u e  →  t, u 풀기 (2D 외적)
        cross = lambda p, q: p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]
        denom = cross(d[:, None, :], e[None, :, :])                   # (R,S)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = cross(w[None, :, :], e[None, :, :]) / denom
            u = cross(w[None, :, :], d[:, None, :]) / denom
        hit = (np.abs(denom) > 1e-9) & (t >= 0) & (u >= 0) & (u <= 1)
        t = np.where(hit, t, np.inf)
        r = np.minimum(t.min(axis=1), self.max_range)
        if circles is not None and len(circles):
            c = circles[:, :2] - pos                                  # (K,2)
            proj = d @ c.T                                            # (R,K)
            perp2 = np.sum(c ** 2, axis=1)[None, :] - proj ** 2
            rad2 = circles[:, 2][None, :] ** 2
            hitc = (perp2 <= rad2) & (proj > 0)
            tc = np.where(hitc, proj - np.sqrt(np.maximum(rad2 - perp2, 0)), np.inf)
            r = np.minimum(r, np.maximum(tc.min(axis=1), 0.0))
        if self.noise_std > 0:
            r = np.clip(r + self.rng.normal(0, self.noise_std, r.shape), 0, self.max_range)
        return r
