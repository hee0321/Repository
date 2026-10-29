"""트랙/도로 경로: 중앙선 + 폭. 닫힌 트랙(서킷)과 열린 경로(실제 도로 A→B) 둘 다 지원."""
from __future__ import annotations

import numpy as np


class Track:
    """중앙선 점들로 정의되는 트랙. closed=False면 시작점→끝점 열린 경로."""

    def __init__(self, centerline: np.ndarray, width: float = 6.0, closed: bool = True, name: str = "track"):
        c = np.asarray(centerline, dtype=float)
        if closed and np.allclose(c[0], c[-1]):
            c = c[:-1]
        self.closed = closed
        self.name = name
        self.center = c
        self.width = float(width)
        nxt = np.roll(c, -1, axis=0) if closed else c[1:]
        base = c if closed else c[:-1]
        seg = nxt - base
        self.m = len(base)                     # 선분 개수
        self.seg_start = base
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.tangent = seg / self.seg_len[:, None]
        self.normal = np.stack([-self.tangent[:, 1], self.tangent[:, 0]], axis=1)  # 진행 방향 왼쪽(+)
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)[:-1]])
        self.length = float(self.seg_len.sum())
        heading = np.arctan2(self.tangent[:, 1], self.tangent[:, 0])
        nxt_h = np.roll(heading, -1) if closed else np.r_[heading[1:], heading[-1]]
        dh = np.angle(np.exp(1j * (nxt_h - heading)))
        self.curvature = dh / self.seg_len
        # 점마다의 법선(경계선 그리기/라이다 벽용)
        if closed:
            pn = self.normal + np.roll(self.normal, 1, axis=0)
        else:
            pn = np.vstack([self.normal[:1], self.normal[1:] + self.normal[:-1], self.normal[-1:]])
        pn /= np.linalg.norm(pn, axis=1, keepdims=True)
        self.left = c + pn * self.width / 2
        self.right = c - pn * self.width / 2

        def segs(b):
            return np.stack([b, np.roll(b, -1, axis=0)], axis=1) if closed else np.stack([b[:-1], b[1:]], axis=1)
        self.walls = np.concatenate([segs(self.left), segs(self.right)])

    def wrap(self, s):
        return s % self.length if self.closed else float(np.clip(s, 0.0, self.length))

    def _idx(self, s):
        return int(np.clip(np.searchsorted(self.s, self.wrap(s), side="right") - 1, 0, self.m - 1))

    def project(self, p: np.ndarray, hint: int | None = None, window: int = 12):
        """점 p를 중앙선에 투영 → (호 길이 s, 좌우 오프셋 d(+왼쪽), 선분 번호).
        hint(이전 선분 번호)를 주면 그 주변만 찾아서 빠르고, 길이 겹치는 곳에서도 헷갈리지 않는다."""
        p = np.asarray(p, dtype=float)
        if hint is None:
            idx = np.arange(self.m)
        else:
            idx = np.arange(hint - window // 2, hint + window)
            idx = idx % self.m if self.closed else np.unique(np.clip(idx, 0, self.m - 1))
        rel = p - self.seg_start[idx]
        t = np.clip(np.einsum("ij,ij->i", rel, self.tangent[idx]), 0.0, self.seg_len[idx])
        foot = self.seg_start[idx] + self.tangent[idx] * t[:, None]
        k = int(np.argmin(np.sum((p - foot) ** 2, axis=1)))
        i = int(idx[k])
        d = float(np.dot(p - foot[k], self.normal[i]))
        s = self.s[i] + t[k]
        return (s % self.length if self.closed else s), d, i

    def point_at(self, s: float) -> np.ndarray:
        s = self.wrap(s)
        i = self._idx(s)
        return self.seg_start[i] + self.tangent[i] * (s - self.s[i])

    def heading_at(self, s: float) -> float:
        i = self._idx(s)
        return float(np.arctan2(self.tangent[i, 1], self.tangent[i, 0]))

    def normal_at(self, s: float) -> np.ndarray:
        return self.normal[self._idx(s)]

    def ahead(self, s: float, horizon: float):
        """s부터 horizon(m) 앞까지 선분 번호와 거리."""
        ds = (self.s - s) % self.length if self.closed else self.s - s
        mask = (ds >= -1.0) & (ds <= horizon)
        return np.nonzero(mask)[0], ds[mask]

    def curvature_ahead(self, s: float, horizon: float) -> float:
        idx, _ = self.ahead(s, horizon)
        return float(np.max(np.abs(self.curvature[idx]))) if len(idx) else 0.0


def ring_track(radii_fn, a=45.0, b=28.0, n=400, width=6.0, name="track") -> Track:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = radii_fn(th)
    return Track(np.stack([a * r * np.cos(th), b * r * np.sin(th)], axis=1), width=width, name=name)


def make_default_track(n: int = 400, width: float = 6.0) -> Track:
    """S자 굽이가 섞인 기본 트랙(약 243 m). 반시계 방향."""
    return ring_track(lambda th: 1.0 + 0.18 * np.sin(2 * th) + 0.08 * np.cos(3 * th), n=n, width=width, name="default")


def make_oval_track(width: float = 8.0) -> Track:
    return ring_track(lambda th: np.ones_like(th), width=width, name="oval_easy")


def make_hard_track(width: float = 5.0) -> Track:
    """굽이가 더 많고 좁은 트랙."""
    return ring_track(lambda th: 1.0 + 0.2 * np.sin(3 * th) + 0.1 * np.cos(5 * th + 0.4),
                      a=50.0, b=35.0, n=600, width=width, name="hard")


def make_random_track(seed: int, width: float | None = None, difficulty: float = 1.0) -> Track:
    """무작위 트랙(학습용). difficulty가 클수록 굽이가 세다. 꼬이지 않는 트랙이 나올 때까지 다시 뽑는다."""
    rng = np.random.default_rng(seed)
    for _ in range(50):
        w = width or float(rng.uniform(5.0, 8.0))
        k = rng.integers(2, 6, size=3)
        amp = rng.uniform(0.0, 0.12 * difficulty, size=3)
        ph = rng.uniform(0, 2 * np.pi, size=3)
        f = lambda th: 1.0 + sum(A * np.sin(K * th + P) for A, K, P in zip(amp, k, ph))
        tr = ring_track(f, a=float(rng.uniform(35, 60)), b=float(rng.uniform(25, 40)), n=500, width=w,
                        name=f"random_{seed}")
        if 1 / np.abs(tr.curvature).max() > w / 2 + 1.5:
            return tr
    return make_default_track()
