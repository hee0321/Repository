"""분할 결과 → 물체 상자·거리(불확실성)·신호 상태·차선/도로 경계 (단안 카메라, 평평한 바닥 가정).

거리 추정 가정(합성 장면 시제품):
- 물체 아래 끝(바닥 닿는 점)이 평평한 도로 위에 있다고 보고, 카메라 높이 1.35 m·기울기 3°로 광선을 바닥에 내려 거리 계산.
  σ = 기울기 ±0.5°(제동·노면 흔들림) + 아래 끝 ±1 px 로 바꿔 본 차이. 멀수록 σ 가 거리² 에 비례해 커져요.
- 아래 끝이 영상 아래 끝에 닿거나 다른 물체에 가려지면 거리는 "이보다 가까울 수 있음"으로 표시(가려짐 = 보수적으로).
- 신호등 거리: 등 높이 5.25 m 라는 이 시뮬레이터의 신호등 모양을 가정(장면 전용 가정). 정지선 = 신호등 22 m 앞(장면 전용).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..vision.camera import LBL, CameraMount

OBJ_CLASSES = {LBL["car"]: "car", LBL["ped"]: "ped", LBL["object"]: "object"}
LAMP_CLASSES = {LBL["lamp_red"]: "red", LBL["lamp_yellow"]: "yellow", LBL["lamp_green"]: "green"}
LAMP_HEIGHT = 5.25          # m (이 시뮬레이터 신호등)
STOPLINE_BEFORE_SIGNAL = 22.0
PITCH_SIGMA_DEG = 0.5
# 이 렌더러의 물체 아래 끝 높이(장면 전용 가정): 차는 바퀴를 안 그리고 몸통이 0.3 m 에서 시작, 사람·물체는 바닥.
BOTTOM_H = {"car": 0.3, "ped": 0.0, "object": 0.0}
# 보이는 면 → 물체 가운데까지 더할 거리 (차 길이 4.5 m 의 절반 등)
HALF_DEPTH = {"car": 2.25, "ped": 0.25, "object": 0.5}


@dataclass
class Detection:
    cls: str
    box: tuple            # (u0, v0, u1, v1) 픽셀
    fwd: float            # 카메라 기준 앞 (m)
    left: float           # 왼쪽 (m)
    sigma: float          # 거리 σ (m)
    width_m: float
    conf: float
    occluded: bool = False
    n_px: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def dist(self):
        return float(np.hypot(self.fwd, self.left))


def components(mask, min_px=3):
    """4-이웃 연결 성분 (scipy 없이, 줄 단위 합치기). 반환: [(rows, cols)]."""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    parent = [0]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    nxt = 1
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        up = lab[y - 1, x] if y > 0 else 0
        lf = lab[y, x - 1] if x > 0 else 0
        if up and lf:
            a, b = find(up), find(lf)
            lab[y, x] = min(a, b)
            if a != b:
                parent[max(a, b)] = min(a, b)
        elif up or lf:
            lab[y, x] = up or lf
        else:
            lab[y, x] = nxt
            parent.append(nxt)
            nxt += 1
    if nxt == 1:
        return []
    roots = np.array([find(i) for i in range(nxt)])
    lab = roots[lab]
    out = []
    for r in np.unique(lab[ys, xs]):
        yy, xx = np.nonzero(lab == r)
        if len(yy) >= min_px:
            out.append((yy, xx))
    return out


class Geometry:
    """픽셀 ↔ 바닥/높이 기하 (카메라 내부·장착값만 사용)."""

    def __init__(self, mount: CameraMount):
        self.m = mount
        self.f = mount.f
        self.cx, self.cy = mount.width / 2 - 0.5, mount.height_px / 2 - 0.5

    def ground(self, u, v, pitch_deg=None, h=0.0):
        """픽셀 (u, v) 가 높이 h(기본 바닥)의 점이면 (앞, 왼쪽) m, 아니면 None."""
        p = np.radians(self.m.pitch_deg if pitch_deg is None else pitch_deg)
        xc = (u - self.cx) / self.f
        yc = (v - self.cy) / self.f
        fwd = np.cos(p) - yc * np.sin(p)
        up = -np.sin(p) - yc * np.cos(p)
        if up > -1e-4:
            return None
        t = (self.m.height - h) / -up
        return fwd * t, -xc * t

    def ground_sigma(self, u, v, h=0.0):
        base = self.ground(u, v, h=h)
        if base is None:
            return None, None, 99.0
        d0 = np.hypot(*base)
        ds = []
        for dp, dv in ((PITCH_SIGMA_DEG, 0), (-PITCH_SIGMA_DEG, 0), (0, 1.0), (0, -1.0)):
            g = self.ground(u, v + dv, self.m.pitch_deg + dp, h=h)
            ds.append(abs((np.hypot(*g) if g is not None else 3 * d0) - d0))
        return base[0], base[1], float(np.sqrt(np.mean(np.square(ds))) + 0.05 * d0 * 0 + 0.1)

    def at_height(self, u, v, h):
        """높이 h 의 점이 픽셀 (u, v) 에 보이면 (앞, 왼쪽)."""
        p = np.radians(self.m.pitch_deg)
        xc = (u - self.cx) / self.f
        yc = (v - self.cy) / self.f
        fwd = np.cos(p) - yc * np.sin(p)
        up = -np.sin(p) - yc * np.cos(p)
        if up <= 1e-4:
            return None
        t = (h - self.m.height) / up
        return fwd * t, -xc * t


def detect(cls_map, conf, geo: Geometry):
    """분할 지도 → Detection 목록 (물체) + 신호 목록."""
    H, W = cls_map.shape
    objs, lamps = [], []
    obj_any = np.isin(cls_map, list(OBJ_CLASSES))
    for c, name in OBJ_CLASSES.items():
        for yy, xx in components(cls_map == c, min_px=4 if name != "ped" else 3):
            u0, u1, v0, v1 = xx.min(), xx.max(), yy.min(), yy.max()
            # 바닥 닿는 점: 아래쪽 줄들의 가운데
            bottom = yy >= v1 - 1
            ub = float(xx[bottom].mean())
            fwd, left, sig = geo.ground_sigma(ub, v1 + 0.5, h=BOTTOM_H[name])
            if fwd is None:
                continue
            k = 1 + HALF_DEPTH[name] / max(np.hypot(fwd, left), 1.0)   # 보이는 면 → 가운데
            fwd, left = fwd * k, left * k
            below = cls_map[min(v1 + 1, H - 1), u0:u1 + 1]
            occl = bool(v1 >= H - 1 or np.mean(np.isin(below, list(OBJ_CLASSES))) > 0.3)
            if occl:
                sig = sig * 2 + 1.0
            wpx = u1 - u0 + 1
            width_m = wpx * max(fwd, 1.0) / geo.f
            objs.append(Detection(name, (int(u0), int(v0), int(u1), int(v1)), float(fwd), float(left), float(sig),
                                  float(width_m), float(conf[yy, xx].mean()), occl, int(len(yy))))
    for c, st in LAMP_CLASSES.items():
        for yy, xx in components(cls_map == c, min_px=1):
            u, v = float(xx.mean()), float(yy.mean())
            g = geo.at_height(u, v, LAMP_HEIGHT)
            if g is None:
                continue
            lamps.append(Detection("signal", (int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())),
                                   float(g[0]), float(g[1]), float(0.15 * g[0] + 1.0), 0.5, float(conf[yy, xx].mean()),
                                   n_px=int(len(yy)), extra={"state": st}))
    return objs, lamps


def lane_points(cls_map, geo: Geometry, max_range=40.0, step=2):
    """차선 표시·도로 경계를 바닥 좌표(앞, 왼쪽)로. 반환: (차선 점들, 도로 왼/오 경계 점들)."""
    H, W = cls_map.shape
    marks, lb, rb = [], [], []
    vh = int(np.ceil(geo.cy - geo.f * np.tan(np.radians(geo.m.pitch_deg)))) + 2
    for v in range(H - 1, max(vh, 0), -step):
        row = cls_map[v]
        road = (row == LBL["road"]) | (row == LBL["marking"])
        g = geo.ground(W / 2, v)
        if g is None or g[0] > max_range:
            break
        for u in np.nonzero(row == LBL["marking"])[0][::2]:
            q = geo.ground(u, v)
            if q:
                marks.append(q)
        idx = np.nonzero(road)[0]
        if len(idx):
            c = int(W / 2)
            # 가운데에서 이어진 도로 구간의 양 끝
            if road[c]:
                l = c
                while l > 0 and road[l - 1]:
                    l -= 1
                r = c
                while r < W - 1 and road[r + 1]:
                    r += 1
                if l > 0:
                    lb.append(geo.ground(l, v))
                if r < W - 1:
                    rb.append(geo.ground(r, v))
    return np.array(marks).reshape(-1, 2), np.array(lb).reshape(-1, 2), np.array(rb).reshape(-1, 2)
