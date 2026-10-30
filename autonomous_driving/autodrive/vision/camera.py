"""앞 카메라 1대 렌더러 (테슬라식 장착: 앞 유리 위쪽 가운데, 룸미러 뒤).

정책이 '보는' 것은 이 영상 하나뿐이다. 시뮬레이터 장면(위성영상 바닥 + 도로 칠·차선·정지선·횡단보도·
어린이보호구역 붉은 포장, 다른 차·보행자·물체, 신호등, OSM 건물)을 핀홀 카메라로 투영한다.

장착 위치 (CameraMount, 기본값):
  - 좌우: 차 중심선 (0 m)  — 운전석과 조수석 사이
  - 높이: 지면 위 1.35 m     — 앞 유리 위쪽(룸미러 뒤), 세단 기준 1.3~1.4 m
  - 앞뒤: 앞 범퍼에서 0.8 m 뒤 (시뮬레이터 차 길이 4.0 m, 뒷바퀴 축 기준 +2.45 m)
  - 아래로 3° 숙임, 수평 화각 55° (테슬라 메인 전방 카메라 ≈ 50~60°)
  - 해상도 128×72 (CPU 학습용 저해상도), 20 Hz
현실과 다른 점(정직하게): 렌즈 왜곡·롤링 셔터·눈부심 없음, 건물·차는 상자, 신호등 불빛은 최소 2 px 로
번짐(bloom)을 흉내 냄, 나무·언덕에 가려짐 없음.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class CameraMount:
    x_from_rear_axle: float = 2.45   # m (앞 범퍼 3.25 m 에서 0.8 m 뒤)
    lateral: float = 0.0             # m, 차 중심선
    height: float = 1.35             # m
    pitch_deg: float = 3.0           # 아래로
    hfov_deg: float = 55.0
    width: int = 128
    height_px: int = 72
    max_range: float = 140.0

    @property
    def f(self):
        return self.width / 2 / np.tan(np.radians(self.hfov_deg) / 2)


SKY_TOP = np.array([110, 150, 205], np.float32)
SKY_HOR = np.array([190, 205, 220], np.float32)
ASPHALT = np.array([70, 72, 76], np.float32)
GRAVEL = np.array([140, 122, 95], np.float32)
SCHOOL = np.array([165, 60, 55], np.float32)   # 어린이보호구역 붉은 포장
WHITE = np.array([235, 235, 235], np.float32)
YELLOW = np.array([235, 190, 40], np.float32)
LAMP = {"red": (255, 40, 30), "yellow": (255, 200, 0), "green": (40, 255, 120)}
CAR_COLORS = [(200, 200, 205), (30, 30, 35), (170, 30, 30), (40, 70, 150), (230, 230, 225), (90, 95, 100), (60, 110, 70)]


class FrontCamera:
    def __init__(self, env, ground=None, buildings=None, mount: CameraMount | None = None, rng=None):
        self.env = env
        self.ground = ground
        self.buildings = buildings or []
        self.m = mount or CameraMount()
        self.rng = rng or np.random.default_rng(0)
        m = self.m
        W, H, f = m.width, m.height_px, m.f
        self.cx, self.cy = W / 2 - 0.5, H / 2 - 0.5
        p = np.radians(m.pitch_deg)
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        xc = (u - self.cx) / f            # 오른쪽
        yc = (v - self.cy) / f            # 아래
        # 차 좌표(앞, 왼쪽, 위)에서 광선: 앞(cos p, 0, -sin p) + xc·오른쪽(0,-1,0) + yc·아래(-sin p, 0, -cos p)
        fwd = np.cos(p) - yc * np.sin(p)
        left = -xc
        up = -np.sin(p) - yc * np.cos(p)
        t = np.where(up < -1e-4, m.height / np.maximum(-up, 1e-6), np.inf)
        self.g_fwd, self.g_left = fwd * t, left * t
        dist = np.hypot(self.g_fwd, self.g_left)
        self.ground_mask = np.isfinite(t) & (dist < m.max_range)
        self.g_dist = np.where(self.ground_mask, dist, m.max_range)
        self.horizon_row = self.cy - f * np.tan(p)
        rows = np.arange(H)[:, None].repeat(W, 1).astype(np.float32)
        a = np.clip((self.horizon_row - rows) / max(self.horizon_row, 1), 0, 1)[..., None]
        self.sky = (SKY_HOR * (1 - a) + SKY_TOP * a).astype(np.float32)
        self._gi = np.nonzero(self.ground_mask.ravel())[0]
        self._noise_tex = None
        self.seed_colors = {}

    # --- 좌표 ---
    def pose(self):
        c = self.env.car
        m = self.m
        cy_, sy_ = np.cos(c.yaw), np.sin(c.yaw)
        px = c.x + m.x_from_rear_axle * cy_ - m.lateral * sy_
        py = c.y + m.x_from_rear_axle * sy_ + m.lateral * cy_
        return px, py, c.yaw

    def project(self, P):
        """세계 점들 (N,3: x, y, z) → (u, v, 깊이). 깊이 ≤ 0.3 은 카메라 뒤."""
        px, py, yaw = self.pose()
        dx, dy = P[:, 0] - px, P[:, 1] - py
        fw = dx * np.cos(yaw) + dy * np.sin(yaw)
        lf = -dx * np.sin(yaw) + dy * np.cos(yaw)
        up = P[:, 2] - self.m.height
        p = np.radians(self.m.pitch_deg)
        zc = fw * np.cos(p) - up * np.sin(p)      # 카메라 앞
        yc = -(fw * np.sin(p) + up * np.cos(p))   # 카메라 아래
        xc = -lf
        z = np.maximum(zc, 0.3)
        return self.cx + self.m.f * xc / z, self.cy + self.m.f * yc / z, zc

    # --- 바닥 ---
    def _ground_rgb(self, X, Y, light, dist=0.0):
        env = self.env
        tr = env.track
        n = len(X)
        rgb = None
        ok = np.zeros(n, bool)
        if self.ground is not None:
            rgb, ok = self.ground.sample(X, Y)
        if rgb is None or not ok.all():
            # 절차 생성 바닥: 풀·흙 얼룩 (위성영상 없는 합성 트랙·절차 시골길)
            q = (np.sin(X * 0.37) * np.cos(Y * 0.29) + np.sin(X * 0.11 + Y * 0.13)) * 0.5
            base = np.where(env.rules.rural, 1.0, 0.0)
            grass = np.stack([70 + 25 * q + 20 * base, 105 + 30 * q, 55 + 15 * q], 1).astype(np.float32)
            rgb = grass if rgb is None else np.where(ok[:, None], rgb, grass)
        # 경로 좌표 (s, d): 차 주변 경로 표본 창에서 가장 가까운 점
        s_car = env._s
        C = tr.center
        L = len(C)
        i0 = tr._idx(s_car)
        idx = np.arange(i0 - 15, i0 + int(self.m.max_range) + 10)
        idx = idx % L if tr.closed else idx[(idx >= 0) & (idx < tr.m)]
        Cw, Tw, Nw = tr.seg_start[idx], tr.tangent[idx], tr.normal[idx]
        dx = X[:, None] - Cw[None, :, 0]
        dy = Y[:, None] - Cw[None, :, 1]
        d2 = dx * dx + dy * dy
        j = np.argmin(d2, axis=1)
        ddx, ddy = dx[np.arange(n), j], dy[np.arange(n), j]
        along = ddx * Tw[j, 0] + ddy * Tw[j, 1]
        d = ddx * Nw[j, 0] + ddy * Nw[j, 1]
        s = tr.s[idx[j]] + along
        half = tr.width / 2
        on = np.abs(d) < half + 0.25
        rules = env.rules
        ii = np.clip((np.mod(s, tr.length) if tr.closed else s) / rules.ds, 0, len(rules.limit_mps) - 1).astype(int)
        school = rules.school[ii]
        gravel = rules.gravel[ii] if rules.gravel is not None else np.zeros(n, bool)
        twoway = rules.twoway[ii]
        road = np.where(school[:, None], SCHOOL, np.where(gravel[:, None], GRAVEL, ASPHALT))
        alpha = 0.88 if self.ground is None else 0.72
        out = rgb.copy()
        out[on] = rgb[on] * (1 - alpha) + road[on] * alpha
        # 차선: 가장자리 흰 실선, 양방향이면 가운데 노란 실선(시골 1차로는 중앙선 없음), 아니면 흰 점선
        edge = on & (np.abs(np.abs(d) - (half - 0.25)) < 0.15 + 0.004 * dist) & ~gravel
        out[edge] = WHITE
        narrow = tr.width < 6.0
        cen = on & (np.abs(d) < 0.12 + 0.004 * dist) & (not narrow) & ~gravel
        dash = (np.mod(s, 8.0) < 3.0)
        out[cen & twoway] = YELLOW
        out[cen & ~twoway & dash] = WHITE
        # 정지선(내 차로 쪽) + 횡단보도
        for sg in rules.signals:
            ds_ = s - sg.s
            if tr.closed:
                ds_ = (ds_ + tr.length / 2) % tr.length - tr.length / 2
            stop = on & (ds_ > 0) & (ds_ < 0.45) & (d < 0.1)
            out[stop] = WHITE
            zebra = on & (ds_ > 1.5) & (ds_ < 5.5) & (np.mod(d, 1.0) < 0.5)
            out[zebra] = out[zebra] * 0.3 + WHITE * 0.7
        return out, s, d

    def _shade(self, rgb, dist, light, bearing=None):
        if light >= 0.99:
            haze = np.clip((dist - 60) / 120, 0, 0.5)[:, None]
            return rgb * (1 - haze) + SKY_HOR * haze
        # 밤: 어둡게 + 전조등 (±22°, 45 m)
        hl = np.exp(-(dist / 45.0) ** 2)
        if bearing is not None:
            hl = hl * np.exp(-(bearing / np.radians(22)) ** 2)
        k = (light * 0.6 + 1.1 * hl)[:, None]
        return rgb * np.clip(k, 0.05, 1.2)

    def render(self, noise=0.0) -> np.ndarray:
        env = self.env
        m = self.m
        light = float(env.rules.light)
        px, py, yaw = self.pose()
        cyaw, syaw = np.cos(yaw), np.sin(yaw)
        F = self.g_fwd.ravel()[self._gi]
        Lf = self.g_left.ravel()[self._gi]
        X = px + F * cyaw - Lf * syaw
        Y = py + F * syaw + Lf * cyaw
        rgb, _, _ = self._ground_rgb(X, Y, light, self.g_dist.ravel()[self._gi])
        rgb = self._shade(rgb, self.g_dist.ravel()[self._gi], light, np.arctan2(Lf, F))
        sky = self.sky if light >= 0.99 else self.sky * 0.12
        img = sky.reshape(-1, 3).copy()
        img[self._gi] = rgb
        img = img.reshape(m.height_px, m.width, 3)
        im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        dr = ImageDraw.Draw(im)
        self._draw_objects(dr, light)
        out = np.asarray(im)
        if noise > 0:
            out = np.clip(out.astype(np.float32) + self.rng.normal(0, noise, out.shape), 0, 255).astype(np.uint8)
        return out

    # --- 물체 ---
    def _box(self, cx, cy, yaw, length, width, z0, z1):
        c, s = np.cos(yaw), np.sin(yaw)
        hl, hw = length / 2, width / 2
        xy = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
        P = np.c_[cx + xy[:, 0] * c - xy[:, 1] * s, cy + xy[:, 0] * s + xy[:, 1] * c]
        return np.r_[np.c_[P, np.full(4, z0)], np.c_[P, np.full(4, z1)]]

    def _poly(self, dr, P3, color, min_px=0):
        u, v, z = self.project(P3)
        if np.all(z < 0.3):
            return False
        if np.any(z < 0.3):
            return False   # 카메라를 가로지르는 면은 생략 (아주 가까운 것만 해당)
        pts = list(zip(u.tolist(), v.tolist()))
        if min_px and (u.max() - u.min() < min_px or v.max() - v.min() < min_px):
            cu, cv = u.mean(), v.mean()
            r = min_px / 2
            dr.ellipse((cu - r, cv - r, cu + r, cv + r), fill=color)
            return True
        dr.polygon(pts, fill=color)
        return True

    def _draw_box(self, dr, B, color, light, depth, cam_xy):
        """상자 그리기: 카메라 쪽을 보는 옆면 + 윗면 (화가 알고리즘)."""
        col = np.array(color, np.float32)
        k = 1.0 if light >= 0.99 else float(np.clip(light * 0.6 + 1.1 * np.exp(-(depth / 45.0) ** 2), 0.06, 1.0))
        bot, top = B[:4], B[4:]
        ctr = bot.mean(axis=0)[:2]
        faces = []
        for i in range(4):
            a, b = bot[i], bot[(i + 1) % 4]
            mid = (a[:2] + b[:2]) / 2
            nrm = mid - ctr
            if np.dot(nrm, np.asarray(cam_xy) - mid) <= 0:
                continue
            faces.append((np.linalg.norm(np.asarray(cam_xy) - mid), np.array([a, b, top[(i + 1) % 4], top[i]]), 0.75 + 0.25 * (i % 2)))
        for _, Q, sh in sorted(faces, key=lambda t: -t[0]):
            self._poly(dr, Q, tuple(int(x) for x in np.clip(col * sh * k, 0, 255)))
        if top[0, 2] < self.m.height:
            self._poly(dr, top, tuple(int(x) for x in np.clip(col * 1.05 * k, 0, 255)))

    def _draw_objects(self, dr, light):
        env = self.env
        tr = env.track
        px, py, yaw = self.pose()
        items = []   # (깊이, 그리기 함수)
        fwd = np.array([np.cos(yaw), np.sin(yaw)])
        # 건물 (OSM, 높이 태그 또는 층수×3.2 m): 120 m 안, 카메라 앞쪽
        for poly, h in self.buildings:
            c = poly.mean(axis=0)
            rel = c - (px, py)
            dd = float(np.hypot(*rel))
            if dd > 120 or np.dot(rel, fwd) < -20:
                continue
            items.append((dd + 5, ("bld", poly, h)))
        for k in range(len(env.ob_s)):
            g = env._gap(env.ob_s[k])
            if not (-12 < g < self.m.max_range):
                continue
            s_k = env.ob_s[k]
            p = tr.point_at(s_k) + tr.normal_at(s_k) * env.ob_d[k]
            rel = p - (px, py)
            dep = float(np.dot(rel, fwd))
            if dep < 0.5:
                continue
            items.append((float(np.hypot(*rel)), ("ob", k, p, tr.heading_at(s_k))))
        for i, sg in enumerate(env.rules.signals):
            for off in (22.0,):
                s_l = sg.s + off
                g = (s_l - env._s) % tr.length if tr.closed else s_l - env._s
                if not (2 < g < self.m.max_range):
                    continue
                lane = env.rules.lane_ref_at(sg.s)
                p = tr.point_at(s_l) + tr.normal_at(s_l) * lane
                items.append((float(np.hypot(*(p - (px, py)))), ("sig", sg, p, tr.heading_at(s_l), lane)))
        for dep, it in sorted(items, key=lambda t: -t[0]):
            if it[0] == "bld":
                self._draw_building(dr, it[1], it[2], light, dep, (px, py))
            elif it[0] == "ob":
                self._draw_obstacle(dr, it[1], it[2], it[3], light, dep, (px, py))
            else:
                self._draw_signal(dr, it[1], it[2], it[3], it[4], light, dep)

    def _draw_building(self, dr, poly, h, light, dep, cam):
        col = np.array([150, 145, 140], np.float32)
        k = 1.0 if light >= 0.99 else 0.08
        walls = []
        n = len(poly)
        ctr = poly.mean(axis=0)
        for i in range(n - 1 if np.allclose(poly[0], poly[-1]) else n):
            a, b = poly[i], poly[(i + 1) % n]
            mid = (a + b) / 2
            e = b - a
            nrm = np.array([e[1], -e[0]])
            if np.dot(nrm, mid - ctr) < 0:
                nrm = -nrm
            if np.dot(nrm, np.asarray(cam) - mid) <= 0:
                continue
            walls.append((np.linalg.norm(np.asarray(cam) - mid), a, b, 0.7 + 0.3 * abs(nrm[0]) / (np.linalg.norm(nrm) + 1e-9)))
        for _, a, b, sh in sorted(walls, key=lambda t: -t[0]):
            Q = np.array([[a[0], a[1], 0], [b[0], b[1], 0], [b[0], b[1], h], [a[0], a[1], h]])
            self._poly(dr, Q, tuple(int(x) for x in col * sh * k))

    def _draw_obstacle(self, dr, k, p, hd, light, dep, cam):
        env = self.env
        kind = env.ob_kind[k]
        v = env.ob_v[k]
        if kind == "ped":
            B = self._box(p[0], p[1], hd, 0.45, 0.5, 0.0, 1.7)
            self._draw_box(dr, B, (230, 120, 40) if k % 2 else (60, 90, 200), light, dep, cam)
            head = self._box(p[0], p[1], hd, 0.3, 0.3, 1.45, 1.75)
            self._draw_box(dr, head, (225, 190, 160), light, dep, cam)
            return
        if kind == "object" or abs(v) < 1e-6 and kind not in ("car", "entering"):
            r = env.ob_r[k]
            B = self._box(p[0], p[1], hd, 2 * r, 2 * r, 0.0, 1.0)
            self._draw_box(dr, B, (240, 120, 20), light, dep, cam)
            return
        L, Wd = env.ob_l[k], min(2 * env.ob_r[k], 1.9)
        yaw = hd + (np.pi if v < -0.5 else 0.0)
        col = CAR_COLORS[k % len(CAR_COLORS)]
        body = self._box(p[0], p[1], yaw, L, Wd, 0.3, 1.0)
        self._draw_box(dr, body, col, light, dep, cam)
        cab = self._box(p[0] - 0.2 * L * np.cos(yaw) * 0.3, p[1] - 0.2 * L * np.sin(yaw) * 0.3, yaw, L * 0.55, Wd * 0.9, 1.0, 1.5)
        self._draw_box(dr, cab, tuple(int(c * 0.6) for c in col), light, dep, cam)
        # 등: 같은 방향 차 = 뒤 빨간 등(밤엔 밝게), 마주 오는 차 = 전조등
        c, s = np.cos(yaw), np.sin(yaw)
        for side in (-1, 1):
            if v < -0.5:   # 마주 옴: 앞이 나를 향함
                q = np.array([[p[0] + L / 2 * c - side * 0.6 * s, p[1] + L / 2 * s + side * 0.6 * c, 0.7]])
                colr = (255, 255, 220)
            else:
                q = np.array([[p[0] - L / 2 * c - side * 0.6 * s, p[1] - L / 2 * s + side * 0.6 * c, 0.8]])
                colr = (255, 30, 30) if (light < 0.99 or env.ob_v[k] < 1.0) else (170, 20, 20)
            u, vv, z = self.project(q)
            if z[0] > 0.5:
                r = max(0.6, 0.18 * self.m.f / z[0]) * (2.0 if light < 0.99 else 1.0)
                dr.ellipse((u[0] - r, vv[0] - r, u[0] + r, vv[0] + r), fill=colr)

    def _draw_signal(self, dr, sg, p, hd, lane, light, dep):
        env = self.env
        tr = env.track
        nrm = np.array([-np.sin(hd), np.cos(hd)])
        side = p + nrm * (-(tr.width / 2 + 0.6) - lane)
        # 기둥 + 가로 팔
        pole = self._box(side[0], side[1], hd, 0.25, 0.25, 0.0, 5.6)
        self._draw_box(dr, pole, (90, 90, 95), light, dep, (0, 0))
        arm = np.array([[side[0], side[1], 5.45], [p[0], p[1], 5.45], [p[0], p[1], 5.6], [side[0], side[1], 5.6]])
        self._poly(dr, arm, (80, 80, 85))
        # 가로형 신호등 (한국식 3구: 빨강·노랑·초록), 0.5 m × 1.5 m
        t = np.array([np.cos(hd), np.sin(hd)])
        hous = []
        for a, b in ((-0.75, 0.75),):
            pa, pb = p + nrm * a, p + nrm * b
            hous = np.array([[pa[0], pa[1], 5.0], [pb[0], pb[1], 5.0], [pb[0], pb[1], 5.5], [pa[0], pa[1], 5.5]])
        self._poly(dr, hous - np.r_[t * 0.05, 0], (20, 20, 20))
        st = sg.state(env.t)
        pos = {"red": 0.5, "yellow": 0.0, "green": -0.5}[st]   # 왼쪽(+)=빨강 … 오른쪽=초록 (운전자 기준 왼→오: 빨·노·초)
        lp = p + nrm * pos
        u, v, z = self.project(np.array([[lp[0], lp[1], 5.25]]))
        if z[0] > 0.5:
            r = max(1.0, 0.25 * self.m.f / z[0])   # 번짐: 최소 지름 2 px
            dr.ellipse((u[0] - r, v[0] - r, u[0] + r, v[0] + r), fill=LAMP[st])


def load_buildings(data_key: str, track_paths=None, clear_m=4.0):
    """OSM 건물 외곽(높이) 캐시가 있으면 [(poly Nx2, h)] — 경로 가운데서 clear_m 안에 걸치는 건물은 뺌
    (OSM·위성 정렬 오차로 길 위에 건물이 서는 것 방지). 없으면 []."""
    import json
    import os
    from .ground import CACHE
    p = os.path.join(CACHE, f"buildings_{data_key}.json")
    if not os.path.exists(p):
        return []
    raw = json.load(open(p))
    out = []
    allp = np.concatenate(track_paths) if track_paths else None
    for poly, h in raw:
        P = np.asarray(poly, float)
        if allp is not None:
            dmin = np.min(np.linalg.norm(allp[None, :, :] - P[:, None, :], axis=2)) if len(P) * len(allp) < 4e6 else 1e9
            if dmin < clear_m + 3.5:
                continue
        out.append((P, float(h)))
    return out


def data_key_for(env):
    """환경 이름(korea 시나리오) → 지도 데이터 이름, 아니면 None."""
    try:
        from korea_roads.scenarios import SCENARIOS
    except ImportError:  # pragma: no cover
        return None
    sc = SCENARIOS.get(env.name)
    return None if sc is None else sc.get("data", env.name)


def camera_for(env, offline=True, buildings=True, rng=None, mount=None):
    """env 에 맞는 카메라: 실제 한국 도로면 위성 바닥 + OSM 건물(캐시 있을 때), 아니면 절차 생성 바닥."""
    from .ground import ground_for
    key = data_key_for(env)
    g = ground_for(key, offline=offline) if key else None
    b = _BLD.get(key)
    if key and buildings and b is None:
        from korea_roads.scenarios import SCENARIOS
        from ..scenarios import korea_route
        paths = [korea_route(k)[0].center for k, v in SCENARIOS.items() if v.get("data", k) == key]
        b = _BLD[key] = load_buildings(key, paths)
    return FrontCamera(env, ground=g, buildings=b if buildings else None, rng=rng, mount=mount)


_BLD = {}
