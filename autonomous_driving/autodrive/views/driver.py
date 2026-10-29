"""운전석 1인칭(운전자 눈 높이) 시점 렌더러. numpy + Pillow만 쓴다.

카메라: 뒤축 기준 앞 1.3 m, 왼쪽 0.37 m(한국 차는 운전석이 왼쪽), 높이 1.25 m, 수평 시야각 90°, 아래로 1.5° 숙임.
그리는 것: 하늘·지평선 산(낮/밤, 밤에는 별), 노면(아스팔트/자갈/어린이보호구역 붉은 포장), 차선(가장자리 흰 선,
양방향 도로의 노란 중앙선 — 좁은 시골 1차로는 중앙선 없음), 길가(논밭·농수로·인도), 나무·전봇대·건물,
신호등·정지선·제한속도/어린이보호구역 표지, 다른 차(3D 상자, 미등/전조등)·보행자·길가 물건(가까운 것이 앞을 가림),
밤 전조등 빛(가로등 없는 길은 전조등 범위 밖이 캄캄), 보닛·A필러, 계기판(속도계·제한속도·핸들 각도), 모드 배너,
위에서 본 미니맵.

좌표: 트랙 법선(+)이 진행 방향 왼쪽. 한국은 우측통행이라 내 차는 오른쪽(d<0)으로 달리고,
화면에서는 도로 중앙(노란 선)이 내 왼쪽에 보여야 한다.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .common import KIND_AS, MODE_LABEL, MODE_RGB, font, mix, scale_rgb, text_center

NEAR = 0.35
SKY_DAY = ((120, 170, 225), (205, 225, 240))
SKY_NIGHT = ((3, 5, 14), (16, 22, 38))
CAR_COLORS = [(235, 235, 235), (170, 175, 182), (35, 35, 40), (40, 60, 110), (160, 30, 35), (90, 95, 80)]


class RoadGeom:
    """중앙선 꼭짓점 + 꼭짓점 법선을 호 길이 s로 보간 (꺾인 곳에서도 노면이 갈라지지 않게)."""

    def __init__(self, track):
        self.tr = track
        c = track.center
        pn = (track.left - track.center) / (track.width / 2)
        vs = np.r_[track.s, track.length] if not track.closed else np.r_[track.s, track.length]
        if track.closed:
            c = np.vstack([c, c[:1]])
            pn = np.vstack([pn, pn[:1]])
        else:
            vs = vs[:len(c)]
        self.vs, self.c, self.pn = vs, c, pn
        self.L, self.closed = track.length, track.closed

    def wrap(self, s):
        return np.mod(s, self.L) if self.closed else np.clip(s, 0.0, self.L)

    def at(self, s, off=0.0):
        s = self.wrap(np.asarray(s, float))
        cx = np.interp(s, self.vs, self.c[:, 0]); cy = np.interp(s, self.vs, self.c[:, 1])
        nx = np.interp(s, self.vs, self.pn[:, 0]); ny = np.interp(s, self.vs, self.pn[:, 1])
        nn = np.hypot(nx, ny) + 1e-9
        off = np.asarray(off, float)
        return np.stack([cx + nx / nn * off, cy + ny / nn * off], axis=-1)


class Camera:
    def __init__(self, W, H, fov_deg=90.0, pitch_deg=1.5, eye=(1.3, 0.37, 1.25)):
        self.W, self.H = W, H
        self.f = (W / 2) / math.tan(math.radians(fov_deg) / 2)
        self.cx, self.cy = W / 2, H / 2
        self.p = math.radians(pitch_deg)
        self.eye = eye

    def set_pose(self, x, y, yaw):
        fx, lx, h = self.eye
        c, s = math.cos(yaw), math.sin(yaw)
        self.c, self.s, self.yaw = c, s, yaw
        self.px = x + fx * c - lx * s
        self.py = y + fx * s + lx * c
        self.h = h

    def to_cam(self, P):
        """world (N,3) → camera (N,3) = (오른쪽 X, 아래 Y, 앞 Z)."""
        P = np.asarray(P, float)
        dx, dy = P[..., 0] - self.px, P[..., 1] - self.py
        fwd = dx * self.c + dy * self.s
        left = -dx * self.s + dy * self.c
        up = P[..., 2] - self.h
        cp, sp = math.cos(self.p), math.sin(self.p)
        Z = fwd * cp - up * sp
        Y = -(fwd * sp + up * cp)
        return np.stack([-left, Y, Z], axis=-1)

    def proj(self, C):
        Z = np.maximum(C[..., 2], 1e-6)
        return np.stack([self.cx + self.f * C[..., 0] / Z, self.cy + self.f * C[..., 1] / Z], axis=-1)

    @property
    def horizon_v(self):
        return self.cy - self.f * math.tan(self.p)


def clip_near(C):
    """카메라 좌표 다각형을 Z>NEAR 쪽만 남게 자른다(차 뒤쪽 도로가 화면에 뒤집혀 나오는 것 방지)."""
    out = []
    n = len(C)
    for i in range(n):
        a, b = C[i], C[(i + 1) % n]
        ia, ib = a[2] > NEAR, b[2] > NEAR
        if ia:
            out.append(a)
        if ia != ib:
            w = (NEAR - a[2]) / (b[2] - a[2])
            out.append(a + (b - a) * w)
    return np.array(out) if len(out) >= 3 else None


class DriverView:
    def __init__(self, env, W=640, H=360, kinds=None):
        self.env = env
        self.tr, self.rules = env.track, env.rules
        self.W, self.H = W, H
        self.cam = Camera(W, H)
        self.geo = RoadGeom(self.tr)
        self.kinds = [KIND_AS.get(k, k) for k in (kinds or env.ob_kind)]
        self.night = self.rules.light < 0.5
        self.rural = bool(self.rules.rural)
        self.amb = 1.0 if not self.night else 0.08 + 0.5 * self.rules.light
        rng = np.random.default_rng(7)
        L = self.tr.length
        w2 = self.tr.width / 2
        # 길가 풍경(고정): 나무, 전봇대, 건물, 표지
        self.trees = []
        if self.rural:
            for s in np.arange(5, L, 9.0):
                for side in (1, -1):
                    if rng.random() < 0.45:
                        off = side * (w2 + (2.2 if side > 0 else 3.0) + rng.uniform(0, 9))
                        self.trees.append((s + rng.uniform(-3, 3), off, rng.uniform(4.0, 7.5), rng.integers(3)))
            self.poles = [(s, w2 + 1.6) for s in np.arange(20, L, 45.0)]
        else:
            self.poles = [(s, -(w2 + 3.0)) for s in np.arange(15, L, 35.0)]
            for s in np.arange(5, L, 30.0):
                for side in (1, -1):
                    if rng.random() < 0.5:
                        self.trees.append((s + rng.uniform(-2, 2), side * (w2 + 2.0), rng.uniform(4.0, 6.0), 0))
        self.buildings = []
        if not self.rural:
            s = 0.0
            while s < L:
                ln = rng.uniform(14, 32)
                for side in (1, -1):
                    if rng.random() < 0.85:
                        off, h = side * (w2 + 4.5 + rng.uniform(0, 2)), rng.uniform(10, 45)
                        col = tuple(int(v) for v in rng.integers(110, 200, 3))
                        # 굽은 길에서 벽이 도로를 가로지르지 않게 4 m 조각으로 나눠 길을 따라 붙인다
                        cuts = np.linspace(s, s + ln * 0.9, max(2, int(ln * 0.9 / 4) + 1))
                        for a, b in zip(cuts[:-1], cuts[1:]):
                            self.buildings.append((a, b, off, h, col))
                s += ln
        self.signs = []  # (s, kind, value)
        lim = np.round(self.rules.limit_mps * 3.6).astype(int)
        ds = self.rules.ds
        prev = None
        for i in range(0, len(lim)):
            if lim[i] != prev or (i * ds) % 250 < ds and i > 0:
                self.signs.append((i * ds + 4, "limit", int(lim[i])))
                prev = lim[i]
        sch = np.asarray(self.rules.school, bool)
        for i in np.nonzero(sch & ~np.r_[False, sch[:-1]])[0]:
            self.signs.append((i * ds - 12, "school", 30))
        self.car_color = {k: CAR_COLORS[(k * 3 + 1) % len(CAR_COLORS)] for k in range(len(self.kinds))}
        self._hood_pts = None

    # ---------- 빛 ----------
    def light(self, X, Z):
        """밤: 주변광 + 전조등(하향등, 약 60 m, 좌우 ±25°). 낮: 1."""
        if not self.night:
            return 1.0
        ang = math.atan2(X + 0.0, max(Z, 0.1))
        beam = math.exp(-(ang / 0.34) ** 2) * max(0.0, 1.0 - max(Z - 4.0, 0.0) / 58.0) ** 0.8
        return self.amb + (1.0 - self.amb) * min(1.0, beam * 1.15)

    def shade(self, col, X, Z):
        k = self.light(X, Z)
        c = scale_rgb(col, k)
        if not self.night:
            c = mix(c, SKY_DAY[1], min(0.55, max(0.0, Z - 40) / 260.0))
        return c

    # ---------- 배경 ----------
    def _background(self, img, yaw, t):
        d = ImageDraw.Draw(img)
        hv = self.cam.horizon_v
        top, bot = SKY_NIGHT if self.night else SKY_DAY
        for v in range(0, int(hv) + 2, 3):
            d.rectangle([0, v, self.W, v + 3], fill=mix(top, bot, min(1.0, v / max(hv, 1))))
        if self.night:
            rng = np.random.default_rng(3)
            az = rng.uniform(-math.pi, math.pi, 260); el = rng.uniform(0.04, 0.6, 260)
            rel = np.angle(np.exp(1j * (az - yaw)))
            m = np.abs(rel) < 0.8
            u = self.cam.cx - self.cam.f * np.tan(rel[m]); v = hv - self.cam.f * np.tan(el[m])
            for uu, vv, b in zip(u, v, rng.uniform(90, 230, m.sum())):
                d.point((uu, vv), fill=(int(b), int(b), int(b)))
        # 지평선 산(방위각 고정 → 차가 돌면 산이 옆으로 흐름)
        us = np.arange(0, self.W + 8, 8)
        az = yaw + np.arctan((self.cam.cx - us) / self.cam.f)
        el = 0.035 + 0.02 * np.sin(3 * az) + 0.014 * np.sin(7 * az + 1) + 0.008 * np.sin(13 * az + 2)
        if not self.rural:
            el = el * 0.5
        vs = hv - self.cam.f * np.tan(el)
        hill = (14, 18, 28) if self.night else ((120, 150, 135) if self.rural else (150, 165, 175))
        d.polygon(list(zip(us, vs)) + [(self.W, hv + 1), (0, hv + 1)], fill=hill)
        if self.rural:
            el2 = 0.018 + 0.012 * np.sin(5 * az + 2) + 0.006 * np.sin(11 * az)
            vs2 = hv - self.cam.f * np.tan(el2)
            d.polygon(list(zip(us, vs2)) + [(self.W, hv + 1), (0, hv + 1)],
                      fill=(9, 12, 18) if self.night else (95, 125, 95))
        ground = (12, 12, 10) if self.night else ((165, 150, 95) if self.rural else (120, 120, 118))
        if self.night:
            ground = scale_rgb((150, 140, 100) if self.rural else (110, 110, 110), self.amb)
        d.rectangle([0, hv, self.W, self.H], fill=ground)

    # ---------- 도로 ----------
    def _strips(self, s0):
        w2 = self.tr.width / 2
        steps = np.r_[np.arange(-6, 30, 1.0), np.arange(30, 80, 2.0), np.arange(80, 170, 4.0)]
        S = s0 + steps
        if not self.tr.closed:
            S = S[(S >= 0) & (S <= self.tr.length)]
        return S, w2

    def _road_items(self, f, items):
        cam, rules = self.cam, self.rules
        S, w2 = self._strips(f["s"])
        if len(S) < 2:
            return
        paved_line = True
        strips = []  # (off_a, off_b, z, color_fn, layer)
        if self.rural:
            strips += [(w2 + 0.4, w2 + 14, 0.0, "field", 0), (w2, w2 + 0.4, 0.0, "shoulder", 0),
                       (-w2, -w2 - 0.35, 0.0, "shoulder", 0), (-w2 - 0.35, -w2 - 1.6, -0.3, "ditch", 0),
                       (-w2 - 1.6, -w2 - 14, 0.0, "field", 0)]
        else:
            strips += [(w2, w2 + 4.5, 0.12, "walk", 0), (-w2, -w2 - 4.5, 0.12, "walk", 0)]
        strips.append((-w2, w2, 0.0, "road", 1))
        P = {}
        offs = sorted({o for a, b, *_ in strips for o in (a, b)} | {-w2 + 0.2, -w2 + 0.35, w2 - 0.35, w2 - 0.2,
                                                                     -0.2, -0.08, 0.08, 0.2})
        for o in offs:
            P[o] = self.geo.at(S, o)
        for i in range(len(S) - 1):
            sa, sb = S[i], S[i + 1]
            ca = cam.to_cam(np.r_[self.geo.at(sa), 0.0][None])[0]
            cb = cam.to_cam(np.r_[self.geo.at(sb), 0.0][None])[0]
            if max(ca[2], cb[2]) < NEAR:
                continue
            depth = max(ca[2], cb[2])
            Xm, Zm = (ca[0] + cb[0]) / 2, max((ca[2] + cb[2]) / 2, NEAR)
            gravel = rules.gravel_at(sa)
            school = rules.school_at(sa)
            polys = []
            for a, b, z, kind, layer in strips:
                if kind == "road":
                    if gravel:
                        col = (150, 134, 108) if int(sa / 1.5) % 2 else (138, 123, 98)
                    elif school:
                        col = (150, 62, 52) if int(sa / 2) % 2 else (142, 58, 49)
                    else:
                        col = (72, 74, 77) if int(sa / 2) % 2 else (66, 68, 71)
                elif kind == "field":
                    col = (170, 152, 92) if int(sa / 6) % 2 else (158, 142, 86)  # 가을 논
                elif kind == "shoulder":
                    col = (125, 108, 80)
                elif kind == "ditch":
                    col = (38, 52, 50)
                else:
                    col = (150, 148, 140) if int(sa / 3) % 2 else (140, 138, 132)
                quad = [np.r_[P[a][i], z], np.r_[P[a][i + 1], z], np.r_[P[b][i + 1], z], np.r_[P[b][i], z]]
                if self.night:  # 전조등 빛은 칸마다 (길가 논밭은 빛 밖이면 캄캄)
                    qc = cam.to_cam(np.mean(quad, axis=0)[None])[0]
                    polys.append((quad, self.shade(col, qc[0], max(qc[2], NEAR))))
                else:
                    polys.append((quad, self.shade(col, Xm, Zm)))
            # 차선 표시 (자갈길에는 없음)
            if not gravel and paved_line:
                white = self.shade((232, 232, 228), Xm, Zm)
                for a, b in ((w2 - 0.35, w2 - 0.2), (-w2 + 0.2, -w2 + 0.35)):
                    polys.append(([np.r_[P[a][i], 0], np.r_[P[a][i + 1], 0], np.r_[P[b][i + 1], 0], np.r_[P[b][i], 0]], white))
                if rules.twoway_at(sa) and self.tr.width >= 6.0:  # 중앙선(노란 실선 두 줄)
                    yel = self.shade((240, 190, 40), Xm, Zm)
                    for a, b in ((-0.2, -0.08), (0.08, 0.2)):
                        polys.append(([np.r_[P[a][i], 0], np.r_[P[a][i + 1], 0], np.r_[P[b][i + 1], 0], np.r_[P[b][i], 0]], yel))
                elif self.tr.width >= 6.0 and int(sa / 4) % 3 == 0:  # 같은 방향 2차로: 흰 점선 차선
                    a, b = -0.08, 0.08
                    polys.append(([np.r_[P[a][i], 0], np.r_[P[a][i + 1], 0], np.r_[P[b][i + 1], 0], np.r_[P[b][i], 0]], white))
            items.append((depth, 0, polys))
        # 정지선 (신호 앞, 내 차로 = 오른쪽 절반)
        for sg in rules.signals:
            g = (sg.s - f["s"]) % self.tr.length if self.tr.closed else sg.s - f["s"]
            if -3 < g < 150:
                A = self.geo.at(np.array([sg.s - 0.5, sg.s]), np.array([[-w2 + 0.2], [-0.25]]))
                quad = [np.r_[A[0][0], 0], np.r_[A[0][1], 0], np.r_[A[1][1], 0], np.r_[A[1][0], 0]]
                C = cam.to_cam(np.array(quad))
                items.append((C[:, 2].max() - 0.01, 1, [(quad, self.shade((240, 240, 240), C[:, 0].mean(), C[:, 2].mean()))]))

    # ---------- 길가 ----------
    def _gap(self, s, s0):
        g = s - s0
        if self.tr.closed:
            g = (g + self.tr.length / 2) % self.tr.length - self.tr.length / 2
        return g

    def _scenery(self, f, items):
        s0 = f["s"]
        for s, off, h, k in self.trees:
            g = self._gap(s, s0)
            if -2 < g < 140:
                p = self.geo.at(s, off)
                items.append(("tree", p, h, k))
        for s, off in self.poles:
            if -2 < self._gap(s, s0) < 150:
                items.append(("pole", self.geo.at(s, off)))
        for sa, sb, off, h, col in self.buildings:
            if -10 < self._gap(sb, s0) and self._gap(sa, s0) < 170:
                A, B = self.geo.at(sa, off), self.geo.at(sb, off)
                items.append(("wall", A, B, h, col))
        for s, kind, val in self.signs:
            if -1 < self._gap(s, s0) < 110:
                items.append(("sign", self.geo.at(s, -(self.tr.width / 2 + 0.9)), kind, val))
        for sg in self.rules.signals:
            g = self._gap(sg.s, s0)
            if -1 < g < 160:
                items.append(("signal", sg))

    def _draw_billboard(self, d, kind, a, f):
        cam = self.cam
        if kind == "tree":
            _, p, h, k = a
            C = cam.to_cam(np.array([[p[0], p[1], 0.0], [p[0], p[1], h]]))
            if C[0, 2] < NEAR:
                return
            u, v0 = cam.proj(C[0]); _, v1 = cam.proj(C[1])
            sc = cam.f / C[0, 2]
            lk = self.light(C[0, 0], C[0, 2])
            if self.night:
                lk = self.amb + (lk - self.amb) * 0.6  # 하향등은 높은 곳을 덜 비춤
            trunk = scale_rgb((90, 65, 40), lk)
            crown = scale_rgb([(70, 110, 55), (150, 120, 50), (95, 120, 60)][k] if self.rural else (75, 120, 70), lk)
            if not self.night:
                crown = mix(crown, SKY_DAY[1], min(0.5, max(0, C[0, 2] - 40) / 260))
            d.rectangle([u - 0.15 * sc, v0 - 0.45 * (v0 - v1), u + 0.15 * sc, v0], fill=trunk)
            r = 1.4 * sc
            d.ellipse([u - r, v1, u + r, v0 - 0.35 * (v0 - v1)], fill=crown)
        elif kind == "pole":
            p = a[1]
            C = cam.to_cam(np.array([[p[0], p[1], 0.0], [p[0], p[1], 9.0]]))
            if C[0, 2] < NEAR:
                return
            (u0, v0), (u1, v1) = cam.proj(C)
            sc = cam.f / C[0, 2]
            col = scale_rgb((120, 115, 105), self.light(C[0, 0], C[0, 2]))
            d.rectangle([u0 - 0.13 * sc, v1, u0 + 0.13 * sc, v0], fill=col)
            d.rectangle([u1 - 0.8 * sc, v1 + 0.5 * sc, u1 + 0.8 * sc, v1 + 0.62 * sc], fill=col)
        elif kind == "wall":
            _, A, B, h, col = a
            quad = [np.r_[A, 0], np.r_[B, 0], np.r_[B, h], np.r_[A, h]]
            C = clip_near(cam.to_cam(np.array(quad)))
            if C is None:
                return
            Zm = max(C[:, 2].mean(), NEAR)
            c = self.shade(col, C[:, 0].mean(), Zm)
            if self.night:
                c = scale_rgb(col, 0.25)
            d.polygon([tuple(q) for q in cam.proj(C)], fill=c, outline=scale_rgb(c, 0.8))
        elif kind == "sign":
            _, p, sk, val = a
            C = cam.to_cam(np.array([[p[0], p[1], 0.0], [p[0], p[1], 2.3]]))
            if C[0, 2] < 1.0:
                return
            (u0, v0), (u1, v1) = cam.proj(C)
            sc = cam.f / C[0, 2]
            lk = max(self.light(C[0, 0], C[0, 2]), 0.25 if self.night else 1.0)  # 반사 표지
            d.rectangle([u0 - 0.04 * sc, v1, u0 + 0.04 * sc, v0], fill=scale_rgb((150, 150, 150), lk))
            r = 0.38 * sc
            if sk == "limit":
                d.ellipse([u1 - r, v1 - r, u1 + r, v1 + r], fill=scale_rgb((215, 30, 35), lk))
                d.ellipse([u1 - 0.78 * r, v1 - 0.78 * r, u1 + 0.78 * r, v1 + 0.78 * r], fill=scale_rgb((250, 250, 250), lk))
                if r > 6:
                    text_center(d, (u1, v1), str(val), r * 0.95, scale_rgb((10, 10, 10), lk))
            else:
                d.rectangle([u1 - 1.3 * r, v1 - 1.1 * r, u1 + 1.3 * r, v1 + 1.1 * r], fill=scale_rgb((250, 205, 30), lk),
                            outline=scale_rgb((20, 20, 20), lk))
                if r > 7:
                    text_center(d, (u1, v1 - 0.45 * r), "어린이", r * 0.55, scale_rgb((10, 10, 10), lk))
                    text_center(d, (u1, v1 + 0.35 * r), "보호구역", r * 0.5, scale_rgb((10, 10, 10), lk))
        elif kind == "signal":
            sg = a[1]
            w2 = self.tr.width / 2
            pole = self.geo.at(sg.s + 12, -(w2 + 1.0))
            head = self.geo.at(sg.s + 12, -w2 * 0.35)
            C = cam.to_cam(np.array([[pole[0], pole[1], 0.0], [pole[0], pole[1], 5.6], [head[0], head[1], 5.6]]))
            if C[:, 2].min() < NEAR:
                return
            P = cam.proj(C)
            sc = cam.f / C[2, 2]
            pc = scale_rgb((90, 90, 90), max(self.light(C[0, 0], C[0, 2]), 0.3 if self.night else 1))
            d.line([tuple(P[0]), tuple(P[1])], fill=pc, width=max(1, int(0.2 * sc)))
            d.line([tuple(P[1]), tuple(P[2])], fill=pc, width=max(1, int(0.12 * sc)))
            u, v = P[2]
            bw, bh = 0.75 * sc, 0.2 * sc
            d.rectangle([u - bw, v - bh, u + bw, v + bh], fill=(25, 25, 25))
            st = sg.state(f["t"])
            for k, (name, on) in enumerate((("red", (255, 50, 40)), ("yellow", (255, 200, 30)), ("arrow", (40, 220, 90)),
                                            ("green", (40, 220, 90)))):
                cu = u - bw + bw * (0.25 + 0.5 * k)
                lit = name == st
                col = on if lit else scale_rgb(on, 0.18)
                rr = 0.75 * bh
                if lit and rr > 1.5:
                    d.ellipse([cu - 2.2 * rr, v - 2.2 * rr, cu + 2.2 * rr, v + 2.2 * rr], fill=mix(on, (25, 25, 25), 0.75))
                d.ellipse([cu - rr, v - rr, cu + rr, v + rr], fill=col)

    # ---------- 다른 차·보행자·물건 ----------
    def _box_faces(self, cx, cy, hd, L, W, z0, z1, dl=0.0):
        c, s = math.cos(hd), math.sin(hd)
        pts = []
        for a, b in ((-L / 2 + dl, -W / 2), (L / 2 + dl, -W / 2), (L / 2 + dl, W / 2), (-L / 2 + dl, W / 2)):
            pts.append((cx + a * c - b * s, cy + a * s + b * c))
        bot = [np.r_[p, z0] for p in pts]
        top = [np.r_[p, z1] for p in pts]
        faces = {"rear": [bot[0], bot[3], top[3], top[0]], "front": [bot[1], bot[2], top[2], top[1]],
                 "right": [bot[0], bot[1], top[1], top[0]], "left": [bot[3], bot[2], top[2], top[3]],
                 "top": top}
        normals = {"rear": (-c, -s), "front": (c, s), "right": (s, -c), "left": (-s, c), "top": None}
        return faces, normals

    def _draw_car(self, d, row, k, oncoming):
        cam = self.cam
        x, y, r, L, hd = row[:5]
        if oncoming:
            hd = hd + math.pi
        W = 2 * r
        base = self.car_color[k]
        C0 = cam.to_cam(np.array([[x, y, 0.0]]))[0]
        if C0[2] < NEAR + 0.5:
            return
        lk = self.light(C0[0], C0[2])
        # 바닥 그림자
        f_sh, _ = self._box_faces(x, y, hd, L * 1.02, W * 1.05, 0.01, 0.01)
        Csh = clip_near(cam.to_cam(np.array(f_sh["top"])))
        if Csh is not None:
            d.polygon([tuple(q) for q in cam.proj(Csh)], fill=(20, 20, 20) if not self.night else (5, 5, 5))
        for (Lb, Wb, z0, z1, dl, shade) in ((L, W, 0.25, 0.95, 0.0, 1.0), (L * 0.55, W * 0.86, 0.95, 1.45, -0.05 * L, 0.85)):
            faces, normals = self._box_faces(x, y, hd, Lb, Wb, z0, z1, dl)
            order = []
            for name, poly in faces.items():
                n = normals[name]
                ctr = np.mean(poly, axis=0)
                if n is not None and (cam.px - ctr[0]) * n[0] + (cam.py - ctr[1]) * n[1] <= 0:
                    continue
                if n is None and cam.h <= z1:
                    continue
                C = clip_near(cam.to_cam(np.array(poly)))
                if C is None:
                    continue
                k_face = {"top": 1.1, "rear": 0.9, "front": 0.9, "left": 0.7, "right": 0.7}[name] * shade
                col = scale_rgb(base, k_face * lk)
                if shade < 1.0 and name in ("rear", "front", "left", "right"):
                    col = scale_rgb((40, 55, 70), lk * 1.2)  # 유리
                order.append((C[:, 2].mean(), C, col, name))
            for _, C, col, name in sorted(order, key=lambda q: -q[0]):
                d.polygon([tuple(q) for q in cam.proj(C)], fill=col, outline=scale_rgb(col, 0.7))
        # 등 (내 쪽을 보는 면)
        c, s = math.cos(hd), math.sin(hd)
        face_x = -L / 2 if not oncoming else L / 2
        facing = (cam.px - x) * (c if oncoming else -c) + (cam.py - y) * (s if oncoming else -s) > 0
        if facing:
            lamp = (255, 245, 210) if oncoming else (230, 25, 25)
            for side in (-1, 1):
                b = side * (W / 2 - 0.25)
                px, py = x + face_x * c - b * s, y + face_x * s + b * c
                Cl = cam.to_cam(np.array([[px, py, 0.7]]))[0]
                if Cl[2] < NEAR:
                    continue
                u, v = cam.proj(Cl)
                rr = max(1.0, 0.13 * cam.f / Cl[2])
                if self.night:
                    glow = rr * (4.5 if oncoming else 2.2)
                    d.ellipse([u - glow, v - glow * 0.7, u + glow, v + glow * 0.7], fill=mix(lamp, (0, 0, 0), 0.55))
                d.ellipse([u - rr, v - rr * 0.7, u + rr, v + rr * 0.7], fill=lamp if (self.night or oncoming) else scale_rgb(lamp, 0.8))

    def _draw_ped(self, d, row, k, t):
        cam = self.cam
        x, y = row[:2]
        C = cam.to_cam(np.array([[x, y, 0.0], [x, y, 1.7]]))
        if C[0, 2] < NEAR + 0.3:
            return
        (u, v0), (_, v1) = cam.proj(C)
        sc = cam.f / C[0, 2]
        lk = self.light(C[0, 0], C[0, 2])
        if self.night:
            lk = max(lk, 0.05)
        h = v0 - v1
        coat = scale_rgb([(200, 60, 40), (40, 90, 170), (230, 200, 60)][k % 3], lk)
        skin = scale_rgb((230, 190, 160), lk)
        dark = scale_rgb((40, 40, 50), lk)
        sw = math.sin(t * 6 + k) * 0.08 * sc
        d.rectangle([u - 0.14 * sc + sw, v0 - 0.48 * h, u - 0.02 * sc + sw, v0], fill=dark)
        d.rectangle([u + 0.02 * sc - sw, v0 - 0.48 * h, u + 0.14 * sc - sw, v0], fill=dark)
        d.rectangle([u - 0.22 * sc, v0 - 0.85 * h, u + 0.22 * sc, v0 - 0.45 * h], fill=coat)
        d.ellipse([u - 0.12 * sc, v1, u + 0.12 * sc, v1 + 0.15 * h], fill=skin)
        if self.night and lk > 0.3:  # 반사띠
            d.rectangle([u - 0.22 * sc, v0 - 0.62 * h, u + 0.22 * sc, v0 - 0.58 * h], fill=(240, 240, 200))

    def _draw_object(self, d, row):
        cam = self.cam
        x, y, r, L, hd = row[:5]
        faces, normals = self._box_faces(x, y, hd, max(L, 0.6), 2 * r, 0.0, 0.5)
        C0 = cam.to_cam(np.array([[x, y, 0.0]]))[0]
        lk = self.light(C0[0], C0[2])
        for name in ("rear", "left", "right", "top"):
            n = normals[name]
            ctr = np.mean(faces[name], axis=0)
            if n is not None and (cam.px - ctr[0]) * n[0] + (cam.py - ctr[1]) * n[1] <= 0:
                continue
            C = clip_near(cam.to_cam(np.array(faces[name])))
            if C is None:
                continue
            col = scale_rgb((120, 110, 95), lk * (1.1 if name == "top" else 0.85))
            d.polygon([tuple(q) for q in cam.proj(C)], fill=col)

    # ---------- 차 안 ----------
    def _draw_hood(self, d, lk):
        """보닛: 차 기준 앞 끝(눈에서 2.45 m 앞, 높이 0.85 m)을 투영한 곡선 아래를 칠한다(차와 함께 움직이므로 고정)."""
        cam = self.cam
        if self._hood_pts is None:
            lat = np.linspace(-0.9, 0.9, 31)
            pts = np.stack([2.45 - 0.15 * (lat / 0.9) ** 4, lat - cam.eye[1], 0.85 - 0.04 * (lat / 0.9) ** 2], axis=1)
            cp, sp = math.cos(cam.p), math.sin(cam.p)
            fwd, left, up = pts[:, 0], pts[:, 1], pts[:, 2] - cam.eye[2]
            C = np.stack([-left, -(fwd * sp + up * cp), fwd * cp - up * sp], axis=1)
            self._hood_pts = cam.proj(C)
        P = self._hood_pts
        body = scale_rgb((30, 44, 70), max(lk, 0.3))
        edge = sorted((tuple(p) for p in P), key=lambda q: q[0])  # 왼쪽→오른쪽
        P = np.array(edge)
        d.polygon(edge + [(self.W, P[-1][1] + 0.12 * self.H), (self.W, self.H), (0, self.H), (0, P[0][1] + 0.12 * self.H)], fill=body)
        # 앞 유리에 비친 하늘처럼 보닛 위 은은한 반사
        hl = scale_rgb((62, 82, 112), max(lk, 0.3))
        d.line(edge, fill=hl, width=3)
        # A필러(왼쪽은 운전석이라 가깝고 굵게)
        d.polygon([(0, 0), (30, 0), (12, self.H * 0.62), (0, self.H * 0.66)], fill=(18, 18, 20))
        d.polygon([(self.W, 0), (self.W - 16, 0), (self.W - 5, self.H * 0.6), (self.W, self.H * 0.62)], fill=(18, 18, 20))

    def _dashboard(self, img, f):
        W, H = self.W, self.H
        dh = int(H * 0.2)
        ov = Image.new("RGBA", (W, dh), (16, 16, 20, 225))
        img.paste(ov, (0, H - dh), ov)
        d = ImageDraw.Draw(img)
        y0 = H - dh
        cy = y0 + dh / 2
        # 핸들 각도 (조향비 15:1 가정)
        wheel = f["steer"] * 15.0
        cx, R = 58, dh * 0.42
        d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(210, 210, 210), width=max(2, int(R * 0.18)))
        for a0 in (0.0, math.pi, -math.pi / 2):
            a = a0 - wheel
            d.line([(cx, cy), (cx + R * 0.9 * math.cos(a), cy + R * 0.9 * math.sin(a))], fill=(190, 190, 190), width=3)
        d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(120, 120, 120))
        mark = -math.pi / 2 - wheel
        d.ellipse([cx + R * math.cos(mark) - 3, cy + R * math.sin(mark) - 3, cx + R * math.cos(mark) + 3,
                   cy + R * math.sin(mark) + 3], fill=(255, 200, 40))
        d.text((cx + R + 6, cy + R - 10), f"핸들 {math.degrees(wheel):+.0f}°", font=font(11), fill=(220, 220, 220))
        # 속도계
        sx, SR = 205, dh * 0.46
        kmh, lim = f["v"] * 3.6, f["limit"] * 3.6
        d.arc([sx - SR, cy - SR, sx + SR, cy + SR], 150, 390, fill=(200, 200, 200), width=3)
        for k in range(0, 121, 20):
            a = math.radians(150 + 240 * k / 120)
            d.line([(sx + SR * 0.82 * math.cos(a), cy + SR * 0.82 * math.sin(a)), (sx + SR * math.cos(a), cy + SR * math.sin(a))],
                   fill=(200, 200, 200), width=2)
        al = math.radians(150 + 240 * min(lim, 120) / 120)
        d.line([(sx + SR * 0.7 * math.cos(al), cy + SR * 0.7 * math.sin(al)), (sx + SR * math.cos(al), cy + SR * math.sin(al))],
               fill=(230, 40, 40), width=4)
        a = math.radians(150 + 240 * min(kmh, 120) / 120)
        over = kmh > lim + 1
        d.line([(sx, cy), (sx + SR * 0.9 * math.cos(a), cy + SR * 0.9 * math.sin(a))], fill=(255, 90, 60) if over else (255, 255, 255), width=3)
        text_center(d, (sx, cy + SR * 0.45), f"{kmh:.0f}", int(SR * 0.55), (255, 120, 90) if over else (255, 255, 255))
        text_center(d, (sx, cy + SR * 0.85), "km/h", 9, (170, 170, 170))
        # 제한속도 표지
        lx, LR = 290, dh * 0.33
        d.ellipse([lx - LR, cy - LR, lx + LR, cy + LR], fill=(220, 30, 35))
        d.ellipse([lx - LR * 0.76, cy - LR * 0.76, lx + LR * 0.76, cy + LR * 0.76], fill=(250, 250, 250))
        text_center(d, (lx, cy), f"{lim:.0f}", int(LR * 0.95), (10, 10, 10))
        if f["school"]:
            d.rectangle([lx - 26, y0 + 3, lx + 26, y0 + 15], fill=(250, 205, 30))
            text_center(d, (lx, y0 + 9), "어린이보호구역", 9, (0, 0, 0))
        # 상태 글
        lines = []
        lines.append(("자갈길(비포장) · 마찰 %.2f" % f["mu"]) if f["gravel"] else ("포장도로 · 마찰 %.2f" % f["mu"]))
        lines.append(f"시야 {f['sight']:.0f} m" + ("   · 밤(가로등 없음)" if self.night else ""))
        lines.append("앞차·장애물 TTC " + ("-" if f["ttc"] is None else f"{f['ttc']:.1f} s"))
        if f["slipping"]:
            lines.append("미끄러짐!")
        for k, ln in enumerate(lines[:4]):
            d.text((330, y0 + 6 + k * 15), ln, font=font(12), fill=(255, 120, 90) if "미끄러짐" in ln else (225, 225, 225))

    def _banner(self, img, f):
        d = ImageDraw.Draw(img)
        mode = f["mode"]
        col = MODE_RGB.get(mode, (69, 123, 157))
        if mode in ("TAKEOVER", "SAFE_STOP") and (f["t"] * 2) % 1 < 0.5:
            col = scale_rgb(col, 1.25)
        txt = MODE_LABEL.get(mode, mode)
        fw = font(15)
        w = d.textlength(txt, font=fw) + 28
        x0 = self.W / 2 - w / 2
        d.rounded_rectangle([x0, 6, x0 + w, 30], radius=8, fill=col)
        d.text((self.W / 2, 18), txt, font=fw, fill=(255, 255, 255), anchor="mm")
        if f["risks"]:
            r = "⚠ " + ", ".join(f["risks"])[:46]
            d.text((self.W / 2, 42), r, font=font(12), fill=(255, 90, 70), anchor="mm", stroke_width=2, stroke_fill=(0, 0, 0))

    def _minimap(self, img, f, size=132):
        W = self.W
        m = Image.new("RGBA", (size, size), (20, 24, 30, 200))
        d = ImageDraw.Draw(m)
        sc = size / 90.0
        c, s = math.cos(f["yaw"]), math.sin(f["yaw"])
        ox, oy = size / 2, size * 0.68

        def tp(P):
            P = np.atleast_2d(P)
            dx, dy = P[:, 0] - f["x"], P[:, 1] - f["y"]
            fwd = dx * c + dy * s
            left = -dx * s + dy * c
            return list(zip(ox - left * sc, oy - fwd * sc))
        S = np.arange(f["s"] - 40, f["s"] + 70, 2.0)
        if not self.tr.closed:
            S = S[(S >= 0) & (S <= self.tr.length)]
        if len(S) > 1:
            w2 = self.tr.width / 2
            Lp, Rp = self.geo.at(S, w2), self.geo.at(S, -w2)
            road = (150, 134, 108) if self.rules.gravel_at(f["s"]) else (95, 95, 100)
            d.polygon(tp(Lp) + tp(Rp)[::-1], fill=road)
            if self.rules.twoway_at(f["s"]) and self.tr.width >= 6:
                d.line(tp(self.geo.at(S, 0.0)), fill=(240, 190, 40), width=1)
        for k, row in enumerate(f["ob"]):
            kind = self.kinds[k] if k < len(self.kinds) else "car"
            col = {"car": (200, 200, 200), "oncoming": (255, 120, 100), "ped": (200, 120, 255), "object": (160, 140, 110)}.get(kind, (220, 220, 220))
            (u, v), = tp(row[:2])
            rr = max(2.0, row[2] * sc)
            d.ellipse([u - rr, v - rr, u + rr, v + rr], fill=col)
        d.polygon([(ox, oy - 7), (ox - 4, oy + 4), (ox + 4, oy + 4)], fill=MODE_RGB.get(f["mode"], (0, 180, 216)))
        d.text((4, 2), "미니맵", font=font(10), fill=(220, 220, 220))
        img.paste(m, (W - size - 8, 8), m)

    # ---------- 한 장 ----------
    def render(self, f) -> Image.Image:
        cam = self.cam
        cam.set_pose(f["x"], f["y"], f["yaw"])
        img = Image.new("RGB", (self.W, self.H))
        self._background(img, f["yaw"], f["t"])
        d = ImageDraw.Draw(img)
        items = []
        road = []
        self._road_items(f, road)
        for depth, layer, polys in road:
            items.append((depth, layer, "road", polys))
        scen = []
        self._scenery(f, scen)
        for it in scen:
            kind = it[0]
            if kind == "signal":
                sg = it[1]
                p = self.geo.at(sg.s + 12, 0.0)
            elif kind == "wall":
                p = (it[1] + it[2]) / 2
            else:
                p = it[1]
            Z = cam.to_cam(np.array([[p[0], p[1], 0.0]]))[0, 2]
            if kind == "wall":
                Z = min(Z, cam.to_cam(np.array([[*it[1], 0.0], [*it[2], 0.0]]))[:, 2].max())
            if Z > NEAR or kind == "wall":
                items.append((Z, 2, kind, it))
        for k, row in enumerate(f["ob"]):
            kind = self.kinds[k] if k < len(self.kinds) else "car"
            Z = cam.to_cam(np.array([[row[0], row[1], 0.0]]))[0, 2]
            if Z > NEAR:
                items.append((Z, 3, "ob", (k, row, kind)))
        items.sort(key=lambda q: (-q[0], q[1]))
        for depth, layer, kind, data in items:
            if kind == "road":
                for quad, col in data:
                    C = clip_near(cam.to_cam(np.array(quad)))
                    if C is not None:
                        d.polygon([tuple(q) for q in cam.proj(C)], fill=col)
            elif kind == "ob":
                k, row, ok = data
                if ok == "ped":
                    self._draw_ped(d, row, k, f["t"])
                elif ok == "object":
                    self._draw_object(d, row)
                else:
                    self._draw_car(d, row, k, oncoming=(ok == "oncoming" or row[5] < -0.5))
            else:
                self._draw_billboard(d, kind, data, f)
        if self.night:  # 전조등 빛이 닿는 곳 살짝 밝게
            pass
        self._draw_hood(d, 1.0 if not self.night else 0.25)
        self._dashboard(img, f)
        self._banner(img, f)
        self._minimap(img, f)
        d = ImageDraw.Draw(img)
        d.text((40, 6), f"운전석 시점  t={f['t']:.1f}s", font=font(12), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        return img
