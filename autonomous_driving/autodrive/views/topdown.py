"""위에서 본 지도 패널 (Pillow). 기존 matplotlib 화면과 같은 정보를 빠르게 그린다(북쪽 위, 차 따라 이동)."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .common import KIND_AS, MODE_RGB, font
from .driver import RoadGeom

KIND_RGB = {"car": (108, 117, 125), "oncoming": (157, 2, 8), "ped": (114, 9, 183), "object": (52, 58, 64)}


class TopDownView:
    def __init__(self, env, W=640, H=360, kinds=None, view_m=30.0, note=None):
        self.env, self.W, self.H = env, W, H
        self.tr, self.rules = env.track, env.rules
        self.geo = RoadGeom(self.tr)
        self.kinds = [KIND_AS.get(k, k) for k in (kinds or env.ob_kind)]
        self.sc = H / (2 * view_m)
        self.note = note
        self.night = self.rules.light < 0.5

    def background(self, f):
        """배경 그림을 바꿔 끼우는 자리(위성 패널이 덮어씀). 기본은 단색."""
        return Image.new("RGB", (self.W, self.H), (29, 53, 87) if self.night else ((167, 201, 87) if self.rules.rural else (150, 150, 145)))

    def render(self, f, bg: Image.Image | None = None, road_alpha=255) -> Image.Image:
        img = (bg or self.background(f)).convert("RGBA")
        lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        W, H, sc = self.W, self.H, self.sc
        fx, fy = f["x"], f["y"]

        def tp(P):
            P = np.atleast_2d(P)
            return list(zip(W / 2 + (P[:, 0] - fx) * sc, H / 2 - (P[:, 1] - fy) * sc))
        span = 1.6 * max(W, H) / sc
        S = np.arange(f["s"] - span, f["s"] + span, 2.0)
        if not self.tr.closed:
            S = S[(S >= 0) & (S <= self.tr.length)]
        w2 = self.tr.width / 2
        if len(S) > 1:
            road = (183, 183, 164) if self.rules.rural else (108, 117, 125)
            # 구간별 색 (자갈길)
            seg = []
            for i in range(len(S) - 1):
                g = self.rules.gravel_at(S[i])
                if not seg or seg[-1][0] != g:
                    seg.append([g, [S[i]]])
                seg[-1][1].append(S[i + 1])
            for g, ss in seg:
                ss = np.array(ss)
                col = (176, 150, 112) if g else road
                d.polygon(tp(self.geo.at(ss, w2)) + tp(self.geo.at(ss, -w2))[::-1], fill=col + (road_alpha,))
            d.line(tp(self.geo.at(S, w2)), fill=(255, 255, 255, 255), width=2)
            d.line(tp(self.geo.at(S, -w2)), fill=(255, 255, 255, 255), width=2)
            if self.rules.twoway_at(f["s"]) and self.tr.width >= 6:
                d.line(tp(self.geo.at(S, 0.0)), fill=(255, 209, 102, 255), width=1)
        for sg in self.rules.signals:
            (u, v), = tp(self.geo.at(sg.s, -w2 / 2))
            st = sg.state(f["t"])
            d.rectangle([u - 4, v - 4, u + 4, v + 4], fill={"green": (45, 198, 83), "yellow": (255, 214, 10), "red": (229, 56, 59)}[st] + (255,))
        for k, row in enumerate(f["ob"]):
            kind = self.kinds[k] if k < len(self.kinds) else "car"
            if kind == "car" and row[5] < -0.5:
                kind = "oncoming"
            col = KIND_RGB.get(kind, (60, 60, 60)) + (255,)
            if kind in ("car", "oncoming"):
                d.polygon(tp(_box(row[0], row[1], row[4], row[3], 2 * row[2])), fill=col)
            else:
                (u, v), = tp(row[:2])
                r = max(3, row[2] * sc)
                d.ellipse([u - r, v - r, u + r, v + r], fill=col)
        car = _box(fx + 1.2 * math.cos(f["yaw"]), fy + 1.2 * math.sin(f["yaw"]), f["yaw"], 4.0, 1.8)
        d.polygon(tp(car), fill=MODE_RGB.get(f["mode"], (0, 180, 216)) + (255,), outline=(255, 255, 255, 255))
        # 진행 방향 표시
        (u0, v0), = tp([fx, fy]); (u1, v1), = tp([fx + 6 * math.cos(f["yaw"]), fy + 6 * math.sin(f["yaw"])])
        d.line([(u0, v0), (u1, v1)], fill=(255, 255, 255, 200), width=2)
        img = Image.alpha_composite(img, lay)
        d = ImageDraw.Draw(img)
        d.text((8, 6), "위에서 본 지도" + (f" · {self.note}" if self.note else ""), font=font(12), fill=(255, 255, 255),
               stroke_width=2, stroke_fill=(0, 0, 0))
        # 축척 막대 20 m
        d.line([(10, H - 14), (10 + 20 * sc, H - 14)], fill=(255, 255, 255), width=3)
        d.text((14 + 20 * sc, H - 22), "20 m", font=font(10), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        return img.convert("RGB")


def _box(x, y, hd, L, W):
    c, s = math.cos(hd), math.sin(hd)
    return np.array([[x + a * c - b * s, y + a * s + b * c] for a, b in ((-L / 2, -W / 2), (L / 2, -W / 2), (L / 2, W / 2), (-L / 2, W / 2))])
