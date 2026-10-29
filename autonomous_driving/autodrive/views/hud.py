"""운전석 중심 화면(기본 배치) — 미니멀 HUD.

큰 운전석 시점 한 장 위에 꼭 필요한 것만 얹는다(평평한 색, 둥근 모서리, 산세리프, 적은 정보):
- 왼쪽 위 카드: 큰 디지털 속도, P R N D(전기차 1단), 제한속도 표지, 배터리 SOC 막대·주행 가능 거리, 전력 막대(회생은 초록·음수)
- 가운데 위: 모드 표시 — 자율 주행은 작은 점+글자, 개입 요청(TAKEOVER)·안전 정지는 빨간 띠로 확실히
- 오른쪽 아래: 반투명 둥근 미니맵(진행 방향 위쪽)
- 모터룸·위성은 기본으로 숨김. show=("motor", "satellite")이면 오른쪽에 작은 카드로 밀려 들어옴(slide 0→1)
main으로 큰 화면을 바꿀 수 있다(driver | top | motor | satellite). 운전석이 큰 화면이 아니면 왼쪽 아래에 작은 운전석 카드.
낮에는 밝은 카드, 밤에는 어두운 카드.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .common import KIND_AS, font
from .driver import DriverView, RoadGeom
from .motor import LABEL as MOTOR_LABEL
from .motor import MotorBayView
from .satellite_panel import SatelliteView
from .topdown import TopDownView

MAINS = ("driver", "top", "motor", "satellite")
EXTRAS = ("motor", "satellite")
TEAL = (38, 190, 160)
RED = (226, 54, 54)
GREEN = (52, 199, 120)
BLUE = (62, 130, 247)
NAMES = {"driver": "운전석", "top": "지도", "motor": "모터룸", "satellite": "위성"}


def _rounded_mask(size, r):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=r, fill=255)
    return m


class MiniMap:
    """진행 방향이 위인 단순 지도(도로·차·보행자·신호)."""

    def __init__(self, env, kinds, size=(340, 230), view_m=75.0, dark=True):
        self.tr, self.rules = env.track, env.rules
        self.geo = RoadGeom(self.tr)
        self.kinds = [KIND_AS.get(k, k) for k in (kinds or env.ob_kind)]
        self.size, self.view_m, self.dark = size, view_m, dark

    def render(self, f, alpha=215):
        W, H = self.size
        bg = (28, 30, 34, alpha) if self.dark else (236, 238, 241, alpha)
        img = Image.new("RGBA", (W, H), bg)
        d = ImageDraw.Draw(img)
        sc = H / self.view_m
        c, s = math.cos(f["yaw"]), math.sin(f["yaw"])
        ox, oy = W / 2, H * 0.7

        def tp(P):
            P = np.atleast_2d(P)
            dx, dy = P[:, 0] - f["x"], P[:, 1] - f["y"]
            fwd, left = dx * c + dy * s, -dx * s + dy * c
            return list(zip(ox - left * sc, oy - fwd * sc))
        S = np.arange(f["s"] - 40, f["s"] + 100, 1.5)
        if not self.tr.closed:
            S = S[(S >= 0) & (S <= self.tr.length)]
        if len(S) > 1:
            w2 = max(self.tr.width / 2, 2.2)
            road = (78, 82, 90, 255) if self.dark else (255, 255, 255, 255)
            d.polygon(tp(self.geo.at(S, w2)) + tp(self.geo.at(S, -w2))[::-1], fill=road)
            ahead = S[S >= f["s"]]
            if len(ahead) > 1:  # 갈 길(파란 선)
                d.line(tp(self.geo.at(ahead, -w2 / 2 if self.tr.width >= 6 else 0.0)), fill=BLUE + (255,), width=4, joint="curve")
        for sg in self.rules.signals:
            if -60 < sg.s - f["s"] < 140:
                (u, v), = tp(self.geo.at(sg.s, -self.tr.width / 2 - 1.5))
                col = {"green": (45, 198, 83), "yellow": (255, 214, 10), "red": (229, 56, 59)}[sg.state(f["t"])]
                d.ellipse([u - 4, v - 4, u + 4, v + 4], fill=col + (255,))
        for k, row in enumerate(f["ob"]):
            kind = self.kinds[k] if k < len(self.kinds) else "car"
            if kind == "car" and row[5] < -0.5:
                kind = "oncoming"
            col = {"ped": (175, 110, 255), "object": (170, 150, 120), "oncoming": (235, 120, 90)}.get(
                kind, (190, 196, 206) if self.dark else (90, 96, 108))
            (u, v), = tp(row[:2])
            if not (-10 < u < W + 10 and -10 < v < H + 10):
                continue
            if kind in ("car", "oncoming"):
                hc, hs = math.cos(row[4]), math.sin(row[4])
                L2, W2 = max(row[3], 3.5) / 2, max(row[2], 0.8)
                box = [(row[0] + a * hc - b * hs, row[1] + a * hs + b * hc) for a, b in ((-L2, -W2), (L2, -W2), (L2, W2), (-L2, W2))]
                d.polygon(tp(np.array(box)), fill=col + (255,))
            else:
                rr = max(3.0, row[2] * sc * 1.3)
                d.ellipse([u - rr, v - rr, u + rr, v + rr], fill=col + (255,))
        d.polygon([(ox, oy - 11), (ox - 7, oy + 7), (ox, oy + 3), (ox + 7, oy + 7)], fill=(255, 255, 255, 255) if self.dark else (30, 30, 34, 255))
        d.text((12, 8), "지도", font=font(13), fill=(230, 230, 235, 255) if self.dark else (40, 40, 45, 255))
        img.putalpha(Image.composite(img.getchannel("A"), Image.new("L", (W, H), 0), _rounded_mask((W, H), 18)))
        return img


class DriverMainUI:
    def __init__(self, env, kinds=None, main="driver", show=(), W=1280, H=720):
        if main not in MAINS:
            raise ValueError(f"main은 {MAINS} 중 하나")
        self.env, self.kinds, self.W, self.H = env, kinds, W, H
        self.main = main
        self.show = [s for s in show if s in EXTRAS]
        self.slide = {s: 1.0 for s in self.show}   # 0 = 숨김, 1 = 다 들어옴
        self.dark = env.rules.light < 0.5
        self._big, self._small = {}, {}
        self.cw, self.ch = 340, 191
        self.minimap = MiniMap(env, kinds, dark=self.dark)
        self.range_wh_km = 150.0
        self.batt_kwh = getattr(getattr(getattr(env, "car", None), "ev", None), "p", None)
        self.batt_kwh = self.batt_kwh.battery_kwh if self.batt_kwh is not None else 64.0

    # 패널은 처음 쓸 때 만든다(크기별로 따로)
    def _panel(self, name, big):
        cache = self._big if big else self._small
        if name not in cache:
            W, H = (self.W, self.H) if big else (self.cw, self.ch)
            env, kinds = self.env, self.kinds
            if name == "driver":
                cache[name] = DriverView(env, W, H, kinds=kinds, hud="minimal")
            elif name == "top":
                cache[name] = TopDownView(env, W, H, kinds=kinds, view_m=40.0)
            elif name == "motor":
                cache[name] = MotorBayView(env, W, H)
            elif name == "satellite":
                cache[name] = SatelliteView(env, W, H, kinds=kinds)
        return cache[name]

    # ---------- 꾸미기 도구 ----------
    @property
    def card_rgba(self):
        return (20, 21, 24, 170) if self.dark else (248, 248, 250, 200)

    @property
    def ink(self):
        return (245, 245, 247) if self.dark else (24, 24, 28)

    @property
    def muted(self):
        return (165, 168, 175) if self.dark else (105, 108, 115)

    def _card(self, lay, box, r=18, rgba=None):
        ImageDraw.Draw(lay).rounded_rectangle(box, radius=r, fill=rgba or self.card_rgba)

    def _paste_round(self, img, panel_img, xy, r=16, alpha=255):
        m = _rounded_mask(panel_img.size, r)
        if alpha < 255:
            m = m.point(lambda v: v * alpha // 255)
        img.paste(panel_img.convert("RGB"), xy, m)

    # ---------- HUD 조각 ----------
    def _speed_card(self, lay, f):
        d = ImageDraw.Draw(lay)
        x0, y0, w, h = 24, 24, 330, 206
        self._card(lay, [x0, y0, x0 + w, y0 + h])
        ink, muted = self.ink, self.muted
        # P R N D (전기차 1단 — D만 켜짐)
        for k, g in enumerate("PRND"):
            d.text((x0 + 22 + k * 22, y0 + 14), g, font=font(15), fill=ink if g == "D" else muted)
        kmh = f["v"] * 3.6
        lim = f["limit"] * 3.6
        over = kmh > lim + 1
        d.text((x0 + 20, y0 + 34), f"{kmh:.0f}", font=font(76), fill=RED if over else ink)
        d.text((x0 + 24, y0 + 118), "km/h", font=font(14), fill=muted)
        # 제한속도 표지
        lx, ly, R = x0 + w - 62, y0 + 70, 36
        d.ellipse([lx - R, ly - R, lx + R, ly + R], fill=(214, 36, 40))
        d.ellipse([lx - R * 0.76, ly - R * 0.76, lx + R * 0.76, ly + R * 0.76], fill=(252, 252, 252))
        d.text((lx, ly), f"{lim:.0f}", font=font(30), fill=(12, 12, 12), anchor="mm")
        if f["school"]:
            d.rounded_rectangle([lx - 46, ly + R + 6, lx + 46, ly + R + 26], radius=6, fill=(250, 205, 30))
            d.text((lx, ly + R + 16), "어린이보호구역", font=font(11), fill=(0, 0, 0), anchor="mm")
        # 배터리 SOC
        soc = float(np.clip(f.get("soc", 0.8), 0, 1))
        bx, by, bw, bh = x0 + 22, y0 + 146, 150, 18
        d.rounded_rectangle([bx, by, bx + bw, by + bh], radius=5, outline=muted, width=2)
        d.rectangle([bx + bw + 2, by + 5, bx + bw + 5, by + bh - 5], fill=muted)
        col = GREEN if soc > 0.2 else RED
        d.rounded_rectangle([bx + 3, by + 3, bx + 3 + (bw - 6) * soc, by + bh - 3], radius=3, fill=col)
        rng_km = soc * self.batt_kwh * 1e3 / self.range_wh_km
        d.text((bx + bw + 14, by - 1), f"{soc * 100:.0f}%  ·  {rng_km:.0f} km", font=font(15), fill=ink)
        # 전력 막대: 가운데 0, 왼쪽 회생(초록), 오른쪽 사용
        pk = f.get("p_motor_kw", 0.0)
        px0, py, pw = x0 + 22, y0 + 182, 180
        zx = px0 + pw * 60 / 210
        d.rounded_rectangle([px0, py, px0 + pw, py + 8], radius=4, fill=(70, 72, 78) if self.dark else (215, 217, 222))
        if pk >= 0:
            d.rounded_rectangle([zx, py, zx + pw * min(pk, 150) / 210, py + 8], radius=4, fill=ink)
        else:
            d.rounded_rectangle([zx - pw * min(-pk, 60) / 210, py, zx, py + 8], radius=4, fill=GREEN)
        d.line([(zx, py - 3), (zx, py + 11)], fill=muted, width=2)
        txt = f"회생 {pk:.0f} kW" if pk < -0.3 else f"{pk:.0f} kW"
        d.text((px0 + pw + 12, py - 6), txt, font=font(15), fill=GREEN if pk < -0.3 else ink)

    def _mode(self, lay, f):
        d = ImageDraw.Draw(lay)
        mode = f["mode"]
        W = self.W
        if mode in ("TAKEOVER", "SAFE_STOP"):
            blink = (f["t"] * 2) % 1 < 0.5
            col = RED if mode == "TAKEOVER" else (170, 20, 24)
            a = 235 if blink else 205
            d.rounded_rectangle([W / 2 - 270, 18, W / 2 + 270, 92], radius=18, fill=col + (a,))
            title = "개입 요청 · 핸들을 잡으세요" if mode == "TAKEOVER" else "안전 정지 중"
            d.text((W / 2, 44), title, font=font(26), fill=(255, 255, 255), anchor="mm")
            if f["risks"]:
                d.text((W / 2, 76), ", ".join(f["risks"])[:44], font=font(15), fill=(255, 230, 230), anchor="mm")
            return
        label, dot = ("자율 주행", TEAL) if mode == "AUTONOMOUS" else ("기준 제어기", (140, 145, 155))
        fw = font(17)
        tw = d.textlength(label, font=fw)
        w = tw + 52
        x0 = W / 2 - w / 2
        self._card(lay, [x0, 22, x0 + w, 58], r=18)
        d.ellipse([x0 + 16, 34, x0 + 28, 46], fill=dot)
        d.text((x0 + 36, 40), label, font=fw, fill=self.ink, anchor="lm")

    def _status(self, lay, f, y):
        parts = ["자갈길" if f["gravel"] else "포장도로"]
        if self.dark:
            parts.append("밤")
        parts.append(f"시야 {f['sight']:.0f} m")
        parts.append("TTC " + ("-" if f["ttc"] is None else f"{f['ttc']:.1f} s"))
        if f["slipping"]:
            parts.append("미끄러짐")
        txt = "  ·  ".join(parts)
        d = ImageDraw.Draw(lay)
        fw = font(14)
        w = d.textlength(txt, font=fw) + 32
        self._card(lay, [24, y, 24 + w, y + 32], r=16)
        d.text((40, y + 16), txt, font=fw, fill=RED if f["slipping"] else self.ink, anchor="lm")

    # ---------- 한 장 ----------
    def render(self, f) -> Image.Image:
        W, H = self.W, self.H
        img = self._panel(self.main, True).render(f).convert("RGB")
        if img.size != (W, H):
            img = img.resize((W, H))
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self._speed_card(lay, f)
        self._mode(lay, f)
        status_y = H - 56
        if self.main != "driver":
            status_y = H - 24 - self.ch - 46
        self._status(lay, f, status_y)
        img = Image.alpha_composite(img.convert("RGBA"), lay)
        # 오른쪽 아래: 미니맵(큰 화면이 지도면 운전석 카드)
        mw, mh = self.minimap.size
        if self.main == "top":
            small = self._panel("driver", False).render(f)
            self._paste_round(img, small, (W - 24 - self.cw, H - 24 - self.ch))
        else:
            mm = self.minimap.render(f)
            img.alpha_composite(mm, (W - 24 - mw, H - 24 - mh))
        if self.main not in ("driver", "top"):
            small = self._panel("driver", False).render(f)
            self._paste_round(img, small, (24, H - 24 - self.ch))
        # 필요할 때만: 모터룸·위성 작은 카드(오른쪽에서 밀려 들어옴)
        d = ImageDraw.Draw(img)
        y = 24
        for name in self.show:
            if name == self.main:
                continue
            p = float(np.clip(self.slide.get(name, 1.0), 0, 1))
            if p <= 0.01:
                continue
            e = 1 - (1 - p) ** 3
            x = int(W - 24 - self.cw + (1 - e) * (self.cw + 40))
            small = self._panel(name, False).render(f)
            self._paste_round(img, small, (x, y + 22), alpha=int(255 * e))
            cap = MOTOR_LABEL if name == "motor" else "위성 (키·모듈 없으면 지도로 대신)"
            d.text((x + 4, y + 2), cap, font=font(11), fill=self.ink, stroke_width=2, stroke_fill=(0, 0, 0) if self.dark else (255, 255, 255))
            y += self.ch + 30
        return img.convert("RGB")
