"""모터룸(전기차) 시점 — 단순 모델 기반 시각화 (실제 차량 모델 아님).

값은 시뮬레이터 신호(속도·가속·브레이크·조향)와 autodrive/ev.py의 에너지 계산에서 온다:
- 모터: RPM = 속도 × 단일 감속비 9.0 / 바퀴 반지름, 토크(N·m)·출력(kW, 회생은 음수). 변속·공회전 없음.
  회전자 그림은 1/40 속도로 느리게 돌려 눈에 보이게 한다.
- 인버터: 배터리 직류 ↔ 모터 교류. 전력 흐름 점(주황 = 방전 배터리→모터, 초록 = 회생 모터→배터리).
- 배터리 팩(바닥): SOC %, 배터리 출력(+방전/-충전), 이번 주행 사용·회생 Wh.
- 회생제동 / 마찰 브레이크 블렌딩: 감속 중 마찰 브레이크 몫만큼 캘리퍼가 빨갛게.
- 감속기 기어(맞물려 도는 두 톱니), 스티어링 랙·앞바퀴 각도 = 실제 조향각, 자갈길 흔들림.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from autodrive.ev import EVParams

from .common import MODE_LABEL, MODE_RGB, font, mix, text_center

VIS_SLOW = 40.0
ORANGE = (255, 140, 40)
GREEN = (60, 220, 130)
LABEL = "모터룸(전기차) · 단순 모델 기반 시각화 (실제 차량 모델 아님)"


def _rot(cx, cy, w, h, ang):
    c, s = math.cos(ang), math.sin(ang)
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts]


def _gear(d, cx, cy, r, teeth, ang, col):
    pts = []
    for k in range(teeth * 2):
        a = ang + k * math.pi / teeth
        rr = r if k % 2 == 0 else r * 0.82
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    d.polygon(pts, fill=col, outline=(60, 62, 66))
    d.ellipse([cx - r * 0.25, cy - r * 0.25, cx + r * 0.25, cy + r * 0.25], fill=(60, 62, 66))


class MotorAnim:
    """그림용 상태(회전각·흐름 점·흔들림). 프레임 순서대로 step()."""

    def __init__(self, seed=0):
        self.rotor = 0.0
        self.flow = 0.0
        self.shake = np.zeros(2)
        self.rng = np.random.default_rng(seed)
        self.prev_t = None

    def step(self, f):
        dt = 0.04 if self.prev_t is None else max(1e-3, min(0.5, f["t"] - self.prev_t))
        self.prev_t = f["t"]
        rpm = f.get("motor_rpm", 0.0)
        self.rotor += rpm / 60 * 2 * math.pi * dt / VIS_SLOW
        p = f.get("p_bat_kw", 0.0)
        self.flow += np.clip(p / 40.0, -2.5, 2.5) * dt
        amp = (0.25 + 0.06 * f["v"]) * (3.0 if f["gravel"] else 0.4) + (2.5 if f["slipping"] else 0)
        self.shake = 0.7 * self.shake + 0.3 * self.rng.normal(0, amp, 2)
        return self


class MotorBayView:
    """전기차 모터룸. 내부는 640×360으로 그리고 다른 크기면 늘리거나 줄인다."""

    BW, BH = 640, 360

    def __init__(self, env, W=640, H=360, params: EVParams | None = None):
        self.W, self.H = W, H
        car = getattr(env, "car", None)
        ev = getattr(car, "ev", None)
        self.p = params or (ev.p if ev is not None else EVParams())
        self.anim = MotorAnim()

    def _gauge(self, d, cx, cy, R, val, vmin, vmax, label, txt, col=(255, 255, 255), zero=None):
        d.arc([cx - R, cy - R, cx + R, cy + R], 150, 390, fill=(90, 92, 98), width=4)
        def ang(v):
            return 150 + 240 * (min(max(v, vmin), vmax) - vmin) / (vmax - vmin)
        a0 = ang(0.0 if zero is None else zero) if vmin < 0 else 150
        a1 = ang(val)
        if a1 != a0:
            d.arc([cx - R, cy - R, cx + R, cy + R], min(a0, a1), max(a0, a1), fill=col, width=5)
        text_center(d, (cx, cy + 2), txt, 13, (245, 245, 245))
        text_center(d, (cx, cy + R + 6), label, 10, (170, 170, 175))

    def _flow(self, d, pts, phase, col, n=6):
        """pts 꺾은선을 따라 흐르는 점(phase 증가 = 앞으로)."""
        P = np.array(pts, float)
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        L = seg.sum()
        cum = np.r_[0, np.cumsum(seg)]
        for k in range(n):
            u = ((k / n + phase) % 1.0) * L
            i = min(int(np.searchsorted(cum, u, side="right") - 1), len(seg) - 1)
            w = (u - cum[i]) / max(seg[i], 1e-6)
            x, y = P[i] + (P[i + 1] - P[i]) * w
            d.ellipse([x - 3.5, y - 3.5, x + 3.5, y + 3.5], fill=col)

    def render(self, f) -> Image.Image:
        st = self.anim.step(f)
        W, H = self.BW, self.BH
        img = Image.new("RGB", (W, H), (24, 25, 28))
        d = ImageDraw.Draw(img)
        sx, sy = st.shake
        p_bat = f.get("p_bat_kw", 0.0)
        p_mot = f.get("p_motor_kw", 0.0)
        regen = f.get("regen_kw", 0.0)
        fric = f.get("fric_frac", 0.0) if f.get("acc", 0.0) < -0.05 else 0.0
        soc = f.get("soc", 0.8)
        tq = f.get("torque", 0.0)
        rpm = f.get("motor_rpm", 0.0)
        regen_on = regen > 0.3
        flow_col = GREEN if regen_on else ORANGE
        d.rounded_rectangle([12, 34, W - 12, H - 80], radius=18, fill=(38, 40, 44), outline=(78, 80, 86), width=3)
        d.text((22, 38), "▲ 차 앞", font=font(10), fill=(140, 140, 145))
        # 스티어링 랙 + 앞바퀴
        ry = 92
        rack = f["steer"] / 0.5 * 26
        d.rectangle([150, ry - 4, 490, ry + 4], fill=(110, 112, 118))
        d.rectangle([300 - rack - 28, ry - 7, 300 - rack + 28, ry + 7], fill=(170, 172, 178))
        d.text((156, ry - 22), f"스티어링 랙 {rack:+.0f}mm · 핸들 {math.degrees(f['steer'] * 15):+.0f}°", font=font(10), fill=(190, 190, 195))
        axle_y = 128
        cal = mix((90, 90, 95), (255, 70, 20), float(np.clip(fric * 1.2, 0, 1)))
        for side, wx in ((-1, 70), (1, 570)):
            wy = axle_y + st.shake[1] * (1.2 if side < 0 else -1.2)
            d.line([(150 if side < 0 else 490, ry), (wx, wy - 22)], fill=(140, 140, 145), width=3)
            d.polygon(_rot(wx, wy, 34, 72, -f["steer"]), fill=(16, 16, 18), outline=(70, 70, 75))
            d.polygon(_rot(wx, wy, 14, 48, -f["steer"]), fill=(120, 120, 125))
            cxp = wx - side * 2
            d.rounded_rectangle([cxp - 8, wy - 14, cxp + 8, wy + 6], radius=3, fill=cal)
            for k in range(4):
                yy = wy - 64 + k * 9 + st.shake[1] * k * 0.4
                d.line([(wx - 9, yy), (wx + 9, yy + 5)], fill=(200, 170, 60), width=2)
        # 모터(가운데, 앞차축 위) + 감속기 + 구동축
        mx, my = 260 + sx * 0.4, axle_y + sy * 0.4
        d.line([(110, axle_y), (mx - 50, my)], fill=(150, 150, 155), width=6)
        d.line([(mx + 140, my), (530, axle_y)], fill=(150, 150, 155), width=6)
        d.rounded_rectangle([mx - 52, my - 34, mx + 52, my + 34], radius=10, fill=(70, 96, 120), outline=(140, 170, 200), width=2)
        rc = (mx, my)
        d.ellipse([rc[0] - 26, rc[1] - 26, rc[0] + 26, rc[1] + 26], fill=(40, 52, 66), outline=(160, 190, 220), width=2)
        for k in range(3):
            a = st.rotor + k * 2 * math.pi / 3
            d.line([rc, (rc[0] + 24 * math.cos(a), rc[1] + 24 * math.sin(a))], fill=(220, 230, 240), width=3)
        coil = int(np.clip(abs(tq) / 200, 0, 1) * 180)
        d.rectangle([mx - 50, my - 32, mx - 34, my + 32], fill=(150 + coil // 2, 90 + coil // 3, 40) if not regen_on else (40, 150 + coil // 2, 90))
        d.text((mx - 48, my + 38), "구동 모터(영구자석 동기)", font=font(10), fill=(200, 210, 220))
        gx = mx + 90
        d.rounded_rectangle([gx - 36, my - 40, gx + 50, my + 40], radius=8, fill=(78, 80, 86), outline=(120, 122, 128), width=2)
        _gear(d, gx - 14, my - 12, 14, 8, st.rotor, (190, 192, 198))
        _gear(d, gx + 20, my + 10, 26, 16, -st.rotor * 8 / 16 - 0.2, (170, 172, 178))
        d.text((gx - 34, my - 56), f"감속기 1단 {self.p.reduction:.1f}:1", font=font(10), fill=(200, 200, 205))
        # 인버터
        ix, iy = 150, 196
        d.rounded_rectangle([ix - 50, iy - 22, ix + 50, iy + 22], radius=6, fill=(90, 70, 110), outline=(160, 130, 190), width=2)
        text_center(d, (ix, iy - 6), "인버터", 12, (240, 230, 250))
        text_center(d, (ix, iy + 10), "DC ↔ AC", 10, (200, 190, 215))
        # 배터리 팩(바닥)
        bx0, by0, bx1, by1 = 260, 176, 610, 270
        d.rounded_rectangle([bx0, by0, bx1, by1], radius=8, fill=(30, 40, 46), outline=(90, 120, 130), width=2)
        ncell = 12
        cw = (bx1 - bx0 - 20) / ncell
        lit = soc * ncell
        for k in range(ncell):
            x0 = bx0 + 10 + k * cw
            fill_c = mix((200, 60, 50), (60, 200, 120), float(np.clip(soc * 1.4 - 0.2, 0, 1)))
            c = fill_c if k + 1 <= lit else (mix((44, 54, 60), fill_c, float(np.clip(lit - k, 0, 1))))
            d.rounded_rectangle([x0 + 2, by0 + 10, x0 + cw - 2, by1 - 30], radius=3, fill=c)
        d.text((bx0 + 10, by1 - 26), f"배터리 {self.p.battery_kwh:.0f} kWh  SOC {soc * 100:.1f}%", font=font(12), fill=(230, 240, 235))
        bt = "충전 " if p_bat < 0 else "방전 "
        d.text((bx0 + 210, by1 - 26), f"{bt}{abs(p_bat):.1f} kW", font=font(12), fill=GREEN if p_bat < 0 else ORANGE)
        # 고전압 케이블 + 전력 흐름(배터리 → 인버터 → 모터, 회생이면 반대)
        cable = [(bx0, (by0 + by1) / 2 - 10), (ix + 50, iy), (ix, iy - 22), (mx - 52, my + 20)]
        d.line(cable, fill=(200, 110, 30), width=5, joint="curve")
        if abs(p_bat) > 0.6 or regen_on:
            self._flow(d, cable, st.flow, flow_col)
        if regen_on:
            d.rounded_rectangle([340, 140, 600, 168], radius=8, fill=(20, 80, 50))
            d.text((350, 145), f"회생제동 {regen:.1f} kW → 배터리", font=font(13), fill=(170, 255, 200))
        if fric > 0.05:
            d.text((24, 250), f"마찰 브레이크 {fric * 100:.0f}% (회생 한계 넘는 감속)", font=font(11), fill=cal)
        if f["gravel"]:
            d.text((24, 234), "자갈길 진동", font=font(11), fill=(255, 200, 80))
        # 아래 계기
        d.rectangle([0, H - 76, W, H], fill=(14, 14, 18))
        gy = H - 44
        self._gauge(d, 52, gy, 24, p_mot, -60, 150, "모터 kW", f"{p_mot:+.0f}", col=GREEN if p_mot < 0 else ORANGE)
        self._gauge(d, 132, gy, 24, tq, -200, 350, "토크 N·m", f"{tq:+.0f}", col=GREEN if tq < 0 else (120, 190, 255))
        self._gauge(d, 212, gy, 24, rpm, 0, 12000, "모터 RPM", f"{rpm / 1000:.1f}k", col=(200, 200, 210))
        self._gauge(d, 292, gy, 24, soc * 100, 0, 100, "SOC %", f"{soc * 100:.0f}", col=mix((220, 70, 50), (60, 210, 120), soc))
        acc = f.get("acc", 0.0)
        drive = "회생 감속" if regen_on else ("정지" if f["v"] < 0.2 else ("가속(즉각 토크)" if acc > 0.2 else "정속"))
        d.text((340, H - 70), f"D (1단·변속 없음)  {f['v'] * 3.6:.0f} km/h  {drive}", font=font(13), fill=(240, 240, 240))
        d.text((340, H - 50), f"가속 {max(acc, 0):.1f}  감속 {max(-acc, 0):.1f} m/s²  · 마찰 {fric * 100:.0f}%", font=font(12), fill=(220, 220, 225))
        d.text((340, H - 30), f"사용 {f.get('e_used_wh', 0):.0f} Wh · 회생 {f.get('e_regen_wh', 0):.0f} Wh · 회전자 1/40 속도", font=font(11), fill=(160, 160, 165))
        # 머리말
        mode = f["mode"]
        d.rectangle([0, 0, W, 30], fill=(10, 10, 12))
        d.text((10, 7), LABEL, font=font(13), fill=(120, 220, 255))
        d.rounded_rectangle([W - 120, 4, W - 6, 26], radius=6, fill=MODE_RGB.get(mode, (69, 123, 157)))
        text_center(d, (W - 63, 15), MODE_LABEL.get(mode, mode).split("  ")[0], 11, (255, 255, 255))
        if (self.W, self.H) != (W, H):
            img = img.resize((self.W, self.H), Image.LANCZOS)
        return img
