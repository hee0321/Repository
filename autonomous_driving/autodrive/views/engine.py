"""엔진룸 시점 — 단순 모델 기반 시각화 (실제 엔진 모델 아님).

시뮬레이터에는 엔진이 없다(차는 가속도 명령을 바로 받는 자전거 모델). 그래서 이 화면은 실제 시뮬레이터 신호로
움직이는 '그림'이다:
- RPM/기어: 속도 + 기어비(5단) + 최종 감속비 + 간단한 자동 변속(가속 페달이 클수록 늦게 올림) → 피스톤·크랭크 회전
  (화면에서는 1/40 속도로 느리게 돌려서 눈으로 보이게 함, 점화 순서 1-3-4-2)
- 스로틀 밸브 열림 = 가속 명령(0~1), 브레이크 캘리퍼 압력·온도 = 브레이크 명령과 속도
- 스티어링 랙 이동·앞바퀴 각도 = 실제 조향각
- 서스펜션 흔들림 = 자갈길 + 속도 (+ 미끄러짐)
- 냉각수 온도·팬 = 장난감 모델(부하가 크면 오르고, 달리는 바람과 팬이 식힘)
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .common import MODE_LABEL, MODE_RGB, font, mix, text_center

GEARS = [3.5, 2.1, 1.4, 1.0, 0.8]
FINAL = 4.0
WHEEL_R = 0.31
IDLE = 800.0
VIS_SLOW = 40.0  # 크랭크 애니메이션을 실제보다 40배 느리게


class EngineModel:
    """장난감 구동계·열 모델. 프레임 순서대로 step()을 부른다."""

    def __init__(self, seed=0):
        self.gear = 1
        self.rpm = IDLE
        self.crank = 0.0      # 화면용 크랭크 각 (rad)
        self.fan_ang = 0.0
        self.coolant = 86.0
        self.fan = 0.0
        self.brake_temp = 40.0
        self.shake = np.zeros(2)
        self.rng = np.random.default_rng(seed)
        self.prev_t = None

    def step(self, f):
        dt = 0.04 if self.prev_t is None else max(1e-3, f["t"] - self.prev_t)
        self.prev_t = f["t"]
        v = f["v"]
        thr = max(0.0, f["accel_cmd"])
        brk = max(0.0, -f["accel_cmd"])
        wheel_rpm = v / WHEEL_R * 60 / (2 * math.pi)
        up = 2300 + 2000 * thr
        down = 1200
        rpm = wheel_rpm * GEARS[self.gear - 1] * FINAL
        if rpm > up and self.gear < len(GEARS):
            self.gear += 1
        elif rpm < down and self.gear > 1:
            self.gear -= 1
        rpm = max(IDLE + 300 * thr if v < 1.0 else IDLE, wheel_rpm * GEARS[self.gear - 1] * FINAL)
        self.rpm += (rpm - self.rpm) * min(1.0, dt * 8)
        self.crank += self.rpm / 60 * 2 * math.pi * dt / VIS_SLOW
        heat = 0.15 + 1.6 * thr * self.rpm / 3000
        cool = (self.coolant - 75) * (0.004 + 0.0015 * v + 0.05 * self.fan)
        self.coolant += (heat - cool) * dt
        self.fan = float(np.clip((self.coolant - 91) / 5, 0, 1))
        self.fan_ang += (0.6 + 9 * self.fan) * dt * 6
        self.brake_temp += (brk * v * 12 - (self.brake_temp - 40) * (0.05 + 0.01 * v)) * dt
        amp = (0.25 + 0.06 * v) * (3.0 if f["gravel"] else 0.4) + (2.5 if f["slipping"] else 0)
        self.shake = 0.7 * self.shake + 0.3 * self.rng.normal(0, amp, 2)
        return {"rpm": self.rpm, "gear": self.gear, "thr": thr, "brk": brk, "coolant": self.coolant, "fan": self.fan,
                "brake_temp": self.brake_temp, "shake": self.shake.copy(), "crank": self.crank, "fan_ang": self.fan_ang}


def _rot(cx, cy, w, h, ang):
    c, s = math.cos(ang), math.sin(ang)
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts]


class EngineView:
    def __init__(self, env, W=640, H=360):
        self.W, self.H = W, H
        self.model = EngineModel()
        self.night = env.rules.light < 0.5

    def _gauge(self, d, cx, cy, R, val, vmax, label, unit, red=None, fmt="{:.0f}"):
        d.arc([cx - R, cy - R, cx + R, cy + R], 150, 390, fill=(190, 190, 190), width=3)
        if red is not None:
            a0 = 150 + 240 * red / vmax
            d.arc([cx - R, cy - R, cx + R, cy + R], a0, 390, fill=(220, 50, 40), width=4)
        a = math.radians(150 + 240 * min(max(val, 0), vmax) / vmax)
        d.line([(cx, cy), (cx + R * 0.88 * math.cos(a), cy + R * 0.88 * math.sin(a))], fill=(255, 140, 60), width=3)
        d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(200, 200, 200))
        text_center(d, (cx, cy + R * 0.5), fmt.format(val) + unit, 12, (240, 240, 240))
        text_center(d, (cx, cy + R + 8), label, 10, (180, 180, 180))

    def render(self, f) -> Image.Image:
        st = self.model.step(f)
        W, H = self.W, self.H
        img = Image.new("RGB", (W, H), (26, 27, 30))
        d = ImageDraw.Draw(img)
        sx, sy = st["shake"]
        # 엔진룸 테두리(위 = 차 앞쪽)
        d.rounded_rectangle([12, 34, W - 12, H - 78], radius=18, fill=(40, 42, 46), outline=(80, 82, 88), width=3)
        # 라디에이터 + 팬
        d.rectangle([150 + sx * 0.3, 42, 490 + sx * 0.3, 62], fill=(70, 74, 80))
        for k in range(160, 486, 8):
            d.line([(k + sx * 0.3, 44), (k + sx * 0.3, 60)], fill=(55, 58, 62))
        fx, fy, fr = 320 + sx * 0.3, 52, 9
        tc = mix((90, 150, 230), (230, 70, 40), np.clip((st["coolant"] - 80) / 25, 0, 1))
        d.text((498, 44), f"팬 {st['fan'] * 100:.0f}%", font=font(11), fill=(200, 200, 200))
        d.text((40, 44), f"냉각수 {st['coolant']:.0f}°C", font=font(11), fill=tc)
        for b in range(4):
            a = st["fan_ang"] + b * math.pi / 2
            d.line([(fx, fy), (fx + fr * math.cos(a), fy + fr * math.sin(a) * 0.5)], fill=(200, 200, 200), width=2)
        # 엔진 블록(직렬 4기통) — 흔들림 적용
        ex, ey = 175 + sx, 78 + sy
        d.rounded_rectangle([ex, ey, ex + 230, ey + 150], radius=10, fill=(95, 98, 104), outline=(130, 132, 138), width=2)
        d.text((ex + 8, ey + 4), "직렬 4기통 (단면)", font=font(10), fill=(220, 220, 220))
        cyc = st["crank"] % (4 * math.pi)  # 4행정 = 720°
        fire_off = [0.0, 3 * math.pi, math.pi, 2 * math.pi]  # 점화 순서 1-3-4-2 → 각 기통 점화 시점
        for i in range(4):
            cx0 = ex + 30 + i * 55
            d.rectangle([cx0 - 18, ey + 22, cx0 + 18, ey + 110], fill=(55, 57, 62))
            phase = st["crank"] + (0 if i in (0, 3) else math.pi)
            py = ey + 48 + 26 * (1 - math.cos(phase)) / 2 * 2 - 0
            # 점화 불꽃: 이 기통의 점화 시점 근처 + 가속 중일수록 밝게
            dphi = (cyc - fire_off[i]) % (4 * math.pi)
            if dphi < 0.7:
                glow = int(150 + 100 * st["thr"])
                d.rectangle([cx0 - 16, ey + 24, cx0 + 16, py - 2], fill=(glow, int(glow * 0.55), 30))
            d.rectangle([cx0 - 16, py, cx0 + 16, py + 16], fill=(185, 188, 194))
            crx = ex + 30 + i * 55 + 10 * math.sin(phase)
            cry = ey + 130 - 10 * math.cos(phase)
            d.line([(cx0, py + 16), (crx, cry)], fill=(160, 160, 165), width=4)
            d.ellipse([ex + 30 + i * 55 - 12, ey + 118, ex + 30 + i * 55 + 12, ey + 142], outline=(120, 120, 125), width=2)
        # 스로틀 밸브
        tx, ty = 535 + sx, 105 + sy
        d.ellipse([tx - 24, ty - 24, tx + 24, ty + 24], outline=(170, 170, 175), width=4, fill=(20, 20, 24))
        ang = math.radians(8 + 82 * st["thr"])
        d.line([(tx - 22 * math.cos(ang), ty - 22 * math.sin(ang) * 0.35), (tx + 22 * math.cos(ang), ty + 22 * math.sin(ang) * 0.35)],
               fill=(210, 210, 215), width=5)
        d.text((tx - 30, ty + 28), f"스로틀 {st['thr'] * 100:.0f}%", font=font(11), fill=(200, 200, 200))
        # 변속기
        d.rounded_rectangle([ex + 232, ey + 60, ex + 290, ey + 140], radius=8, fill=(80, 82, 88))
        text_center(d, (ex + 261, ey + 100), f"D{st['gear']}", 18, (255, 220, 120))
        # 스티어링 랙 + 앞바퀴(서스펜션 흔들림)
        rack = f["steer"] / 0.5 * 26
        ry = 252
        d.rectangle([150, ry - 4, 490, ry + 4], fill=(110, 112, 118))
        d.rectangle([300 - rack - 30, ry - 7, 300 - rack + 30, ry + 7], fill=(170, 172, 178))
        wheel_ang = -f["steer"]
        bt = st["brake_temp"]
        cal = mix((90, 90, 95), (255, 70, 20), float(np.clip(st["brk"] * 0.7 + (bt - 60) / 300, 0, 1)))
        for side, wx in ((-1, 92), (1, 548)):
            wy = ry + st["shake"][1] * (1.2 if side < 0 else -1.2)
            d.line([(150 if side < 0 else 490, ry), (wx, wy)], fill=(140, 140, 145), width=4)
            d.polygon(_rot(wx, wy, 34, 70, wheel_ang), fill=(18, 18, 20), outline=(70, 70, 75))
            d.polygon(_rot(wx, wy, 14, 46, wheel_ang), fill=(120, 120, 125))
            cxp = wx - side * 2
            d.rounded_rectangle([cxp - 8, wy - 14, cxp + 8, wy + 6], radius=3, fill=cal)
            # 서스펜션 스프링
            for k in range(5):
                yy = 140 + k * 12 + st["shake"][1] * k * 0.4
                d.line([(wx - 10, yy), (wx + 10, yy + 6)], fill=(200, 170, 60), width=2)
        d.text((60, ry + 42), f"브레이크 {st['brk'] * 100:.0f}% · 디스크 {bt:.0f}°C", font=font(11), fill=cal)
        d.text((380, ry + 42), f"핸들 {math.degrees(f['steer'] * 15):+.0f}° · 랙 {rack:+.0f}mm", font=font(11), fill=(200, 200, 200))
        if f["gravel"]:
            d.text((40, 200), "자갈길 진동", font=font(12), fill=(255, 200, 80))
        # 아래 계기
        d.rectangle([0, H - 74, W, H], fill=(16, 16, 20))
        self._gauge(d, 60, H - 44, 22, st["rpm"], 7000, "RPM", "", red=5500)
        self._gauge(d, 160, H - 44, 22, f["v"] * 3.6, 120, "속도", "")
        self._gauge(d, 260, H - 44, 22, st["coolant"], 130, "냉각수", "°", red=105)
        d.text((310, H - 66), f"기어 D{st['gear']}   RPM {st['rpm']:.0f}", font=font(13), fill=(240, 240, 240))
        d.text((310, H - 46), f"가속 {st['thr'] * 100:3.0f}%  브레이크 {st['brk'] * 100:3.0f}%", font=font(13), fill=(240, 240, 240))
        d.text((310, H - 26), "크랭크 그림은 1/40 속도로 느리게", font=font(11), fill=(160, 160, 160))
        # 머리말 + 라벨
        mode = f["mode"]
        d.rectangle([0, 0, W, 30], fill=(10, 10, 12))
        d.text((10, 7), "엔진룸 · 단순 모델 기반 시각화 (실제 엔진 모델 아님)", font=font(13), fill=(255, 210, 90))
        d.rounded_rectangle([W - 170, 4, W - 6, 26], radius=6, fill=MODE_RGB.get(mode, (69, 123, 157)))
        text_center(d, (W - 88, 15), MODE_LABEL.get(mode, mode).split("  ")[0], 12, (255, 255, 255))
        return img
