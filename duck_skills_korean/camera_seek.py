"""머리 카메라 검출만으로 빨간 공을 찾고, 맞추고, 다가가 멈추는 규칙 제어.

보행 명령(vx, wz)과 고개 명령은 기존 v8 velstand 정책에 넣는다. 새 강화학습이 아니다.
이 파일은 시뮬레이터 공 좌표·세그멘테이션·거리·방위를 읽지 않는다.
검출 dict 의 bearing 은 픽셀과 카메라 자신의 방향으로 환산한 값이고, diam 은 픽셀 지름이다.
넘어짐 여부(upright)만 고유감각으로 받아, 넘어진 채 걷지 않게 한다.

고개 yaw 명령의 양수는 왼쪽이다 (2026-11-07 측정: 명령 +1.2 rad 일 때 카메라가 왼쪽을 봄).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from duck_sim import Cmd

YAW_LIM = 1.25
SWEEP_W = 0.90
TURN_W = 0.70
TURN_HOLD_S = 1.5
SWEEPS_BEFORE_TURN = 2
MAX_SECTORS = 4
# 픽셀 등가지름. 320×180·세로 45° 에서 지름 7 cm 공은 0.65 m 에서 약 26 px.
# 48 px 는 그 비례로 약 0.35 m. 제어기는 미터를 받지 않는다. 값은 영상을 보고 손으로 고름.
STOP_DIAM = 46
SLOW_DIAM = 32


def _clip(x, a, b=None):
    if b is None:
        a, b = -a, a
    return max(a, min(b, x))


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class SeekState:
    mode: str
    said: str
    detail: str


class CameraSeek:
    def __init__(self):
        self.mode = "search"
        self.head_yaw = 0.0
        self.head_pitch = 0.22
        self.sweep_dir = 1.0
        self.sweeps = 0
        self.sector = 0
        self.phase = "sweep"
        self.phase_t = 0.0
        self.lost_s = 0.0
        self.stable = 0
        self.last = None
        self.t = 0.0

    def update(self, det: dict | None, upright: bool, dt: float) -> Cmd:
        self.t += dt
        if not upright and self.t > 0.4:
            return Cmd(head=self._head())
        good = det is not None and det.get("area", 0) >= 18 and "bearing" in det
        if good:
            self.lost_s = 0.0
            self.stable += 1
            self.last = det
        else:
            self.lost_s += dt
            self.stable = 0
            det = None
        if self.mode == "search":
            return self._search(det, dt)
        if self.mode == "align":
            return self._align(det)
        if self.mode == "approach":
            return self._approach(det)
        if self.mode == "stop":
            return self._hold(det)
        return Cmd()

    def snapshot(self) -> SeekState:
        said, detail = {
            "search": ("빨간 공을 찾아요", "고개로 살피는 중 · 아직 화면 밖"),
            "align": ("빨간 공 발견", "화면 방향으로 몸을 돌림"),
            "approach": ("공을 보고 다가가요", "빨간 덩어리가 커지면 멈춤"),
            "stop": ("가까워 보여서 멈춤", "픽셀 크기로 판단 · 미터 거리는 제어에 안 씀"),
            "giveup": ("빨간 공이 안 보여요", "한 바퀴 살펴도 없음 · 멈춤"),
        }[self.mode]
        return SeekState(self.mode, said, detail)

    def _head(self) -> tuple:
        return (0.0, self.head_pitch, self.head_yaw, 0.0)

    def _aim(self, det: dict) -> None:
        # 공이 있는 쪽으로 고개를 보낸다. bearing 은 몸통 기준 왼쪽이 양수.
        self.head_yaw = _clip(0.45 * self.head_yaw + 0.55 * det["bearing"], YAW_LIM)
        ey = float(det["ey"])
        diam = float(det["diam"])
        self.head_pitch = _clip(0.18 + 0.30 * max(0.0, ey) + 0.010 * max(0.0, diam - 16), 0.12, 0.85)

    def _search(self, det, dt) -> Cmd:
        if self.stable >= 2:
            self.mode = "align"
            self._aim(det)
            return Cmd(head=self._head())
        self.head_pitch = 0.22
        if self.phase == "sweep":
            self.head_yaw += self.sweep_dir * SWEEP_W * dt
            if abs(self.head_yaw) >= YAW_LIM:
                self.head_yaw = _clip(self.head_yaw, YAW_LIM)
                self.sweep_dir *= -1.0
                self.sweeps += 1
                if self.sweeps >= SWEEPS_BEFORE_TURN:
                    self.phase = "turn"
                    self.phase_t = 0.0
                    self.sweeps = 0
                    self.sector += 1
            return Cmd(head=self._head())
        self.phase_t += dt
        if self.sector > MAX_SECTORS:
            self.mode = "giveup"
            self.head_yaw = 0.0
            return Cmd()
        if self.phase_t >= TURN_HOLD_S:
            self.phase = "sweep"
            self.phase_t = 0.0
            return Cmd(head=self._head())
        return Cmd(wz=TURN_W, head=self._head())

    def _align(self, det) -> Cmd:
        if det is None:
            if self.lost_s > 0.5:
                self.mode = "search"
                self.phase = "sweep"
                self.stable = 0
            return Cmd(head=self._head())
        self._aim(det)
        bearing = _wrap(det["bearing"])
        if abs(bearing) < 0.20:
            self.mode = "approach"
            return Cmd(head=self._head())
        wz = _clip(1.6 * bearing, 1.0)
        if abs(wz) < 0.35:
            wz = math.copysign(0.35, bearing)
        return Cmd(wz=wz, head=self._head())

    def _approach(self, det) -> Cmd:
        if det is None:
            if self.last is not None and self.last["diam"] >= SLOW_DIAM and self.last["ey"] > 0.0:
                self.mode = "stop"
                return self._hold(None)
            if self.lost_s > 0.6:
                self.mode = "search"
                self.phase = "sweep"
                self.stable = 0
            return Cmd(head=self._head())
        self._aim(det)
        bearing = _wrap(det["bearing"])
        diam = float(det["diam"])
        if diam >= STOP_DIAM and abs(bearing) < 0.50:
            self.mode = "stop"
            return self._hold(det)
        wz = 0.0
        if abs(bearing) > 0.10:
            wz = _clip(1.4 * bearing, 0.8)
            if abs(bearing) > 0.16 and abs(wz) < 0.32:
                wz = math.copysign(0.32, bearing)
        vx = 0.26 if abs(bearing) < 0.40 else 0.0
        if diam >= SLOW_DIAM:
            vx = min(vx, 0.14)
        return Cmd(vx=vx, wz=wz, head=self._head())

    def _hold(self, det) -> Cmd:
        if det is not None:
            self._aim(det)
        return Cmd(head=self._head())
