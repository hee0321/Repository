"""운동학적 자전거 모델 (kinematic bicycle model) + 간단한 노면 마찰 한계."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.81


@dataclass
class CarParams:
    wheelbase: float = 2.5      # 축간 거리 L (m)
    length: float = 4.0         # 그리기용 차 길이
    car_width: float = 1.8      # 차 폭 (충돌 판정용)
    max_steer: float = 0.5      # 최대 조향각 (rad, 약 29도)
    max_steer_rate: float = 1.5 # 조향 속도 제한 (rad/s)
    max_accel: float = 3.0      # 최대 가속 (m/s^2)
    max_brake: float = 6.0      # 최대 감속 (m/s^2)
    max_speed: float = 20.0     # m/s (72 km/h)
    mu: float = 0.9             # 노면 마찰계수 (마른 아스팔트 ~0.9, 젖은 노면 ~0.5)


class BicycleCar:
    """상태: x, y, yaw(ψ), v, steer(δ). 뒷바퀴 축 기준.

    x' = v cos ψ,  y' = v sin ψ,  ψ' = v / L · tan δ,  v' = a
    마찰 한계: 옆 가속도 v·ψ' 가 μg를 넘으면 회전율을 μg/v로 잘라서 '밀려 나가게'(언더스티어) 한다.
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, params: CarParams | None = None):
        self.p = params or CarParams()
        self.x, self.y, self.yaw, self.v = float(x), float(y), float(yaw), float(v)
        self.steer = 0.0
        self.slipping = False
        self.lat_acc = 0.0

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def step(self, steer_cmd: float, accel_cmd: float, dt: float) -> None:
        p = self.p
        target = float(np.clip(steer_cmd, -p.max_steer, p.max_steer))
        dmax = p.max_steer_rate * dt
        self.steer += float(np.clip(target - self.steer, -dmax, dmax))
        a = float(np.clip(accel_cmd, -p.max_brake, p.max_accel))
        yaw_rate = self.v / p.wheelbase * np.tan(self.steer)
        lim = p.mu * G / max(self.v, 0.1)
        self.slipping = abs(yaw_rate) > lim
        yaw_rate = float(np.clip(yaw_rate, -lim, lim))
        self.lat_acc = self.v * yaw_rate
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw = float(np.angle(np.exp(1j * (self.yaw + yaw_rate * dt))))
        self.v = float(np.clip(self.v + a * dt, 0.0, p.max_speed))
