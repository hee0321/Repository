"""전기차(EV) 구동계 — 에너지 계산용 단순 모델.

시뮬레이터 차의 움직임(자전거 모델: 가속도 명령 → 속도)은 그대로 두고, 그 움직임을 만들려면 모터·배터리가
얼마나 일해야 하는지만 계산한다. 그래서 주행 결과·점수·한국 규칙·PPO 관측은 바뀌지 않는다.

- 즉각 토크: 단일 감속기(변속 없음, 공회전 RPM 없음), 기본 속도까지 토크 일정 → 그 위는 출력 한계.
  시뮬레이터 최대 가속 3 m/s²는 모터 최대 토크 안쪽이라, 0 km/h부터 바로 최대 가속이 나온다(EV 특성과 같음).
- 회생제동: 감속에 필요한 힘을 먼저 모터(발전기)로 받고(최대 회생 출력·감속 한계, 저속에서 줄어듦),
  모자라면 마찰 브레이크가 나머지를 맡는다(블렌딩). 회생 에너지는 효율을 곱해 배터리로 돌아간다.
- 배터리 SOC: 모터 출력/효율 + 보조 전력(에어컨·전장)만큼 줄고, 회생만큼 늘어난다.
수치는 중형 전기 SUV 정도의 그럴듯한 값이지 특정 차 모델이 아니다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

G = 9.81


@dataclass
class EVParams:
    mass: float = 1900.0            # kg (사람 포함)
    wheel_r: float = 0.33           # m
    reduction: float = 9.0          # 단일 감속비 (모터:바퀴)
    max_torque: float = 350.0       # N·m (모터)
    max_power_kw: float = 150.0     # 모터 최대 출력
    max_regen_kw: float = 60.0      # 최대 회생 출력
    max_regen_decel: float = 2.0    # m/s² — 이보다 센 감속은 마찰 브레이크가 보탬
    regen_fade_mps: float = 2.5     # 이 속도 아래에서 회생이 0까지 줄어듦(정지 직전은 마찰 브레이크)
    cda: float = 0.62               # 공기저항 Cd·A (m²)
    crr: float = 0.011              # 구름저항
    rho: float = 1.2
    eta_drive: float = 0.90         # 배터리→바퀴 효율
    eta_regen: float = 0.70         # 바퀴→배터리 효율
    aux_kw: float = 0.4             # 보조 전력
    battery_kwh: float = 64.0
    soc0: float = 0.80
    torque_tau: float = 0.03        # s — 토크 응답 시간(내연기관보다 훨씬 빠름)


class EVPowertrain:
    def __init__(self, params: EVParams | None = None, soc0: float | None = None):
        self.p = params or EVParams()
        self.soc = self.p.soc0 if soc0 is None else float(soc0)
        self.torque = 0.0           # 모터 토크(N·m, 음수 = 회생)
        self.motor_rpm = 0.0
        self.p_motor_kw = 0.0       # 모터 축 출력(음수 = 회생)
        self.p_bat_kw = 0.0         # 배터리 출력(+방전, -충전)
        self.regen_kw = 0.0         # 배터리로 돌아가는 전력(≥0)
        self.fric_kw = 0.0          # 마찰 브레이크가 열로 버리는 전력(≥0)
        self.fric_frac = 0.0        # 감속 중 마찰 브레이크 몫(0~1)
        self.e_used_wh = 0.0
        self.e_regen_wh = 0.0
        self.dist_m = 0.0

    def max_force(self, v):
        p = self.p
        f_t = p.max_torque * p.reduction / p.wheel_r
        f_p = p.max_power_kw * 1e3 / max(v, 0.1)
        return min(f_t, f_p)

    def step(self, v: float, a: float, dt: float) -> None:
        """v: 이번 스텝 평균 속도(m/s), a: 실제 가속도(m/s²)."""
        p = self.p
        v = max(0.0, float(v))
        resist = (0.5 * p.rho * p.cda * v * v + p.crr * p.mass * G) if v > 0.05 else 0.0
        need = p.mass * a + resist  # 바퀴에서 필요한 힘(+구동, -제동)
        if need >= 0:
            f_motor, f_fric = min(need, self.max_force(v)), 0.0
        else:
            fade = min(1.0, v / p.regen_fade_mps)
            f_lim = min(p.max_regen_decel * p.mass, p.max_regen_kw * 1e3 / max(v, 0.1)) * fade
            f_motor = -min(-need, f_lim)
            f_fric = -need + f_motor
        tq = f_motor * p.wheel_r / p.reduction
        self.torque += (tq - self.torque) * min(1.0, dt / p.torque_tau)
        self.motor_rpm = v / p.wheel_r * p.reduction * 60 / (2 * math.pi)
        self.p_motor_kw = f_motor * v / 1e3
        self.fric_kw = f_fric * v / 1e3
        self.fric_frac = 0.0 if need >= 0 else f_fric / max(-need, 1e-6)
        if self.p_motor_kw >= 0:
            self.regen_kw = 0.0
            self.p_bat_kw = self.p_motor_kw / p.eta_drive + p.aux_kw
        else:
            self.regen_kw = -self.p_motor_kw * p.eta_regen
            self.p_bat_kw = p.aux_kw - self.regen_kw
        wh = self.p_bat_kw * 1e3 * dt / 3600
        if wh >= 0:
            self.e_used_wh += wh
        else:
            self.e_regen_wh -= wh
        self.soc = min(1.0, max(0.0, self.soc - wh / (p.battery_kwh * 1e3)))
        self.dist_m += v * dt

    def snapshot(self) -> dict:
        return {"soc": self.soc, "torque": self.torque, "motor_rpm": self.motor_rpm, "p_motor_kw": self.p_motor_kw,
                "p_bat_kw": self.p_bat_kw, "regen_kw": self.regen_kw, "fric_kw": self.fric_kw,
                "fric_frac": self.fric_frac, "e_used_wh": self.e_used_wh, "e_regen_wh": self.e_regen_wh}

    def range_km(self, wh_per_km: float = 150.0) -> float:
        return self.soc * self.p.battery_kwh * 1e3 / wh_per_km
