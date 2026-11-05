"""DCAS 01 시나리오 모음 — UN Regulation No. 171 01 series (DCAS, 2025년 9월 시행) 시험을 2D 시뮬에 맞춘 것.

출처: UN R171 01 series 정본 (EU 관보 OJ L 2025/1899, 26.9.2025, ELI http://data.europa.eu/eli/reg/2025/1899/oj).
EUR-Lex/UNECE 사이트가 자동 접근을 막아 같은 관보의 스페인어 정본(BOE 사본)을 읽었고, 문단 번호는 언어와 무관하게 같다.

차이(정직하게):
  - R171은 '운전자 보조(L2)'다. 이 저장소의 목표는 그보다 높다: 운전자 개입(takeover) 없이 끝까지 가기
    → 모든 시나리오에 '자율 완주'(autonomous completion)를 따로 센다 (출발→도착, 개입 0, 충돌·이탈 0).
  - 한국 맥락: 우측 통행, 같은 방향 여러 차로(1차로 = 가장 왼쪽), 제한속도 50(도심 안전속도 5030)·70(도시고속/자동차전용).
    차 최고속도가 72 km/h라 120 km/h 끼어들기(Type 2)는 같은 상대속도(50 km/h)로 70/20 km/h 로 줄였다.
  - 방향지시등: 정책 출력이 [조향, 가속] 두 개뿐이라, '경로 계획(내비)이 요청한 차로 변경'은 환경이 지시등을 켜고
    (= 시스템이 켬), 전문가(교사)는 장애물 회피용 차로 변경 때 스스로 켠다. 학습 모델이 요청 없이 차로를 바꾸면
    '지시등 없는 차로 변경'으로 6.2.7 불합격.
  - 판정에 쓰는 측정값은 시뮬 진실값(차 횡가속도 = v·요레이트, 뒤차 필요 감속도 계산)이다.
"""
from __future__ import annotations

import numpy as np

from .car import CarParams
from .env import CAR_HALF_LEN, AutoDriveEnv
from .rules import KMH, TrafficRules
from .track import Track

LW = 3.5                      # 차로 폭 (R171 Annex 4 4.2.5.1.3: 차로 변경 = 횡 3.5 m)
SHOULDER = 0.5
LAT_MAX = 3.0                 # 선언 최대 횡가속도 (5.3.7.1.2: M1 ≤ 3 m/s²)
SUITE = "dcas01"


# ---------------------------------------------------------------- 도로
def _clothoid_profile(parts, ds=1.0):
    """parts: [('straight', L) | ('arc', L, R(+왼쪽)) | ('clothoid', L, R0, R1)] → 곡률 배열 (1 m 간격)."""
    k = []
    for p in parts:
        n = max(1, int(round(p[1] / ds)))
        if p[0] == "straight":
            k += [0.0] * n
        elif p[0] == "arc":
            k += [1.0 / p[2]] * n
        else:
            k0 = 0.0 if p[2] in (None, 0) else 1.0 / p[2]
            k1 = 0.0 if p[3] in (None, 0) else 1.0 / p[3]
            k += list(np.linspace(k0, k1, n))
    return np.array(k)


def build_track(parts, n_lanes=2, name="dcas", twoway=False):
    kap = _clothoid_profile(parts)
    h = np.concatenate([[0.0], np.cumsum(kap)])
    P = np.column_stack([np.concatenate([[0.0], np.cumsum(np.cos(h[:-1]))]),
                         np.concatenate([[0.0], np.cumsum(np.sin(h[:-1]))])])
    width = n_lanes * LW + 2 * SHOULDER
    return Track(P, width=width, closed=False, name=name)


def lane_centres(n_lanes):
    """같은 방향 차로 가운데 (오른쪽 → 왼쪽). 2차로: (-1.75, +1.75), 3차로: (-3.5, 0, +3.5)."""
    return tuple(float((i - (n_lanes - 1) / 2) * LW) for i in range(n_lanes))


def make_rules(length, n_lanes, limit_kmh, ego_lane=0, twoway=False, lc_requests=(), school=None):
    n = int(np.ceil(length)) + 1
    lanes = lane_centres(n_lanes)
    lim = np.full(n, limit_kmh * KMH)
    sch = np.zeros(n, bool)
    if school:
        a, b = school
        sch[int(a):int(b)] = True
        lim[int(a):int(b)] = 30 * KMH
    return TrafficRules(1.0, lim, sch, np.full(n, twoway), np.full(n, lanes[ego_lane]), [], False, length,
                        lanes=None if twoway else lanes, lane_w=LW, lc_requests=list(lc_requests), lat_acc_max=LAT_MAX)


# 표준 도로 모양
STRAIGHT = [("straight", 700)]
# 4.2.4.1 S자 굽이 (클로소이드 파라미터 A, 반지름 R, 길이 L 표 그대로: A²=R·L)
S_CURVE = [("straight", 200), ("clothoid", 30.0, None, 787), ("arc", 57.1, 787), ("clothoid", 14.0, 787, None),
           ("clothoid", 26.0, None, -374), ("arc", 5.1, -374), ("clothoid", 39.0, -374, None), ("straight", 300)]


def curve(R, arc=200, lead=200, tail=250):
    """일정 반지름 굽이 (진입·진출 클로소이드 40 m)."""
    return [("straight", lead), ("clothoid", 40, None, R), ("arc", arc, R), ("clothoid", 40, R, None), ("straight", tail)]


def _car(s, d, v, **kw):
    return dict(s=s, d=d, v=v, r=0.9, l=4.5, kind="car", **kw)


def _cutin_events(trigger_ttc, d_to, vd):
    return [dict(when="ttc", at=trigger_ttc, d_to=d_to, vd=vd)]


R, L = lane_centres(2)            # 2차로: 오른쪽(바깥) -1.75, 왼쪽 +1.75
R3, M3, L3 = lane_centres(3)

# ---------------------------------------------------------------- 시나리오
# 각 항목: ref(문단), desc, road, lanes, ego_lane, limit, v0(km/h), obstacles, lc(차로 변경 요청), need_lc(목표 차로),
#          ref_speed(효율 기준 km/h), checks(추가 판정)
DCAS01 = {
    # --- 차로 유지 (Annex 4 4.2.5.1.1) ---
    "dcas_lk_curve_85": dict(
        ref="Annex4 4.2.5.1.1.1 / 5.3.7.1.1, 5.3.7.1.2", road=curve(150), lanes=2, ego_lane=0, limit=70, v0=70,
        desc="굽이 R150, 70 km/h → 필요 횡가속도 2.52 m/s² (선언 최대 3.0의 84%, 시험 80~90%). 차로 안에 머물기"),
    "dcas_lk_curve_over": dict(
        ref="Annex4 4.2.5.1.1.2 / 5.3.7.1.1.1 (굽이에 맞춰 감속)", road=curve(90), lanes=2, ego_lane=0, limit=70, v0=70,
        desc="굽이 R90을 70 km/h 그대로면 4.2 m/s² (최대 초과) → 미리 줄여 ≤3.3 m/s², 차로 이탈 없음"),
    "dcas_lk_scurve": dict(
        ref="Annex4 4.2.4.1 (S자 클로소이드 표) / 4.2.5.1.1", road=S_CURVE, lanes=2, ego_lane=0, limit=70, v0=70,
        desc="규정의 S자 굽이(R787 → 반대 R374) 70 km/h 차로 유지"),
    # --- 시스템이 시작하는 차로 변경 (4.2.5.1.3, 6.2) ---
    "dcas_lc_request": dict(
        ref="Annex4 4.2.5.1.3.1 / 6.2.3, 6.2.6, 6.2.7, 6.2.9.5", road=STRAIGHT, lanes=2, ego_lane=0, limit=70, v0=60,
        lc=[dict(s=150, to=L, by=450, reason="경로: 왼쪽 차로로")], need_lc=L,
        desc="경로 계획이 왼쪽 차로 요청 → 지시등 ≥3 s 뒤 MCC 시작(≤7 s), 횡 3.5 m, 횡가속도·저크 한도"),
    "dcas_lc_rear_approach": dict(
        ref="Annex4 4.2.5.1.3.2.1 b) / 6.2.4.1, 6.2.9.1.2", road=STRAIGHT, lanes=2, ego_lane=0, limit=70, v0=60,
        lc=[dict(s=150, to=L, by=650, reason="경로: 왼쪽 차로로")], need_lc=L,
        obstacles=[_car(-60, L, 77 * KMH)],
        desc="목표 차로 뒤에서 제한+10%(77 km/h)로 다가오는 차 → 그 차가 3 m/s² 넘게 감속하거나 1초 거리보다 가까워지면 불합격"),
    "dcas_lc_beside": dict(
        ref="Annex4 4.2.5.1.3.2.1 c) / 6.2.4", road=STRAIGHT, lanes=2, ego_lane=0, limit=50, v0=50,
        lc=[dict(s=120, to=L, by=600, reason="경로: 왼쪽 차로로")], need_lc=L,
        obstacles=[_car(2, L, 50 * KMH)],
        desc="목표 차로 바로 옆에 같은 속도로 달리는 차 → 기다렸다가(뒤로 빠져) 안전할 때 변경"),
    "dcas_lc_same_space": dict(
        ref="Annex4 4.2.5.1.3.2.1 d) / 6.2.4", road=STRAIGHT, lanes=3, ego_lane=0, limit=60, v0=55,
        lc=[dict(s=150, to=M3, by=600, reason="경로: 2차로로")], need_lc=M3,
        obstacles=[_car(8, L3, 55 * KMH, events=[dict(when="t", at=11.5, d_to=M3, vd=1.0)])],
        desc="3차로: 내가 가운데 차로로 가려는데 1차로 차도 같은 자리로 들어옴 → 양보 후 변경"),
    "dcas_lc_oncoming_refuse": dict(
        ref="6.2.2 (반대 방향 차로로 변경 금지) / Annex4 4.2.5.1.3.2.1 a)", road=STRAIGHT, lanes=2, ego_lane=0,
        limit=60, v0=40, twoway=True,
        obstacles=[_car(60, R, 30 * KMH), _car(400, L, -50 * KMH), _car(700, L, -50 * KMH)],
        desc="편도 1차로(중앙선): 앞에 느린 차 → 중앙선 넘어 앞지르지 않고 따라가기(마주 오는 차 있음)"),
    "dcas_lane_end_merge": dict(
        ref="6.1.2 (차로 합류·감소) / 6.2.4.1", road=STRAIGHT, lanes=2, ego_lane=0, limit=50, v0=50,
        lc=[dict(s=200, to=L, by=470, reason="차로 감소: 오른쪽 차로 끝")], need_lc=L,
        obstacles=[dict(s=480 + 6 * i, d=R, v=0.0, r=1.0, l=2.0, kind="object") for i in range(4)] +
                  [_car(-30 + 45 * i, L, 50 * KMH) for i in range(8)],
        desc="오른쪽 차로가 480 m에서 끝남(공사 방호벽) + 왼쪽 차로 50 km/h 차량 흐름(45 m 간격) → 틈에 합류"),
    # --- 앞 장애물·앞차 (4.2.5.2) ---
    "dcas_stationary_70": dict(
        ref="Annex4 4.2.5.2.1 / 5.3.2.2", road=STRAIGHT, lanes=2, ego_lane=0, limit=70, v0=70,
        obstacles=[_car(320, R + 0.4, 0.0)],
        desc="직선 70 km/h, 내 차로에 선 차(횡 0.4 m 치우침) → 충돌 없이 멈추거나 안전하게 차로 변경해 지나가기"),
    "dcas_stationary_curve": dict(
        ref="Annex4 4.2.5.2.2 (4.2.4.1 굽이, 차로 가운데서 0.5 m)", road=S_CURVE, lanes=2, ego_lane=0, limit=70, v0=70,
        obstacles=[_car(330, R + 0.5, 0.0)],
        desc="S자 굽이 둘째 굽이 쪽에 선 차 → 충돌 없음"),
    "dcas_slower_lead": dict(
        ref="Annex4 4.2.5.2.3 (50 km/h 느린 차) / 5.3.7.2.5.2", road=STRAIGHT, lanes=1, ego_lane=0, limit=70, v0=70,
        obstacles=[_car(120, 0.0, 20 * KMH)],
        desc="편도 1차로, 70 km/h로 20 km/h 앞차에 접근 → 급제동 없이 따라가기"),
    "dcas_lead_brake_4": dict(
        ref="Annex4 4.2.5.2.4 (50 km/h, 앞차 4 m/s² 감속)", road=STRAIGHT, lanes=1, ego_lane=0, limit=50, v0=50,
        obstacles=[_car(30, 0.0, 50 * KMH, events=[dict(when="t", at=8.0, v_to=0.0, decel=4.0),
                                                       dict(when="t", at=16.0, v_to=50 * KMH, accel=2.0)])],
        desc="안정된 차간으로 따라가다 앞차가 4 m/s²로 정지 → 충돌 없이 정지, 앞차 출발하면 따라 출발"),
    "dcas_lead_brake_6": dict(
        ref="Annex4 4.2.5.2.4.2 확장(더 큰 감속·다른 속도)", road=STRAIGHT, lanes=1, ego_lane=0, limit=60, v0=60,
        obstacles=[_car(35, 0.0, 60 * KMH, events=[dict(when="t", at=8.0, v_to=0.0, decel=6.0),
                                                       dict(when="t", at=16.0, v_to=60 * KMH, accel=2.0)])],
        desc="60 km/h에서 앞차가 6 m/s²로 급정지 (내 최대 제동과 같음)"),
    "dcas_cut_out": dict(
        ref="Annex4 4.2.5.2.5 (앞차 빠짐: 70/50 km/h, TTC 3 s, 1.5 m/s², 44 m)", road=STRAIGHT, lanes=2, ego_lane=0,
        limit=70, v0=50,
        obstacles=[_car(40, R, 50 * KMH, events=[dict(when="gap", at=-1e9)]),
                   _car(400, R, 0.0)],
        desc="50 km/h 앞차가 TTC 3 s에 정지 차를 피해 옆 차로로 빠짐 → 드러난 정지 차에 충돌 없음"),
    "dcas_cut_in_1": dict(
        ref="Annex4 4.2.5.2.6 Type 1 (50/10 km/h, 0.5 m/s², 14 m, 완료 TTC 0 s)", road=STRAIGHT, lanes=2, ego_lane=0,
        limit=50, v0=50,
        obstacles=[_car(140, L, 10 * KMH, events=_cutin_events(5.0, R, 0.69))],
        desc="옆 차로 10 km/h 차가 14 m 동안 내 차로로 끼어듦 (그대로 가면 끝날 때 TTC 0) → 충돌 없음"),
    "dcas_cut_in_2": dict(
        ref="Annex4 4.2.5.2.6 Type 2 (120/70 → 같은 상대속도 70/20 km/h, 1.5 m/s², 완료 TTC 1.5 s)", road=STRAIGHT,
        lanes=2, ego_lane=0, limit=70, v0=70,
        obstacles=[_car(200, L, 20 * KMH, events=_cutin_events(4.6, R, 1.13))],
        desc="빠른 접근 중 앞 느린 차가 3.1 s 만에 끼어듦 → 충돌 없음"),
    # --- 보행자·자전거 (4.2.5.2.7, .9, .10) ---
    "dcas_ped_standing": dict(
        ref="Annex4 4.2.5.2.7 (차로에 선 보행자)", road=STRAIGHT, lanes=2, ego_lane=0, limit=50, v0=50,
        obstacles=[dict(s=300, d=R, v=0.0, r=0.3, l=0.6, kind="ped")],
        desc="내 차로 가운데 서 있는 보행자 → 멈추거나 지시등 켜고 옆 차로로 넉넉히 비켜 서행"),
    "dcas_ped_crossing": dict(
        ref="Annex4 4.2.5.2.9 (5 km/h 직각 횡단, TTC ≥ 4 s, 충돌점 = 차 중심선)", road=STRAIGHT, lanes=2, ego_lane=0,
        limit=50, v0=50,
        obstacles=[dict(s=300, d=R - 4.0 * 5 * KMH, v=0.0, r=0.3, l=0.6, kind="ped",
                        events=[dict(when="ttc", at=4.0, d_to=L + 3.0, vd=5 * KMH)])],
        desc="오른쪽 보도에서 보행자가 5 km/h로 건너옴 (그대로 가면 4 s 뒤 차 앞 가운데) → 충돌 없음"),
    "dcas_bike_crossing": dict(
        ref="Annex4 4.2.5.2.10 (자전거 15 km/h 횡단)", road=STRAIGHT, lanes=2, ego_lane=0, limit=50, v0=50,
        obstacles=[dict(s=300, d=R - 4.0 * 15 * KMH, v=0.0, r=0.9, l=0.7, kind="ped", label="bicycle",
                        events=[dict(when="ttc", at=4.0, d_to=L + 4.0, vd=15 * KMH)])],
        desc="오른쪽에서 자전거가 15 km/h로 건너옴 → 충돌 없음"),
}

# 앞차 빠짐(cut-out): 앞차가 정지 차와 TTC 3 s(= 3 s × 50 km/h ≈ 41.7 m)에서 옆 차로로 44 m 동안 이동
_CO = DCAS01["dcas_cut_out"]["obstacles"]
_CO[0]["events"] = [dict(when="t", at=1e9)]  # 아래 make_env 에서 정지 차 기준 위치 트리거로 바꿈
_CO[0]["cutout_target"] = 1                    # 정지 차 번호
_CO[0]["cutout"] = dict(ttc=3.0, d_to=L, length=44.0)

DCAS_NAMES = list(DCAS01)
REF_SPEED = {k: v.get("ref_speed", v["limit"] * 0.8) for k, v in DCAS01.items()}
REF_SPEED.update(dcas_lead_brake_4=30, dcas_lead_brake_6=35, dcas_slower_lead=22, dcas_cut_out=25,
                 dcas_lane_end_merge=40, dcas_lc_oncoming_refuse=28, dcas_stationary_70=40, dcas_ped_standing=30,
                 dcas_stationary_curve=40, dcas_ped_crossing=35, dcas_bike_crossing=35, dcas_cut_in_1=25, dcas_cut_in_2=30)


def make_env(name_or_spec, seed=0, **kw) -> AutoDriveEnv:
    sc = DCAS01[name_or_spec] if isinstance(name_or_spec, str) else name_or_spec
    name = name_or_spec if isinstance(name_or_spec, str) else sc.get("name", "dcas_custom")
    twoway = sc.get("twoway", False)
    n = sc["lanes"]
    track = build_track(sc["road"], n_lanes=n, name=name)
    rules = make_rules(track.length, n, sc["limit"], sc.get("ego_lane", 0), twoway, sc.get("lc", ()))
    if twoway:   # 편도 1차로 양방향: 차로 = 오른쪽 절반
        rules.lane_ref[:] = R
    obs = []
    for o in sc.get("obstacles", []):
        o = {k: (list(v) if isinstance(v, list) else v) for k, v in o.items()}
        o["events"] = [dict(e) for e in o.get("events", [])]
        if "cutout" in o:
            co, tgt = o.pop("cutout"), sc["obstacles"][o.pop("cutout_target")]
            # 앞차와 정지 차 사이 거리가 TTC·앞차 속도가 되는 순간 → 내 차와의 간격으로 미리 계산할 수 없어
            # 앞차 기준 '도착 시각'으로 바꿈: 앞차가 (정지 차 - ttc·v) 지점에 닿는 시각
            t_go = (tgt["s"] - o["s"] - co["ttc"] * o["v"] - (o["l"] + tgt["l"]) / 2) / o["v"]
            o["events"] = [dict(when="t", at=t_go, d_to=co["d_to"], vd=abs(co["d_to"] - o["d"]) / (co["length"] / o["v"]))]
        obs.append(o)
    params = CarParams()
    v_lim = rules.limit_at(0)
    max_steps = kw.pop("max_steps", int(track.length / 4.0 / 0.05) + 600)
    env = AutoDriveEnv(track=track, rules=rules, params=params, seed=seed, max_steps=max_steps, start_s=0.0,
                       name=name, obstacles=obs, **kw)
    env.init_speed = min(sc.get("v0", 0) * KMH, v_lim)
    env.dcas = sc
    return env


# ---------------------------------------------------------------- 판정
class DcasMonitor:
    """매 스텝 차 상태를 보고 R171 판정값을 모은다 (env.monitor 로 붙임)."""

    def __init__(self):
        self.t, self.alat, self.dec = [], [], []
        self.max_rear_req = 0.0
        self.rear_gap_short = 0
        self.lcp_decel_max = 0.0
        self.emergency = False
        self.min_thw = np.inf
        self.max_brake = 0.0
        self._pv = None

    def _imminent(self, env):
        lo = env.lead_obstacle()
        if lo is None:
            return False
        g = lo[0] - env.ob_l[lo[4]] / 2 - CAR_HALF_LEN
        cl = env.car.v - max(lo[1], 0)
        return cl > 0 and cl * cl / (2 * max(g, 0.1)) > 5.0 * 0.6   # 5 m/s² 근처 → 회피 제동 허용 (2.16)

    def update(self, env):
        c = env.car
        self.t.append(env.t)
        self.alat.append(float(c.lat_acc))
        dv = 0.0 if self._pv is None else (c.v - self._pv) / env.dt
        self._pv = c.v
        self.dec.append(dv)
        self.max_brake = max(self.max_brake, -dv)
        imm = self._imminent(env)
        self.emergency |= imm
        # 6.2.4.3: PCC 동안(지시등 켠 뒤 ~ MCC 끝) 감속 ≤ 2 m/s² (충돌 임박 회피 제외)
        if env.indicator and not imm:
            self.lcp_decel_max = max(self.lcp_decel_max, -dv)
        # 6.2.4.1: MCC 시작 뒤 ~ 끝+2 s, 내 차로 뒤에서 오는 차가 1초 거리 유지하려면 필요한 감속도
        evs = env.lc_events
        active = env._lcm_on or (evs and "t_end" in evs[-1] and env.t - evs[-1]["t_end"] < 2.0)
        lo = env.lead_obstacle()
        if lo is not None and lo[1] > 1.0:
            g = lo[0] - env.ob_l[lo[4]] / 2 - CAR_HALF_LEN
            if c.v > 5:
                self.min_thw = min(self.min_thw, g / c.v)
        if active:
            for k in range(len(env.ob_s)):
                if env.ob_kind[k] not in ("car", "entering"):
                    continue
                if abs(env.ob_d[k] - env._d) > env.ob_r[k] + env.params.car_width / 2 + 0.3:
                    continue
                g = -env._gap(env.ob_s[k]) - env.ob_l[k] / 2 - CAR_HALF_LEN
                if g <= -1.0 or g > 120:
                    continue
                vr, ve = env.ob_v[k], c.v
                if vr > ve:
                    need = (vr - ve) ** 2 / (2 * max(g - ve * 1.0, 0.1))
                    self.max_rear_req = max(self.max_rear_req, float(need))
                    if g < ve * 1.0:
                        self.rear_gap_short += 1

    def results(self, env, res, sc):
        dt = env.dt
        a = np.array(self.alat) if self.alat else np.zeros(1)
        n5 = max(1, int(round(0.5 / dt)))
        ma = np.convolve(a, np.ones(n5) / n5, mode="same")
        jerk = np.abs(np.diff(ma)) / dt if len(ma) > 1 else np.zeros(1)
        jerk_ma = np.convolve(jerk, np.ones(n5) / n5, mode="valid") if len(jerk) >= n5 else jerk
        t = np.array(self.t) if self.t else np.zeros(1)
        lc_mask = np.zeros(len(a), bool)
        for e in env.lc_events:
            t1 = e.get("t_end", t[-1])
            lc_mask |= (t >= e["t"] - 2.0) & (t <= t1 + 1.0)
        a_lk = float(np.max(np.abs(ma[~lc_mask]))) if (~lc_mask).any() else 0.0
        a_lc = float(np.max(np.abs(ma[lc_mask]))) if lc_mask.any() else 0.0
        ev = env.lc_events
        done = [e for e in ev if e.get("done")]
        unsig = [e for e in ev if e["lcp_lead_s"] is None or e["lcp_lead_s"] < 3.0 or not e["ind_whole"]]
        m = {
            "max_lat_acc": round(a_lk, 2), "max_lat_acc_lc": round(a_lc, 2),
            "max_lat_jerk": round(float(np.max(jerk_ma)) if len(jerk_ma) else 0.0, 2),
            "max_brake": round(self.max_brake, 2), "emergency_brake": bool(self.emergency),
            "lane_changes": len(done), "lc_unsignalled": len(unsig),
            "lcp_lead_s": [e["lcp_lead_s"] for e in ev],
            "lcp_decel_max": round(self.lcp_decel_max, 2),
            "rear_req_decel": round(self.max_rear_req, 2), "rear_gap_short_steps": self.rear_gap_short,
            "min_thw_s": None if not np.isfinite(self.min_thw) else round(float(self.min_thw), 2),
        }
        checks = {
            "no_collision (5.3.2.2, 5.3.7.2.5.1)": res["collisions"] == 0,
            "no_road_departure (5.3.7.1.1)": not (res["crashed"] and res["collisions"] == 0),
            "lat_acc<=3.3 (5.3.7.1.2)": a_lk <= LAT_MAX + 0.3 + 1e-6,
            "lat_jerk<=5 (5.3.7.1.2.1, 6.2.3)": m["max_lat_jerk"] <= 5.0,
            "no_speeding (도로교통법 17조)": res["speeding_s"] <= 1.0,
        }
        if ev or sc.get("need_lc") is not None:
            checks.update({
                "indicator>=3s_before_LCM & whole_LCM (6.2.6, 6.2.7)": len(unsig) == 0,
                "LC_lat_acc<=3.5 (6.2.3)": a_lc <= 3.5,
                "rear_vehicle_decel<=3 & 1s_gap (6.2.4.1)": self.max_rear_req <= 3.0 and self.rear_gap_short == 0,
                "LCP_decel<=2 (6.2.4.3)": self.lcp_decel_max <= 2.0 + 1e-6,
            })
        if sc.get("need_lc") is not None:
            tgt = sc["need_lc"]
            checks["lane_change_completed (4.2.5.1.3)"] = any(abs(e.get("to", 1e9) - tgt) < 0.1 for e in done)
        if sc.get("twoway"):
            checks["no_opposite_lane (6.2.2)"] = res["centerline_s"] == 0
        m["checks"] = {k: bool(v) for k, v in checks.items()}
        m["dcas_pass"] = all(checks.values())
        return m


def autonomous_completion(res: dict) -> bool:
    """자율 완주: 출발→도착을 끝까지, 개입(실드·전문가 인계·감시 인계) 0회, 충돌·이탈 0."""
    return bool(res.get("finished") and not res.get("crashed") and res.get("collisions", 0) == 0
                and res.get("takeovers", 0) == 0)


def run(env, act, seed=0):
    """bench.run_episode 와 같은 결과 + DCAS 판정 + 자율 완주."""
    from . import bench
    env.monitor = DcasMonitor()
    res = bench.run_episode(env, act, seed=seed)
    res.update(env.monitor.results(env, res, env.dcas))
    res["takeovers"] = res.get("takeovers", 0)
    res["autonomous"] = autonomous_completion(res)
    return res


def evaluate(policy, names=None, seed=0, shield=False):
    from . import bench
    act = bench.make_policy(policy, shield=shield)
    card = {}
    for name in names or DCAS_NAMES:
        env = make_env(name, seed=seed)
        r = run(env, act, seed=seed)
        r["score"] = bench.scenario_score(r, REF_SPEED.get(name, 40))
        sh = getattr(act, "cache", {}).get(id(env))
        if sh is not None:
            r.update(sh.stats())
            r["takeovers"] = int(round(r.get("long_override_share", 0) * r.get("steps", 0)
                                       + r.get("lat_override_share", 0) * r.get("steps", 0)))
            r["autonomous"] = autonomous_completion(r)
        card[name] = r
    return summarize(card)


def summarize(card):
    sc = list(card.values())
    return {"suite": SUITE, "total": round(float(np.mean([r["score"] for r in sc])), 2),
            "dcas_pass": sum(r["dcas_pass"] for r in sc), "n": len(sc),
            "autonomous": sum(bool(r["autonomous"]) for r in sc),
            "autonomous_rate": round(sum(bool(r["autonomous"]) for r in sc) / max(len(sc), 1), 3),
            "crashes": sum(bool(r["crashed"]) for r in sc), "collisions": sum(r["collisions"] for r in sc),
            "scenarios": card}


def table_md(card_by_policy: dict) -> str:
    """정책별 DCAS 점수표 (시나리오 × 정책: 합격·자율 완주·점수)."""
    pols = list(card_by_policy)
    head = "| 시나리오 | 근거 | " + " | ".join(pols) + " |\n|---|---|" + "---|" * len(pols) + "\n"
    rows = []
    for name in DCAS_NAMES:
        cells = []
        for p in pols:
            r = card_by_policy[p]["scenarios"].get(name)
            if r is None:
                cells.append("-")
                continue
            fail = [k.split(" ")[0] for k, v in r["checks"].items() if not v]
            cells.append(f"{'✅' if r['dcas_pass'] else '❌'}{'🚗' if r['autonomous'] else '✋'} {r['score']}"
                         + (f" ({', '.join(fail)})" if fail else ""))
        rows.append(f"| {name} | {DCAS01[name]['ref'].split(' /')[0]} | " + " | ".join(cells) + " |")
    foot = "| **합계** | | " + " | ".join(
        f"pass {c['dcas_pass']}/{c['n']}, 자율완주 {c['autonomous']}/{c['n']}, 점수 {c['total']}" for c in card_by_policy.values()) + " |"
    return head + "\n".join(rows) + "\n" + foot + "\n\n✅ = R171 판정 모두 통과, 🚗 = 자율 완주(개입 0·충돌 0·도착), ✋ = 도착 못 함/개입 필요\n"
