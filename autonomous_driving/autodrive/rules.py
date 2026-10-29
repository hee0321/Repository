"""한국 교통 규칙(단순화): 우측 통행, 안전속도 5030, 어린이보호구역 30 km/h, 신호등 정지/출발."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .krrules import load as _load_cfg

KMH = 1 / 3.6
CFG = _load_cfg()
LANE_W = float(CFG["lane_width_m"])  # 차로 폭 (m)
_SIG = CFG["traffic_signal"]


@dataclass
class Signal:
    s: float                 # 경로 위 정지선 위치(m)
    green: float = float(_SIG["green_s"])
    yellow: float = float(_SIG["yellow_s"])
    red: float = float(_SIG["red_s"])
    offset: float = 0.0      # 시작 위상(s)
    name: str = ""
    right_turn: bool = False  # 이 신호를 지나 우회전하는가 (우회전 일시정지 대상)

    @property
    def cycle(self):
        return self.green + self.yellow + self.red

    def phase_id(self, t):
        return int((t + self.offset) // self.cycle)

    def state(self, t: float) -> str:
        c = (t + self.offset) % (self.green + self.yellow + self.red)
        if c < self.green:
            return "green"
        if c < self.green + self.yellow:
            return "yellow"
        return "red"


@dataclass
class TrafficRules:
    """경로 1 m 간격 표본으로 제한속도·어린이보호구역·양방향 여부·목표 차선 오프셋을 가진다."""
    ds: float
    limit_mps: np.ndarray            # 각 표본의 제한속도 (m/s)
    school: np.ndarray               # 어린이보호구역 여부
    twoway: np.ndarray               # 양방향 도로(가운데 = 중앙선)인지
    lane_ref: np.ndarray             # 목표 횡오프셋 (우측 통행이면 음수 = 오른쪽)
    signals: list[Signal] = field(default_factory=list)
    closed: bool = False
    length: float = 0.0
    # --- 시골길 정보 (도시 시나리오에서는 기본값) ---
    mu: np.ndarray | None = None       # 구간별 노면 마찰 (포장 0.9 / 비포장·자갈 0.45~0.6)
    gravel: np.ndarray | None = None   # 비포장 여부
    light: float = 1.0                 # 밝기 (낮 1.0, 가로등 없는 밤 ~0.15)
    roadside_clear_m: float = 6.0      # 길옆 시야 여유 (논두렁·나무·건물이 가까울수록 굽이에서 앞이 안 보임)
    crests: list = field(default_factory=list)  # [(s, 그 너머 보이는 거리)] 언덕 꼭대기
    rural: bool = False
    passing_ok: np.ndarray | None = None  # 앞지르기 허용 구간(중앙선 점선·직선·시야 확보). None = 전부 금지

    def mu_at(self, s, default=0.9):
        return default if self.mu is None else float(self.mu[self._i(s)])

    def passing_ok_at(self, s):
        return False if self.passing_ok is None else bool(self.passing_ok[self._i(s)])

    def gravel_at(self, s):
        return False if self.gravel is None else bool(self.gravel[self._i(s)])

    def sight_distance(self, track, s, cap=150.0):
        """앞이 보이는 거리(m): 굽이(S ≈ √(8·R·m)), 언덕, 밤(전조등) 제한."""
        S = cap
        idx, dist = track.ahead(s, cap)
        if len(idx):
            k = np.abs(track.curvature[idx])
            S = min(S, float(np.min(np.maximum(dist, 0) + np.sqrt(8 * self.roadside_clear_m / np.maximum(k, 1e-6)))))
        for cs, sight in self.crests:
            g = cs - s
            if 0 < g < cap:
                S = min(S, g + sight)
        S = min(S, 40.0 + 110.0 * self.light)
        return float(S)

    def _i(self, s):
        n = len(self.limit_mps)
        i = int(s / self.ds)
        return i % n if self.closed else min(max(i, 0), n - 1)

    def limit_at(self, s):
        return float(self.limit_mps[self._i(s)])

    def school_at(self, s):
        return bool(self.school[self._i(s)])

    def twoway_at(self, s):
        return bool(self.twoway[self._i(s)])

    def lane_ref_at(self, s):
        return float(self.lane_ref[self._i(s)])

    def limits_ahead(self, s, horizon):
        """(앞까지 거리, 제한속도) 배열."""
        n = int(horizon / self.ds)
        idx = np.arange(int(s / self.ds), int(s / self.ds) + n)
        idx = idx % len(self.limit_mps) if self.closed else idx[idx < len(self.limit_mps)]
        dist = (idx * self.ds - s) % self.length if self.closed else idx * self.ds - s
        return np.maximum(dist, 0.0), self.limit_mps[idx]

    def next_signal(self, s, horizon=150.0):
        best = None
        for sg in self.signals:
            d = (sg.s - s) % self.length if self.closed else sg.s - s
            if 0 <= d <= horizon and (best is None or d < best[0]):
                best = (d, sg)
        return best


def uniform_rules(length, limit_kmh=None, n_signals=0, closed=True, seed=0, lane_ref=0.0, twoway=False):
    """합성 트랙용 규칙. limit_kmh=None이면 제한 없음(차 최고 속도)."""
    rng = np.random.default_rng(seed)
    n = int(np.ceil(length)) + 1
    lim = np.full(n, (limit_kmh or 72.0) * KMH)
    sigs = [Signal(s=float(x), offset=float(rng.uniform(0, 48)), name=f"sig{k}")
            for k, x in enumerate(np.sort(rng.uniform(0.1, 0.9, n_signals)) * length)]
    return TrafficRules(1.0, lim, np.zeros(n, bool), np.full(n, twoway), np.full(n, lane_ref), sigs, closed, length)


class ViolationLog:
    """주행 중 규칙 위반을 센다."""

    def __init__(self, tol_kmh: float | None = None):
        self.tol = (CFG["speeding_tolerance_kmh"] if tol_kmh is None else tol_kmh) * KMH
        self.rt_cfg = CFG["right_turn_stop"]
        self.right_turn_no_stop = 0
        self.right_turn_ok = 0
        self.collisions = 0
        # 시골길 지표
        self.edge_close_s = 0.0
        self.sight_overspeed_s = 0.0
        self.unsafe_overtake = 0
        self.unsafe_meet = 0
        self.yield_ok = 0
        self.ped_unsafe = 0
        self.slip_s = 0.0
        self._rt_stopped = set()
        self.speeding_s = 0.0
        self.school_speeding_s = 0.0
        self.speeding_events = 0
        self.max_over_kmh = 0.0
        self.red_runs = 0
        self.centerline_s = 0.0
        self.centerline_events = 0
        self.signal_stops = 0
        self._speeding = False
        self._cross = False
        self._stopped_at = set()

    def update(self, rules: TrafficRules, prev_s, s, t, v, d, dt, car_half_w, passing=False):
        lim = rules.limit_at(s)
        over = v - lim
        sp = over > self.tol
        if sp:
            self.speeding_s += dt
            self.max_over_kmh = max(self.max_over_kmh, over / KMH)
            if rules.school_at(s):
                self.school_speeding_s += dt
        if sp and not self._speeding:
            self.speeding_events += 1
        self._speeding = sp
        cross = rules.twoway_at(s) and d + car_half_w > float(CFG["centerline"]["margin_m"])  # 차 왼쪽 끝이 중앙선을 넘음
        if cross and passing and rules.passing_ok_at(s):
            cross = False      # 허용 구간에서 느린 차를 앞지르는 중이면 중앙선(점선) 넘기 허용
        self.passing_s = getattr(self, "passing_s", 0.0) + (dt if passing else 0.0)
        if cross:
            self.centerline_s += dt
        if cross and not self._cross:
            self.centerline_events += 1
        self._cross = cross
        events = {"red_run": False}
        for k, sg in enumerate(rules.signals):
            passed = (prev_s < sg.s <= s) if not rules.closed else (0 < (sg.s - prev_s) % rules.length <= (s - prev_s) % rules.length)
            dist = (sg.s - s) % rules.length if rules.closed else sg.s - s
            if 0 <= dist < 6 and v < self.rt_cfg["stop_speed_mps"]:
                self._rt_stopped.add((k, sg.phase_id(t)))
            if passed and sg.state(t) == "red":
                if sg.right_turn and self.rt_cfg["enabled"]:
                    if (k, sg.phase_id(t)) in self._rt_stopped:
                        self.right_turn_ok += 1          # 멈췄다가 우회전: 허용
                    else:
                        self.right_turn_no_stop += 1     # 우회전 일시정지 위반
                        events["red_run"] = True
                else:
                    self.red_runs += 1
                    events["red_run"] = True
            if 0 <= dist < 12 and v < 0.3 and sg.state(t) != "green" and (k, int(t // 48)) not in self._stopped_at:
                self.signal_stops += 1
                self._stopped_at.add((k, int(t // 48)))
        events["speeding"] = sp
        events["centerline"] = cross
        return events

    def summary(self):
        return {"speeding_s": round(self.speeding_s, 2), "speeding_events": self.speeding_events,
                "school_speeding_s": round(self.school_speeding_s, 2), "max_over_kmh": round(self.max_over_kmh, 1),
                "red_runs": self.red_runs, "centerline_s": round(self.centerline_s, 2),
                "centerline_events": self.centerline_events, "signal_stops": self.signal_stops,
                "right_turn_no_stop": self.right_turn_no_stop, "right_turn_ok": self.right_turn_ok,
                "collisions": self.collisions, "edge_close_s": round(self.edge_close_s, 2),
                "sight_overspeed_s": round(self.sight_overspeed_s, 2), "unsafe_overtake": self.unsafe_overtake,
                "unsafe_meet": self.unsafe_meet, "yield_ok": self.yield_ok, "ped_unsafe": self.ped_unsafe,
                "slip_s": round(self.slip_s, 2), "legal_overtake": getattr(self, "legal_overtake", 0),
                "entering_conflict": getattr(self, "entering_conflict", 0)}
