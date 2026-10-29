"""고전 제어기: Pure Pursuit 조향 + 속도 계획(굽이·제한속도·신호) + PID 속도 제어."""
from __future__ import annotations

import numpy as np

from .env import AutoDriveEnv


class PID:
    def __init__(self, kp: float, ki: float = 0.0, kd: float = 0.0, i_limit: float = 5.0):
        self.kp, self.ki, self.kd, self.i_limit = kp, ki, kd, i_limit
        self.i, self.prev = 0.0, None

    def __call__(self, err: float, dt: float) -> float:
        self.i = float(np.clip(self.i + err * dt, -self.i_limit, self.i_limit))
        der = 0.0 if self.prev is None else (err - self.prev) / dt
        self.prev = err
        return self.kp * err + self.ki * self.i + self.kd * der


class PurePursuitController:
    """중앙선(+우측 차로 오프셋) 위 '앞쪽 목표점'을 바라보고 조향하는 Pure Pursuit.

    δ = atan(2 L sin α / Ld),  Ld = k·v + Ld0
    목표 속도 = min( v_max, √(a_lat/|κ|) (굽이),
                    √(v_lim² + 2·b·거리) (앞 구간 제한속도, 예: 어린이보호구역 30 km/h),
                    √(2·b·정지선까지 거리) (빨강/노랑 신호) )
    """

    def __init__(self, env: AutoDriveEnv, lookahead_gain: float = 0.6, lookahead_min: float = 4.0,
                 v_max: float = 15.0, a_lat: float = 4.0, decel: float = 2.0, speed_margin_kmh: float = 1.0,
                 speed_pid=(1.5, 0.1, 0.0)):
        self.env = env
        self.k, self.ld0 = lookahead_gain, lookahead_min
        self.v_max, self.a_lat, self.decel = v_max, a_lat, decel
        self.margin = speed_margin_kmh / 3.6
        self.pid = PID(*speed_pid)
        self._rt_done = set()
        self.rule_note = ""
        self._ot = None   # 앞지르는 중인 차 번호

    def reset(self):
        self.pid = PID(self.pid.kp, self.pid.ki, self.pid.kd)

    def avoid_offset(self, s):
        """앞에 서 있는 장애물이 내 차로(목표 차선)를 막으면 공간이 넓은 쪽으로 비켜 갈 횡오프셋.
        반환 (offset 또는 None, 막혀서 멈춰야 하는지)."""
        env = self.env
        L = env.track.length
        half = env.track.width / 2
        cw = env.params.car_width
        for k in np.argsort(env.ob_s) if len(env.ob_s) else []:
            if abs(env.ob_v[k]) > 1.0:
                continue
            if not env.visible(k):
                continue
            g = (env.ob_s[k] - s) % L if env.track.closed else env.ob_s[k] - s
            if env.track.closed and g > L / 2:
                g -= L
            if not (-8.0 < g < 40.0):
                continue
            d, r = env.ob_d[k], env.ob_r[k]
            lane = env.rules.lane_ref_at(env.ob_s[k])
            if abs(d - lane) >= r + cw / 2 + 0.4:
                continue
            need = r + cw / 2 + 0.6
            left_room, right_room = half - (d + r), (d - r) + half
            if max(left_room, right_room) < cw + 0.6 or env.ob_vd[k] != 0 and env.ob_d[k] != env.ob_dlim[k]:
                return None, True  # 비켜 갈 공간이 없거나 길을 건너는 중 → 멈춤
            if env.ob_kind[k] == "ped":
                need = r + cw / 2 + env.rules_cfg["rural"]["ped_pass_clearance_m"] + 0.2
                if env.rules.rural and env.rules.twoway_at(env.ob_s[k]) and d < 0:
                    # 중앙선 있는 길: 보행자 때문에 중앙선을 넘지 않고 내 차로 안에서 최대한 띄우고 서행(20 km/h 미만)
                    from .rules import CFG
                    return min(d + need, float(CFG["centerline"]["margin_m"]) - cw / 2 - 0.1), False
            lim = half - cw / 2 - 0.3
            off = d + need if left_room >= right_room else d - need
            if abs(off) > lim:
                off = float(np.clip(off, -lim, lim))
                if abs(off - d) < r + cw / 2 + 0.15:
                    return None, True
            return off, False
        return None, False

    def overtake_update(self, s):
        """시골길 앞지르기: 같은 방향 느린 차(제한속도보다 15 km/h 이상 느림)가 앞에 있고, 앞 120 m가 전부
        허용 구간(중앙선 점선·직선·보호구역 아님), 시야 ≥ 앞지르기 최소 시야 + 10 m, 마주 오는 차 없음일 때만 시작.
        마주 오는 차가 보이면 옆에 닿기 전엔 포기하고 제 차로로 돌아온다. 반환: 목표 횡오프셋 또는 None."""
        env = self.env
        if not env.rules.rural or env.rules.passing_ok is None:
            return None
        rc = env.rules_cfg["rural"]
        on = env.oncoming()
        if self._ot is not None:
            g = env._gap(env.ob_s[self._ot])
            if g < -10 or (on is not None and g > 2):
                self._ot = None
            else:
                self.rule_note = "허용 구간 앞지르기(시야 확보·마주 오는 차 없음)"
                return 1.6
        lo = env.lead_obstacle()
        if lo is None or on is not None:
            return None
        g, vo, _, _, k = lo
        lim = env.rules.limit_at(s)
        if not (0.5 < vo < lim - 15 / 3.6 and g < 40):
            return None
        ahead = all(env.rules.passing_ok_at(s + x) for x in range(0, 121, 5))
        if ahead and env.sight >= rc["overtake_min_sight_m"] + 10:
            self._ot = k
            return 1.6
        return None

    def yield_offset(self):
        """좁은 길에서 마주 오는 차가 보이면 오른쪽 끝으로 붙는다."""
        env = self.env
        on = env.oncoming()
        if on is None or env.track.width >= 6.0 or on[0] > 60:
            return None
        return -(env.track.width / 2 - env.params.car_width / 2 - 0.3)

    def target_speed(self, s: float, v: float) -> float:
        env = self.env
        a_lat = min(self.a_lat, 0.7 * env.params.mu * 9.81)
        horizon = max(15.0, v * v / (2 * self.decel) + 8.0)
        idx, dist = env.track.ahead(s, horizon)
        if len(idx):
            kap = np.abs(env.track.curvature[idx])
            v_curve = np.sqrt(a_lat / np.maximum(kap, 1e-6))
            v_c = float(np.min(np.sqrt(v_curve ** 2 + 2 * self.decel * np.maximum(dist - 2.0, 0))))
        else:
            v_c = self.v_max
        dl, lims = env.rules.limits_ahead(s, horizon)
        v_l = float(np.min(np.sqrt(np.maximum(lims - self.margin, 0.5) ** 2 + 2 * self.decel * np.maximum(dl - 0.8 * v, 0)))) if len(lims) else self.v_max
        v_ref = min(self.v_max, v_c, v_l)
        nx = env.rules.next_signal(s, 150.0)
        self.rule_note = "규정 속도 유지"
        if v_l < v_c and v_l < self.v_max:
            self.rule_note = "제한속도/어린이보호구역 감속"
        if nx is not None:
            d, sg = nx
            st = sg.state(env.t)
            can_stop = v * v / (2 * env.params.max_brake) < d - 0.5
            if st == "red" or (st == "yellow" and can_stop):
                key = (env.rules.signals.index(sg), sg.phase_id(env.t))
                if sg.right_turn and st == "red" and key in self._rt_done:
                    rt = env.rules_cfg["right_turn_stop"]["max_turn_speed_kmh"] / 3.6
                    v_ref = min(v_ref, rt)          # 일시정지 끝 → 천천히 우회전
                    self.rule_note = "우회전 일시정지 후 서행"
                else:
                    v_ref = min(v_ref, np.sqrt(2 * self.decel * max(d - 1.5, 0.0)))
                    self.rule_note = "우회전 일시정지" if sg.right_turn else "빨간불 정지"
                    if sg.right_turn and d < 6 and v < 0.3:
                        self._rt_done.add(key)
        # 시골길: 앞이 보이는 거리 안에서 멈출 수 있는 속도 (반응 1초 + 감속 4 m/s²)
        rc = env.rules_cfg["rural"]
        if env.rules.rural:
            tr, b = rc["reaction_time_s"], rc["safe_decel_mps2"]
            v_s = b * (-tr + np.sqrt(tr * tr + 2 * max(env.sight - 2.0, 0) / b)) * 0.95
            if v_s < v_ref:
                v_ref = v_s
                self.rule_note = "시야 거리 안 정지 가능 속도"
            vg = rc.get("gravel_speed_kmh", 22) / 3.6          # 비포장·자갈: 미끄러짐·흔들림 → 서행
            if env.rules.gravel is not None:
                for x in range(0, int(horizon) + 1, 2):
                    if env.rules.gravel_at(s + x):
                        v_g = float(np.sqrt(vg ** 2 + 2 * self.decel * max(x - 2.0, 0)))
                        if v_g < v_ref:
                            v_ref = v_g
                            self.rule_note = "비포장·자갈길 서행"
                        break
            on = env.oncoming()
            if on is not None and env.track.width < 6.0 and on[0] < 50:
                v_ref = min(v_ref, rc["meet_speed_kmh"] / 3.6 - 0.5)
                self.rule_note = "마주 오는 차 양보(오른쪽 붙기·서행)"
            for k in range(len(env.ob_s)):
                g = env._gap(env.ob_s[k])
                if env.ob_kind[k] == "ped" and env.visible(k) and -3 < g < 25:
                    v_ref = min(v_ref, rc["ped_pass_speed_kmh"] / 3.6 - 1.0)
                    self.rule_note = "보행자 옆 서행"
        # 앞 장애물: 움직이면 따라가기(ACC), 서 있고 비켜 갈 수 없으면 멈추기
        lo = env.lead_obstacle()
        if lo is not None and lo[4] == self._ot:
            lo = None
        if lo is not None:
            g, vo, dd, r, _ = lo
            off, blocked = self.avoid_offset(s)
            if vo > 1.0 or blocked or off is None:
                # 차간 여유: 정지 장애물은 6 m, 움직이는 앞차는 길이 반영(범퍼 사이) + 6 m + 0.8초 거리
                gap = g - r - 1.2 - 6.0 if vo <= 1.0 else g - env.ob_l[lo[4]] / 2 - 2.2 - 6.0 - 0.8 * v
                v_ref = min(v_ref, float(np.sqrt(max(vo, 0) ** 2 + 2 * self.decel * max(gap, 0))),
                            max(vo, 0) + max(gap, 0) * 0.5)
                self.rule_note = "앞차 따라가기" if vo > 1.0 else "장애물 앞 정지"
        return float(v_ref)

    def act(self, obs=None) -> np.ndarray:
        env, car = self.env, self.env.car
        p = env.params
        s = env._s
        ld = self.k * car.v + self.ld0
        off, _ = self.avoid_offset(s)
        y = self.yield_offset()
        ot = self.overtake_update(s)
        lat = env.rules.lane_ref_at(s + ld) if off is None else off
        if ot is not None and off is None:
            lat = ot
        if y is not None and off is None:
            lat = y
        tgt = env.track.point_at(s + ld) + env.track.normal_at(s + ld) * lat
        dx, dy = tgt - car.pos
        alpha = np.arctan2(dy, dx) - car.yaw
        steer = np.arctan2(2 * p.wheelbase * np.sin(alpha), ld)
        v_ref = self.target_speed(s, car.v)
        accel = self.pid(v_ref - car.v, env.dt)
        if v_ref < 0.05 and car.v < 0.3:
            accel = -p.max_brake  # 정지 유지
        a_norm = accel / p.max_accel if accel >= 0 else accel / p.max_brake
        return np.array([steer / p.max_steer, np.clip(a_norm, -1, 1)], dtype=np.float32)
