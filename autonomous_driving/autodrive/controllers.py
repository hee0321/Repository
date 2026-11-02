"""고전 제어기: Pure Pursuit 조향 + 속도 계획(굽이·제한속도·신호) + PID 속도 제어."""
from __future__ import annotations

import numpy as np

from .env import CAR_HALF_LEN, AutoDriveEnv


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
        # DCAS 01 모드 (같은 방향 여러 차로 + 선언 최대 횡가속도가 있는 도로, autodrive/dcas.py)
        self.dcas = env.rules.lat_acc_max is not None
        self.lc = None
        self._safety = False
        if self.dcas:
            self.a_lat = min(self.a_lat, 0.9 * env.rules.lat_acc_max)          # 5.3.7.1.2: ≤ 3 m/s² (여유 10%)
            self.v_max = min(env.params.max_speed, float(np.max(env.rules.limit_mps)))

    def reset(self):
        self.pid = PID(self.pid.kp, self.pid.ki, self.pid.kd)

    def avoid_offset(self, s):
        """앞에 서 있는 장애물이 내 차로(목표 차선)를 막으면 공간이 넓은 쪽으로 비켜 갈 횡오프셋.
        반환 (offset 또는 None, 막혀서 멈춰야 하는지)."""
        env = self.env
        if self.dcas:
            return None, False   # 여러 차로: 차로 안에서 비틀어 피하지 않고, 멈추거나 지시등 켜고 차로 변경(lane_plan)
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

    # ---------------- DCAS 01: 시스템이 시작하는 차로 변경 (R171 6.2) ----------------
    def _in_lane(self, k, lane):
        env = self.env
        return abs(env.ob_d[k] - lane) < env.rules.lane_w / 2

    def _blocker(self, lane, s, rng=120.0, stationary=False):
        """내 차로 앞 rng m 안: 서 있는 것(차·물체·보행자) 또는 아주 느린 차(제한의 40% 미만) → 번호."""
        env = self.env
        lim = env.rules.limit_at(s)
        for k in np.argsort(env.ob_s):
            g = env._gap(env.ob_s[k])
            if not (0 < g < rng) or not self._in_lane(k, lane):
                continue
            if env._lat_on[k] and abs(env.ob_d[k] - env.ob_dlim[k]) > 0.05:
                continue                      # 건너는 중/끼어드는 중: 곧 비키거나 따라갈 대상
            if abs(env.ob_v[k]) < 1.0 or (not stationary and env.ob_kind[k] == "car" and env.ob_v[k] < 0.4 * lim and g < 50):
                return k
        return None

    def _free_adjacent(self, lane, s):
        env = self.env
        lanes = env.rules.lanes
        for cand in (lane + env.rules.lane_w, lane - env.rules.lane_w):   # 한국: 앞지르기는 왼쪽 먼저
            if any(abs(cand - c) < 0.1 for c in lanes) and self._blocker(cand, s) is None:
                return float(min(lanes, key=lambda c: abs(c - cand)))
        return None

    def _gap_ok(self, to, s, strict=True):
        """목표 차로 틈 검사 (6.2.4.1: 뒤에서 오는 차가 3 m/s² 넘게 감속하지 않고 1초 거리 유지 — 여유 두고 2 m/s²,
        6.2.4.2 미검지 가정 대신 시뮬 진실값). 반환 (가능?, 막은 차 번호, 앞쪽인지)."""
        env = self.env
        ve = env.car.v
        sc = 1.0 if strict else 0.7
        for k in range(len(env.ob_s)):
            moving_in = env._lat_on[k] and abs(env.ob_dlim[k] - to) < env.rules.lane_w / 2
            if not (abs(env.ob_d[k] - to) < env.rules.lane_w / 2 + 0.3 or moving_in):
                continue
            g = env._gap(env.ob_s[k])
            half = env.ob_l[k] / 2 + CAR_HALF_LEN
            vk = env.ob_v[k]
            if g >= 0:
                gb = g - half
                if env.ob_kind[k] == "ped":
                    need = 40.0
                else:
                    need = max(ve * 1.0 + 3.0, (ve * ve - vk * vk) / (2 * 2.0) + 6.0 if ve > vk else 0.0)
                if gb < need * sc:
                    return False, k, True
            else:
                gb = -g - half
                fut = gb - max(vk - ve, 0.0) * 2.5          # MCC 시작까지 ~1.4 s + 여유 (6.2.4.1 가정)
                if fut < (ve * 1.0 + 3.0) * sc:
                    return False, k, False
                if vk > ve and (vk - ve) ** 2 / (2 * max(fut - ve * 1.0, 0.1)) > 2.0 / sc:
                    return False, k, False
        return True, None, None

    def lane_plan(self, s, ld):
        """목표 횡오프셋 (차로 가운데 또는 차로 변경 중 최소 저크 곡선). 지시등은 env.set_indicator 로 켠다."""
        env, ru, c = self.env, self.env.rules, self.env.car
        cur = env._lane_c if env._lane_c is not None else ru.lane_of(env._d)
        want = None
        req = ru.lc_request_at(s)
        if req is not None and abs(req["to"] - cur) > 0.1:
            want = req["to"]
        elif self._blocker(cur, s) is not None:
            want = self._free_adjacent(cur, s)
        lc = self.lc
        if lc is not None and lc["s0"] is None and (want is None or abs(want - lc["to"]) > 0.1):
            self.lc = lc = None
            env.set_indicator(0)
        if lc is None and want is not None:
            lc = self.lc = dict(to=want, frm=cur, s0=None, side=int(np.sign(want - cur)), wait=None)
        if lc is None:
            return cur
        if lc["s0"] is not None and not env._lcm_on and abs(env._lane_c - lc["to"]) < 0.1:
            self.lc = None                                  # 끝: 새 차로 안정 (env가 지시등 끔)
            return lc["to"]
        if env.indicator != lc["side"]:
            env.set_indicator(lc["side"])                   # 7 s 안에 MCC 못 하면 env가 끔 → 다시 켬 = 새 PCC
        ok, k, ahead = self._gap_ok(lc["to"], s, strict=lc["s0"] is None)
        lc["wait"] = None if ok else (k, ahead)
        if lc["s0"] is None:
            if ok and env.indicator_t0 is not None and env.t - env.indicator_t0 >= 3.2:   # 6.2.7: 3 s
                lc["s0"] = s + ld
                lc["len"] = max(5.0 * c.v, 15.0)            # 최소 저크 횡 이동: 최대 횡가속도 ≈ 5.8·3.5/25 < 1 m/s²
                b = self._blocker(lc["frm"], s)
                if b is not None:                           # 막힌 차로에서 비켜 나감: 남은 거리 안에서, 그만큼 천천히
                    gb = env._gap(env.ob_s[b]) - env.ob_l[b] / 2 - CAR_HALF_LEN
                    lc["len"] = float(np.clip(gb + CAR_HALF_LEN - 1.0, 10.0, lc["len"]))
                    lc["blk"] = b
        elif not ok and not env._lcm_on and abs(env._lane_c - lc["frm"]) < 0.1 and abs(env._d - lc["frm"]) < 0.6:
            lc["s0"] = None                                 # 아직 차선 안 넘음 → 취소하고 기다림
        if lc["s0"] is None:
            return lc["frm"]
        return self._ramp(s + ld)

    def _ramp(self, x):
        lc = self.lc
        u = float(np.clip((x - lc["s0"]) / lc["len"], 0.0, 1.0))
        return lc["frm"] + (lc["to"] - lc["frm"]) * (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5)

    def _path_clears(self, k):
        """차로 변경 중: 계획한 횡 경로가 그 장애물 뒤끝에서 이미 비켜 있으면 True (앞 장애물로 보지 않음)."""
        env, lc = self.env, self.lc
        if not self.dcas or lc is None or lc["s0"] is None:
            return False
        lat = self._ramp(env.ob_s[k] - env.ob_l[k] / 2 - 3.0)
        return abs(lat - env.ob_d[k]) >= env.ob_r[k] + env.params.car_width / 2 + 0.3

    def _predicted_conflict(self, s, v):
        """옆으로 움직이는 것(건너는 보행자·자전거, 끼어드는 차)이 내가 도착할 때 내 길에 있을지 → (간격, 속도, 번호)."""
        env = self.env
        best = None
        ve = max(v, 2.0)
        for k in range(len(env.ob_s)):
            if not (env._lat_on[k] and abs(env.ob_d[k] - env.ob_dlim[k]) > 0.05):
                continue
            g = env._gap(env.ob_s[k])
            if not (-3 < g < 90):
                continue
            step = np.sign(env.ob_dlim[k] - env.ob_d[k]) * env.ob_vd[k]
            for tau in np.arange(0.0, 6.0, 0.25):
                dk = env.ob_d[k] + step * tau
                if (step > 0 and dk > env.ob_dlim[k]) or (step < 0 and dk < env.ob_dlim[k]):
                    dk = env.ob_dlim[k]
                ds = g + env.ob_v[k] * tau - ve * tau
                if abs(ds) < env.ob_l[k] / 2 + CAR_HALF_LEN + 1.5 and \
                        abs(dk - env._d) < env.ob_r[k] + env.params.car_width / 2 + 0.5:
                    if best is None or g < best[0]:
                        best = (g, max(env.ob_v[k], 0.0), k)
                    break
        return best

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
        self._safety = False
        if self.dcas:
            pc = self._predicted_conflict(s, v)
            if pc is not None:
                g, vo, k = pc
                margin = 3.0 if env.ob_kind[k] == "ped" else 6.0 + 0.8 * v
                gap = g - env.ob_l[k] / 2 - CAR_HALF_LEN - margin
                v_ref = min(v_ref, float(np.sqrt(vo ** 2 + 2 * 3.5 * max(gap, 0))), vo + max(gap, 0) * 0.5)
                self.rule_note = "건너는 보행자·끼어드는 차 예측 → 감속/정지"
                self._safety = True
            if self.lc is not None and self.lc["wait"] is not None and self.lc["s0"] is None:
                k, ahead = self.lc["wait"]
                if env.ob_kind[k] != "ped":   # 목표 차로 차 때문에 못 들어감 → 늦춰 그 차를 먼저 보냄
                    v_ref = min(v_ref, max(env.ob_v[k] - (2.0 if ahead else 3.0), 0.0))
                    self.rule_note = "차로 변경 틈 기다림(속도 맞춤)"
            if self.lc is not None and self.lc["s0"] is not None:
                D = abs(self.lc["to"] - self.lc["frm"])   # 최소 저크 곡선 최대 횡가속도 5.77·D·v²/len² ≤ 1.2 m/s²
                v_ref = min(v_ref, float(np.sqrt(1.2 * self.lc["len"] ** 2 / (5.77 * D))))
        if self.dcas and env.rules.lanes and (self.lc is None or self.lc["s0"] is None):
            cur = env._lane_c if env._lane_c is not None else env.rules.lane_of(env._d)
            b = self._blocker(cur, s, 150.0, stationary=True)      # 멀리 선 것: 미리 부드럽게 (≤ 1.8 m/s²)
            if b is not None:
                gb = env._gap(env.ob_s[b]) - env.ob_l[b] / 2 - CAR_HALF_LEN
                v_ref = min(v_ref, float(np.sqrt(2 * 1.8 * max(gb - 10.0, 0.0))))
                self.rule_note = "앞에 선 차·물체 → 미리 감속(옆 차로 비면 차로 변경)"
        lo = env.lead_obstacle()
        if lo is not None and (lo[4] == self._ot or self._path_clears(lo[4])):
            lo = None
        if lo is not None:
            g, vo, dd, r, _ = lo
            off, blocked = self.avoid_offset(s)
            if vo > 1.0 or blocked or off is None:
                # 차간 여유: 정지 장애물은 6 m, 움직이는 앞차는 길이 반영(범퍼 사이) + 6 m + 0.8초 거리
                gap = g - r - 1.2 - 6.0 if vo <= 1.0 else g - env.ob_l[lo[4]] / 2 - 2.2 - 6.0 - 0.8 * v
                if self.dcas and vo <= 1.0:
                    gap = g - env.ob_l[lo[4]] / 2 - CAR_HALF_LEN - (10.0 if self.lc is None or self.lc["s0"] is None else 1.0)
                elif self.dcas:
                    gap = g - env.ob_l[lo[4]] / 2 - CAR_HALF_LEN - 4.0 - 1.4 * v   # 차간 ≈ 1.4 s + 4 m
                self._safety = True
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
        if self.dcas and env.rules.lanes:
            lat = self.lane_plan(s, ld)
        if ot is not None and off is None:
            lat = ot
        if y is not None and off is None:
            lat = y
        tgt = env.track.point_at(s + ld) + env.track.normal_at(s + ld) * lat
        dx, dy = tgt - car.pos
        alpha = np.arctan2(dy, dx) - car.yaw
        steer = np.arctan2(2 * p.wheelbase * np.sin(alpha), ld)
        v_ref = self.target_speed(s, car.v)
        self.last_v_ref = v_ref
        accel = self.pid(v_ref - car.v, env.dt)
        if self.dcas and not self._safety and car.v > v_ref - 3.0:
            accel = max(accel, -2.0)   # 5.3.6.3.1/6.2.4.3: 안전 때문이 아니면 부드럽게 (≤ 2 m/s²)
        if self.dcas and self._safety:
            # 앞차를 따라 멈출 때도 필요한 만큼만 (급제동은 실제로 필요할 때만)
            pass
        if v_ref < 0.05 and car.v < 0.3:
            accel = -p.max_brake  # 정지 유지
        a_norm = accel / p.max_accel if accel >= 0 else accel / p.max_brake
        return np.array([steer / p.max_steer, np.clip(a_norm, -1, 1)], dtype=np.float32)
