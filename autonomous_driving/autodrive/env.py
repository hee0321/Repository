"""자율주행 환경. gymnasium이 있으면 gymnasium.Env로, 없으면 같은 인터페이스의 일반 클래스로 동작.

합성 서킷(닫힌 트랙), 실제 한국 도로 경로(열린 경로 + 교통 규칙), 시골길(좁은 길·비포장·밤·일반 차량 흐름·보행자)
모두 같은 환경으로 돈다.
"""
from __future__ import annotations

import numpy as np

from .car import BicycleCar, CarParams
from .rules import TrafficRules, ViolationLog, uniform_rules
from .sensors import Lidar
from .track import Track, make_default_track

try:  # gymnasium은 선택 사항 (RL 학습할 때만 필요)
    import gymnasium as gym
    from gymnasium import spaces
    _Base = gym.Env
except ImportError:  # pragma: no cover
    gym, spaces, _Base = None, None, object

N_EXTRA = 17
CAR_HALF_LEN = 2.0
OBS_NAMES = ["speed", "steer", "lane_dev", "sin_he", "cos_he", "limit", "stop_dist", "curv_ahead",
             "lead_gap", "lead_closing", "lead_lat", "mu", "road_width", "sight", "light", "edge_clear",
             "oncoming_dist"]


class AutoDriveEnv(_Base):
    """관측: [라이다 거리(정규화) n개] + 17개:
      속도, 조향각, 목표 차선 대비 오프셋, 방향 오차 sin/cos, 제한속도, 다음 정지 신호까지 거리, 앞 굽이 곡률,
      내 차로 앞 장애물 거리/접근 속도/좌우 위치, 노면 마찰 추정, 도로 폭, 시야 거리, 밝기, 길 가장자리 여유,
      마주 오는 차까지 거리
    행동: [조향 -1~1, 가속 -1~1]  (가속 음수 = 브레이크)
    보상: 이번 스텝에 경로를 따라 나아간 거리(m) - 차선/과속/신호위반/중앙선/가장자리/시야 과속 벌점, 이탈·충돌 -20
    """

    metadata = {"render_modes": []}

    def __init__(self, track: Track | None = None, rules: TrafficRules | None = None, dt: float = 0.05,
                 max_steps: int = 3000, n_rays: int = 15, lidar_noise: float = 0.0, random_start: bool = False,
                 target_laps: int | None = None, seed: int | None = None, params: CarParams | None = None,
                 start_s: float | None = None, name: str | None = None, obstacles: list | None = None):
        from .krrules import load
        self.rules_cfg = load()
        self.track = track or make_default_track()
        self.rules = rules or uniform_rules(self.track.length, None, 0, closed=self.track.closed)
        self.name = name or self.track.name
        self.dt = dt
        self.max_steps = max_steps
        self.random_start = random_start
        self.start_s = start_s
        self.target_laps = target_laps
        self.params = params or CarParams()
        self.base_mu = self.params.mu
        self.rng = np.random.default_rng(seed)
        light = self.rules.light
        self.lidar = Lidar(n_rays=n_rays, noise_std=lidar_noise + 0.15 * (1 - light), rng=self.rng,
                           max_range=30.0 * (0.5 + 0.5 * light))
        self.obs_dim = n_rays + N_EXTRA
        if spaces is not None:
            self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.car: BicycleCar | None = None
        # [{s, d, v(경로 방향 속도, 음수=마주 오는 차), vd(옆 방향), r(반폭), l(길이), kind}]
        self.obstacle_specs = obstacles or []

    # --- 장애물 ---
    def _gap(self, s_obj):
        L = self.track.length
        g = s_obj - self._s
        if self.track.closed:
            g = (g + L / 2) % L - L / 2
        return g

    def visible(self, k):
        return -10 < self._gap(self.ob_s[k]) <= self.sight

    def obstacle_xy(self, visible_only=True):
        ks = [k for k in range(len(self.ob_s)) if (not visible_only or self.visible(k))]
        if not ks:
            return np.zeros((0, 3)), ks
        P = np.array([self.track.point_at(self.ob_s[k]) + self.track.normal_at(self.ob_s[k]) * self.ob_d[k] for k in ks])
        return np.column_stack([P, self.ob_r[ks]]), ks

    def lead_obstacle(self):
        """내 진행 경로에 걸치는 가장 가까운 '보이는' 앞 장애물 → (gap, 속도, d, 반폭, 번호) 또는 None."""
        best = None
        for k in range(len(self.ob_s)):
            g = self._gap(self.ob_s[k])
            if -2.0 < g < min(80, self.sight) and abs(self.ob_d[k] - self._d) < self.ob_r[k] + self.params.car_width / 2 + 0.4:
                if best is None or g < best[0]:
                    best = (g, self.ob_v[k], self.ob_d[k], self.ob_r[k], k)
        return best

    def oncoming(self):
        """보이는 마주 오는 차 중 가장 가까운 것 → (gap, k) 또는 None."""
        best = None
        for k in range(len(self.ob_s)):
            g = self._gap(self.ob_s[k])
            if self.ob_v[k] < -0.5 and -3 < g <= self.sight and (best is None or g < best[0]):
                best = (g, k)
        return best

    def ttc(self):
        lo = self.lead_obstacle()
        if lo is None:
            return np.inf
        gap = lo[0] - self.ob_l[lo[4]] / 2 - CAR_HALF_LEN
        closing = self.car.v - lo[1]
        return max(gap, 0.0) / closing if closing > 0.1 else np.inf

    # --- 상태 ---
    def stop_distance(self):
        """다음 신호가 빨강/노랑이면 정지선까지 거리(m), 아니면 inf."""
        nx = self.rules.next_signal(self._s, 150.0)
        if nx is None:
            return np.inf
        d, sg = nx
        return d if sg.state(self.t) != "green" else np.inf

    def edge_clearance(self):
        return self.track.width / 2 - abs(self._d) - self.params.car_width / 2

    def _observe(self) -> np.ndarray:
        c, p = self.car, self.params
        s, d, self._seg = self.track.project(c.pos, hint=getattr(self, "_seg", None))
        self._s, self._d = s, d
        self.sight = self.rules.sight_distance(self.track, s)
        self.circles, self._vis = self.obstacle_xy()
        self.lidar_ranges = self.lidar.scan(c.pos, c.yaw, self.track.walls, self.circles)
        he = np.angle(np.exp(1j * (c.yaw - self.track.heading_at(s))))
        self._he = he
        self._dref = self.rules.lane_ref_at(s)
        stop = self.stop_distance()
        lo = self.lead_obstacle()
        lead = [1.0, 0.0, 0.0] if lo is None else [min(lo[0], 50.0) / 50.0, np.clip((c.v - lo[1]) / 20.0, -1, 1),
                                                    (lo[2] - d) / (self.track.width / 2)]
        on = self.oncoming()
        obs = np.concatenate([
            self.lidar_ranges / 30.0,
            [c.v / p.max_speed, c.steer / p.max_steer, (d - self._dref) / (self.track.width / 2), np.sin(he), np.cos(he),
             self.rules.limit_at(s) / p.max_speed, min(stop, 100.0) / 100.0,
             np.clip(self.track.curvature_ahead(s, 15.0) * 5.0, 0, 2)],
            lead,
            [self.params.mu, self.track.width / 8.0, self.sight / 150.0, self.rules.light,
             np.clip(self.edge_clearance() / 2.0, -1, 2), 1.0 if on is None else min(on[0], 100) / 100.0],
        ])
        return obs.astype(np.float32)

    def _info(self) -> dict:
        L = self.track.length
        return {"s": self._s, "offset": self._d, "lane_dev": self._d - self._dref, "heading_err": self._he,
                "speed": self.car.v, "laps": self.progress / L, "progress_m": self.progress,
                "completion": min(1.0, self.progress / self.goal_m), "crashed": self.crashed,
                "finished": self.finished, "t": self.t, "slipping": self.car.slipping,
                "limit": self.rules.limit_at(self._s), "sight": self.sight, "mu": self.params.mu,
                "violations": self.vlog.summary()}

    @property
    def goal_m(self):
        if self.track.closed:
            return (self.target_laps or 1) * self.track.length
        return self.track.length - (self.start_s or 0.0) - 3.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.lidar.rng = self.rng
        L = self.track.length
        if self.start_s is not None:
            s0 = self.start_s
        elif self.random_start:
            s0 = float(self.rng.uniform(0, L if self.track.closed else max(1.0, L - 60)))
        else:
            s0 = 0.0
        self._seg = None
        p0 = self.track.point_at(s0) + self.track.normal_at(s0) * self.rules.lane_ref_at(s0)
        yaw0 = self.track.heading_at(s0)
        v0 = 0.0
        if self.random_start:
            p0 = p0 + self.track.normal_at(s0) * self.rng.uniform(-0.5, 0.5) * min(1.0, self.track.width / 6.4)
            yaw0 += self.rng.uniform(-0.15, 0.15)
            v0 = float(self.rng.uniform(0, min(8.0, self.rules.limit_at(s0))))
        self.car = BicycleCar(p0[0], p0[1], yaw0, v=v0, params=self.params)
        self.steps, self.t, self.progress, self.crashed, self.finished = 0, 0.0, 0.0, False, False
        self.vlog = ViolationLog()
        sp = self.obstacle_specs
        self.ob_s = np.array([o["s"] for o in sp], float)
        self.ob_d = np.array([o.get("d", 0.0) for o in sp], float)
        self.ob_v = np.array([o.get("v", 0.0) for o in sp], float)
        self.ob_vd = np.array([o.get("vd", 0.0) for o in sp], float)
        self.ob_r = np.array([o.get("r", 1.0) for o in sp], float)
        self.ob_l = np.array([o.get("l", 2 * o.get("r", 1.0)) for o in sp], float)
        self.ob_kind = [o.get("kind", "car" if o.get("v", 0) else "object") for o in sp]
        self.ob_dlim = np.array([o.get("d_to", o.get("d", 0.0)) for o in sp], float)
        self._enter_on = np.zeros(len(sp), bool)
        self._ob_vdes = np.array([o.get("v_after", o.get("v", 0.0)) if o.get("kind") == "entering" else o.get("v", 0.0) for o in sp], float)
        self._pass_t = -1e9
        self._seg = self.track.project(p0)[2]
        self._s = s0
        obs = self._observe()
        self._prev_s = self._s
        self._prev_gap = np.array([self._gap(x) for x in self.ob_s])
        self.prev_action = np.zeros(2)
        self.params.mu = self.rules.mu_at(self._s, self.base_mu)
        return obs, self._info()

    def _move_obstacles(self):
        if not len(self.ob_s):
            return
        # 옆길에서 들어오는 차: 내 차가 trigger m 안으로 오면 내 차로로 들어오고, 다 들어오면 v_after 까지 가속
        for k, o in enumerate(self.obstacle_specs):
            if o.get("kind") != "entering":
                continue
            if not self._enter_on[k] and 0 < self._gap(self.ob_s[k]) < o.get("trigger", 35.0):
                self._enter_on[k] = True
            if self._enter_on[k]:
                tgt = self.ob_dlim[k]
                stp = o.get("vd", 1.5) * self.dt
                self.ob_d[k] = tgt if abs(tgt - self.ob_d[k]) <= stp else self.ob_d[k] + np.sign(tgt - self.ob_d[k]) * stp
                if abs(self.ob_d[k] - tgt) >= 1.0:
                    self.ob_v[k] = 0.0      # 아직 들어오는 중: 서서 기다림 (다 들어오면 위의 차간 유지 규칙으로 출발)
        # 같은 방향 차들은 내 차를 뒤에서 들이받지 않게 따라온다(간단한 차간 유지): 내 차가 앞 25 m 안 같은 차로면 내 속도까지
        for k in range(len(self.ob_s)):
            if self.ob_kind[k] in ("car", "entering") and self._ob_vdes[k] > 0.5 and (self.ob_kind[k] != "entering" or self._enter_on[k]):
                behind = -self._gap(self.ob_s[k])   # 내 차가 그 차보다 앞선 거리
                same = abs(self._d - self.ob_d[k]) < self.ob_r[k] + self.params.car_width / 2 + 0.3
                tgt = self._ob_vdes[k]
                if same and 0 < behind < 25:
                    tgt = min(tgt, self.car.v * min(1.0, max(0.0, (behind - 7.0) / 10.0)))
                self.ob_v[k] = max(0.0, min(tgt, self.ob_v[k] + 1.5 * self.dt)) if tgt >= self.ob_v[k] else max(tgt, self.ob_v[k] - 6.0 * self.dt)
        self.ob_s = self.ob_s + self.ob_v * self.dt
        if self.track.closed:
            self.ob_s %= self.track.length
        else:
            self.ob_s = np.clip(self.ob_s, -50, self.track.length + 50)
        # 옆으로 움직이는 것(길 건너는 보행자): 차가 15 m 안으로 오면 건너기 시작, 목표 d에서 멈춤
        for k in np.nonzero(self.ob_vd)[0]:
            if self.ob_kind[k] == "entering":
                continue
            if 0 < self._gap(self.ob_s[k]) < 25 or self.ob_d[k] != self.ob_specs_d0(k):
                step = self.ob_vd[k] * self.dt
                tgt = self.ob_dlim[k]
                if abs(tgt - self.ob_d[k]) <= abs(step):
                    self.ob_d[k] = tgt
                else:
                    self.ob_d[k] += np.sign(tgt - self.ob_d[k]) * abs(step)

    def ob_specs_d0(self, k):
        return self.obstacle_specs[k].get("d", 0.0)

    def _pass_events(self, ev):
        """장애물을 지나친 순간(앞→뒤)의 안전성 평가 (시골길 지표)."""
        cfg = self.rules_cfg["rural"]
        half = self.params.car_width / 2
        for k in range(len(self.ob_s)):
            g = self._gap(self.ob_s[k])
            if self._prev_gap[k] > 0 >= g:
                clear = abs(self._d - self.ob_d[k]) - self.ob_r[k] - half
                kind = self.ob_kind[k]
                v = self.car.v
                if kind == "ped" and (clear < cfg["ped_pass_clearance_m"] and v > cfg["ped_pass_speed_kmh"] / 3.6):
                    self.vlog.ped_unsafe += 1; ev["unsafe"] = True
                elif self.ob_v[k] < -0.5:  # 마주 오는 차와 엇갈림
                    if self.track.width < 6.0 and v > cfg["meet_speed_kmh"] / 3.6 + 0.5:
                        self.vlog.unsafe_meet += 1; ev["unsafe"] = True
                    else:
                        self.vlog.yield_ok += 1; ev["yield_ok"] = True
                elif self.ob_v[k] > 0.5:   # 같은 방향 느린 차 앞지르기: 허용 구간 + 시야 확보 + 마주 오는 차 없음일 때만
                    on = self.oncoming()
                    if (not self.rules.passing_ok_at(self._s) or self.sight < cfg["overtake_min_sight_m"]
                            or on is not None):
                        self.vlog.unsafe_overtake += 1; ev["unsafe"] = True
                    else:
                        self.vlog.legal_overtake = getattr(self.vlog, "legal_overtake", 0) + 1
        self._prev_gap = np.array([self._gap(x) for x in self.ob_s])

    def is_passing(self):
        """같은 방향으로 가는 더 느린 차의 -15 m ~ +25 m 안에서 그 차보다 왼쪽에 있으면 '앞지르는 중'
        (나가기·옆 지나기), 그 뒤 3초는 제 차로로 돌아오는 중으로 본다."""
        for k in range(len(self.ob_s)):
            if self.ob_v[k] > 0.5 and -15 < self._gap(self.ob_s[k]) < 25 and self._d > self.ob_d[k] + 0.5:
                self._pass_t = self.t
                return True
        return self.t - self._pass_t < 3.0

    def collided(self):
        for k in range(len(self.ob_s)):
            g = self._gap(self.ob_s[k])
            if abs(g) < self.ob_l[k] / 2 + CAR_HALF_LEN and abs(self._d - self.ob_d[k]) < self.ob_r[k] + self.params.car_width / 2:
                return True
        return False

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float).reshape(-1)[:2], -1, 1)
        p = self.params
        p.mu = self.rules.mu_at(self._s, self.base_mu)
        gravel = self.rules.gravel_at(self._s)
        steer = a[0] * p.max_steer
        accel = a[1] * (p.max_accel if a[1] >= 0 else p.max_brake)
        self.car.step(steer, accel, self.dt)
        if gravel and self.car.v > 1.0:  # 자갈길: 바퀴가 조금씩 미끄러져 방향이 흔들림
            self.car.yaw += float(self.rng.normal(0, 0.003 * self.car.v))
        self._move_obstacles()
        self.steps += 1
        self.t += self.dt
        prev_s = self._prev_s
        obs = self._observe()
        L = self.track.length
        ds = (self._s - prev_s + L / 2) % L - L / 2 if self.track.closed else self._s - prev_s
        self._prev_s = self._s
        self.progress += ds
        ev = self.vlog.update(self.rules, prev_s, self._s, self.t, self.car.v, self._d, self.dt, p.car_width / 2,
                              passing=self.is_passing())
        ev["unsafe"] = False
        ev["yield_ok"] = False
        self._pass_events(ev)
        rc = self.rules_cfg["rural"]
        edge = self.edge_clearance()
        if edge < rc["min_edge_clearance_m"]:
            self.vlog.edge_close_s += self.dt
        tr, b = rc["reaction_time_s"], rc["safe_decel_mps2"]
        v = self.car.v
        sight_over = v * tr + v * v / (2 * b) > self.sight + 0.5
        if sight_over:
            self.vlog.sight_overspeed_s += self.dt
        if self.car.slipping:
            self.vlog.slip_s += self.dt
        margin = self.track.width / 2 - p.car_width / 2
        hit = self.collided()
        if hit:
            self.vlog.collisions += 1
            if any(self.ob_kind[k] == "entering" and abs(self._gap(self.ob_s[k])) < 6 for k in range(len(self.ob_s))):
                self.vlog.entering_conflict = getattr(self.vlog, "entering_conflict", 0) + 1
        self.crashed = abs(self._d) > margin or hit
        lim = self.rules.limit_at(self._s)
        over = max(0.0, v - lim)
        reward = (ds - 0.05 * abs(self._d - self._dref) - 0.02 * abs(a[0] - self.prev_action[0])
                  - 0.3 * over * self.dt * 10 - (1.0 if ev["centerline"] else 0.0) * self.dt * 10
                  - 0.5 * max(0.0, rc["min_edge_clearance_m"] + 0.25 - edge) * self.dt * 10
                  - (0.5 if sight_over else 0.0) * self.dt * 10)
        if ev["red_run"]:
            reward -= 20.0
        if ev["unsafe"]:
            reward -= 10.0
        if ev["yield_ok"]:
            reward += 2.0
        self.prev_action = a
        terminated = self.crashed
        if self.crashed:
            reward -= 20.0
        if self.progress >= self.goal_m:
            self.finished = True
            terminated = True
        truncated = self.steps >= self.max_steps
        return obs, float(reward), bool(terminated), bool(truncated), self._info()
