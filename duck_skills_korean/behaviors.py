"""행동 컨트롤러: Behavior → 매 스텝(50 Hz) 정책 명령(Cmd). 새 학습 없이 v8 정책의 명령 입력만 조종합니다."""
from __future__ import annotations
import math
import os
import numpy as np
from duck_sim import Cmd, DuckSim
from korean_commands import Behavior


def _clip(x, a):
    return max(-a, min(a, x))


class Controller:
    def __init__(self, b: Behavior, sim: DuckSim):
        self.b, self.sim, self.t0 = b, sim, sim.t
        self.head = tuple(b.params.get("head", (0, 0, 0, 0)))

    @property
    def tau(self):
        return self.sim.t - self.t0

    def cmd(self) -> Cmd:
        return Cmd()


# ── 장애물 피하기 (가상 거리 센서 = MuJoCo 광선) ─────────────────────────────
CLEAR = 0.75       # 이보다 가까운 방향은 막힌 것으로 봄 (m, 몸통 중심 기준)
CORRIDOR = 0.21    # 덕이 지나갈 통로 반폭 (몸 반폭 약 0.07 m + 여유)
SIDE = 0.30        # 옆(35°~85°)으로 이보다 가까우면 반대쪽으로 살짝 밀어냄
SEE = 0.6          # 앞쪽 이 거리 안에 들어오면 피하기 시작
STOP = 0.22        # 앞이 이만큼 가까우면 전진 멈추고 제자리 회전


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Avoider:
    """광선 거리로 갈 방향을 골라 v8 속도·회전·고개 명령을 만든다. 목표 지점이 있으면 그쪽, 없으면 처음 방향으로."""

    def __init__(self, sim, heading=None, goal=None):
        self.sim = sim
        self.heading = sim.pose()[1] if heading is None else heading
        self.goal = goal
        self.head_yaw = 0.0
        self.event, self.event_t = None, -9.0
        self.avoiding = False
        self.side = 0
        self.target = 0.0
        self.clear_since = None

    def step(self, vx_cruise=0.3):
        sim = self.sim
        p, yaw, _ = sim.pose()
        scan = sim.range_scan(angles_deg=np.arange(-82.5, 83, 7.5))
        scan_max = 1.0
        ang = np.array([a for a, _ in scan]); dist = np.array([d for _, d in scan])
        if self.goal is not None:
            dg = self.goal - p[:2]
            gb = _wrap(math.atan2(dg[1], dg[0]) - yaw)
        else:
            gb = _wrap(self.heading - yaw)
        near_i = int(np.argmin(dist)); near_a, near_d = ang[near_i], dist[near_i]
        # 각 후보 방향의 여유 = 그 방향으로 폭 2×CORRIDOR 의 통로를 그렸을 때, 통로 안에 들어오는 가장 가까운 장애물까지 거리
        free = []
        for a in ang:
            dlt = ang - a
            lat = dist * np.abs(np.sin(dlt)); lon = dist * np.cos(dlt)
            inside = (np.abs(dlt) < math.radians(80)) & (lat < CORRIDOR) & (lon > 0.1) & (dist < scan_max)
            free.append(lon[inside].min() if inside.any() else scan_max)
        free = np.array(free)
        front = free[int(np.argmin(np.abs(ang)))]  # 정면 통로 안 가장 가까운 장애물
        # 비용 = 목표 방향에서 벗어난 정도 + 막힌 정도 (+ 정한 쪽과 반대면 벌점: 좌우 왔다 갔다 방지)
        cost = np.abs(ang - gb) + 4.0 * np.maximum(0.0, CLEAR - free)
        if self.side:
            cost += 0.6 * (np.sign(ang) == -self.side)
        target_raw = float(ang[int(np.argmin(cost))])
        gb_free = free[int(np.argmin(np.abs(ang - _clip(gb, ang.max()))))]
        blocked = front < SEE or gb_free < CLEAR
        if blocked:
            self.clear_since = None
            if not self.avoiding:
                self.avoiding = True
                self.side = 1 if target_raw > 0 else -1
                self._say("장애물 발견 → " + ("왼쪽" if self.side > 0 else "오른쪽") + "으로 피하기")
        elif self.avoiding:
            self.clear_since = self.clear_since if self.clear_since is not None else sim.t
            if sim.t - self.clear_since > 1.0:
                self.avoiding, self.side = False, 0
                self._say("지나감 → 원래 방향으로")
        self.target += 0.25 * (target_raw - self.target)
        target = self.target
        wz = _clip(1.3 * target, 1.0)
        if abs(target) > 0.2 and abs(wz) < 0.3:
            wz = math.copysign(0.3, target)
        if abs(target) <= 0.08:
            wz = 0.0
        # 옆에 바짝 붙으면 반대쪽으로 회전 보정 (모서리 스치기 방지)
        for sgn in (1, -1):
            m_ = (np.sign(ang) == sgn) & (np.abs(ang) > math.radians(35))
            dmin = float(dist[m_].min()) if m_.any() else scan_max
            if dmin < SIDE:
                wz = _clip(wz - sgn * 1.2 * (SIDE - dmin) / SIDE, 1.0)
        vx = vx_cruise * min(1.0, max(0.0, (front - STOP) / 0.35))
        if front > STOP:
            vx = max(vx, 0.18)
        if abs(target) > 0.9:
            vx = min(vx, 0.1)
        # 고개: 가까운 장애물을 흘끗, 아니면 가는 방향
        look = near_a if near_d < SEE else target
        self.head_yaw += 0.1 * (_clip(look, 1.0) - self.head_yaw)
        if os.environ.get("AVOID_DEBUG") and int(sim.t / sim.dt) % 25 == 0:
            print(f"  [avoid] t={sim.t:5.1f} front={front:.2f} gb={gb:+.2f} target={target:+.2f} side={self.side} vx={vx:.2f} wz={wz:+.2f} free={np.round(free,2).tolist()}", flush=True)
        return Cmd(float(vx), 0.0, float(wz), head=(0.0, 0.15, float(self.head_yaw), 0.0)), front

    def _say(self, msg):
        self.event, self.event_t = msg, self.sim.t


class Walk(Controller):
    """걷기. 앞으로 걷는 중 장애물이 SEE 안에 들어오면 자동으로 피한다(처음 방향 유지)."""

    def __init__(self, b, sim):
        super().__init__(b, sim)
        self.av = Avoider(sim) if getattr(sim, "obstacles", None) and b.params.get("vx", 0) > 0 \
            and not b.params.get("wz") else None

    @property
    def event(self):
        return self.av.event if self.av and self.sim.t - self.av.event_t < 2.0 else None

    def cmd(self):
        p = self.b.params
        if self.av:
            c, front = self.av.step(p.get("vx", 0.3))
            if self.av.avoiding or front < SEE:
                return c
            c.head = self.head
            return c
        return Cmd(p.get("vx", 0.0), p.get("vy", 0.0), p.get("wz", 0.0), head=self.head)


class AvoidGo(Controller):
    """'장애물 피해서 가': 지금 방향으로 GOAL_D m 앞 지점을 목표로, 장애물을 피해 가서 멈춘다."""
    GOAL_D = 3.0

    def __init__(self, b, sim):
        super().__init__(b, sim)
        p, yaw, _ = sim.pose()
        yaw = getattr(sim, "course_heading", yaw)   # 코스가 있으면 코스 방향으로
        d = b.params.get("goal_d", self.GOAL_D)
        self.goal = p[:2] + d * np.array([math.cos(yaw), math.sin(yaw)])
        self.av = Avoider(sim, goal=self.goal)
        self.arrived = False

    @property
    def event(self):
        if self.arrived:
            return "도착!"
        return self.av.event if self.sim.t - self.av.event_t < 2.0 else None

    def cmd(self):
        p = self.sim.pose()[0]
        if self.arrived or np.linalg.norm(self.goal - p[:2]) < 0.15:
            self.arrived = True
            return Cmd()
        return self.av.step()[0]


class Stop(Controller):
    pass


class Head(Controller):
    def cmd(self):
        if self.b.params.get("nod"):
            return Cmd(head=(0, 0.45 * max(0.0, math.sin(2 * math.pi * 1.2 * max(0.0, self.tau))), 0, 0))
        # 부드럽게 0.4초에 걸쳐 목표 자세로 (알아들음 신호 동안은 tau < 0 → 정면 유지)
        a = min(1.0, max(0.0, self.tau) / 0.4)
        return Cmd(head=tuple(a * h for h in self.head))


class GetUp(Controller):
    """시연용: 덕을 엎어 놓은 상태에서 시작. 일어나는 동작 자체는 v8 정책이 이미 배운 것."""
    def __init__(self, b, sim, knock=True):
        super().__init__(b, sim)
        if knock:
            sim.knock_over(b.params.get("face", "down"))


class Dance(Controller):
    """안무 = 정책이 이미 아는 명령들을 박자에 맞춰 바꾸기.
    몸 기울이기(body roll/pitch)는 서 있을 때만 먹고 학습 범위가 ±12°라 그 안에서만 씀."""
    LENGTH = 14.0

    def cmd(self):
        t = self.tau
        if t < 0:       # 알아들음 신호(고개 들기)가 끝날 때까지 가만히
            return Cmd()
        s = math.sin(2 * math.pi * 0.5 * t)
        if t < 4.0:     # 좌우로 몸 흔들기 + 고개 갸웃 (반대 방향)
            return Cmd(head=(0, 0.1, 0.6 * s, -0.3 * s), body=(0, 0, 0, 0.2 * s, 0, 0))
        if t < 7.0:     # 앞뒤로 까딱 + 빠른 고개 끄덕
            return Cmd(head=(0.1, 0.5 * abs(math.sin(2 * math.pi * 1.0 * t)), 0, 0),
                       body=(0, 0, 0, 0, 0.15 * math.sin(2 * math.pi * 1.0 * t), 0))
        if t < 9.5:     # 제자리에서 왼쪽으로 빙글
            return Cmd(wz=1.0, head=(0, 0, 0.8, 0.2))
        if t < 12.0:    # 옆걸음 왼쪽 → 오른쪽 + 고개 도리도리
            vy = 0.2 if t < 10.75 else -0.2
            return Cmd(vy=vy, head=(0, 0, 0.9 * math.sin(2 * math.pi * 1.2 * t), 0))
        return Cmd(wz=-1.0, head=(0, 0, -0.8, -0.2))  # 오른쪽으로 빙글 마무리


class FollowBall(Controller):
    """공 위치(시뮬레이터가 알려 줌; 실제 로봇에선 카메라 검출로 대체)를 보고 방향·속도·고개 명령을 만든다."""
    STOP_D = 0.22   # 이만큼 가까우면 멈추고 공을 내려다봄
    HEAD_H = 0.17   # 눈 높이(대략)

    def __init__(self, b, sim, ball_plan=None):
        super().__init__(b, sim)
        p, yaw, _ = sim.pose()
        # 공을 덕의 왼쪽 앞 0.55 m 에 놓고, 정해진 길을 따라 천천히 굴린다
        rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        self.rot = rot
        sim.set_ball(p[:2] + rot @ np.array([0.65, 0.3]), (0, 0))
        # (시작 s, 끝 s, 덕 기준 속도 vx, vy)
        # 처음 덕 방향 기준. 덕(실제 약 0.13 m/s)이 따라올 수 있게 공은 0.06–0.09 m/s 로 굴림
        self.plan = ball_plan or [(0.0, 2.5, 0.0, 0.0), (2.5, 9.0, 0.07, -0.06), (9.0, 15.0, -0.03, -0.08),
                                  (15.0, 19.0, -0.07, 0.0), (19.0, 99.0, 0.0, 0.0)]
        self.head_yaw = 0.0

    def cmd(self):
        sim, t = self.sim, self.tau
        v = np.zeros(2)
        for a, b_, vx, vy in self.plan:
            if a <= t < b_:
                v = self.rot @ np.array([vx, vy])
        sim.ball_v = v
        p, yaw, _ = sim.pose()
        d_w = sim.ball_xy - p[:2]
        dist = float(np.linalg.norm(d_w))
        bearing = math.atan2(d_w[1], d_w[0]) - yaw
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        # 고개: 공 쪽으로 (부드럽게)
        self.head_yaw += 0.15 * (_clip(bearing, 1.2) - self.head_yaw)
        pitch_down = _clip(math.atan2(self.HEAD_H, max(dist, 0.05)) * 1.2, 0.9)
        head = (0.0, pitch_down, self.head_yaw, 0.0)
        if dist < self.STOP_D and abs(bearing) < 0.5:
            return Cmd(head=head)
        wz = _clip(1.8 * bearing, 1.0)
        if abs(wz) < 0.25 and abs(bearing) > 0.12:   # v8 은 0.3 rad/s 이하 제자리 회전이 약함
            wz = math.copysign(0.3, bearing)
        vx = 0.3 if abs(bearing) < 0.5 else (0.12 if abs(bearing) < 1.0 else 0.0)
        if dist < 0.35:
            vx = min(vx, 0.18)
        return Cmd(vx, 0.0, wz if abs(bearing) > 0.06 or vx == 0 else 0.0, head=head)


# ── 알아들음 신호: 명령을 알아들으면 고개를 잠깐 든다 ──────────────────────────
# head_pitch 는 + 가 아래, - 가 위. v8 의 고개 명령만 쓰므로 정책은 그대로.
ACK_HEAD_UP = (-0.3, -0.75, 0.0, 0.0)   # neck_pitch, head_pitch, head_yaw, head_roll
ACK_S = 0.9                             # 올리기 0.3 s → 유지 0.3 s → 내리기 0.3 s
ACK_LABEL = "알아들었어요 → 고개 들기"


def _ack_weight(u: float) -> float:
    """u = 신호 시작 후 경과 초. 0..1 의 부드러운(코사인) 사다리꼴."""
    r = ACK_S / 3
    if u < 0 or u > ACK_S:
        return 0.0
    if u < r:
        return 0.5 - 0.5 * math.cos(math.pi * u / r)
    if u > 2 * r:
        return 0.5 - 0.5 * math.cos(math.pi * (ACK_S - u) / r)
    return 1.0


class Acknowledged(Controller):
    """모든 행동을 감싸는 공통 층: 알아들음 신호(고개 들기)를 행동 명령 위에 섞는다.
    - 고개·춤: 신호가 끝난 뒤 행동 시작 (행동 시계를 ACK_S 만큼 늦춤)
    - 일어나기: 넘어진 덕이 다시 선 직후(0.3 s 안정) 신호
    - 그 밖(걷기·돌기·멈춤·공 따라가기): 행동을 시작하면서 동시에 신호, 끝나면 원래 고개 명령으로 복귀"""

    def __init__(self, inner: Controller):
        self.inner, self.sim, self.b = inner, inner.sim, inner.b
        self.t0 = inner.t0
        self.ack_t0 = None if self.b.kind == "getup" else self.sim.t
        self._upright_since = None
        # 고개·춤·Pollen 원샷 정책: 신호가 끝난 뒤 시작 (앉아/일어서는 sitstand 가 고개를 받으므로 동시에)
        if self.b.kind in ("head", "dance", "ground_pick", "roller_crouch", "roulade", "kick", "alpha_stand", "alpha_walk", "beak_pick"):
            inner.t0 += ACK_S

    @property
    def ack_weight(self) -> float:
        return 0.0 if self.ack_t0 is None else _ack_weight(self.sim.t - self.ack_t0)

    @property
    def event(self):
        return getattr(self.inner, "event", None)

    @property
    def ack_active(self) -> bool:
        return self.ack_t0 is not None and 0.0 <= self.sim.t - self.ack_t0 <= ACK_S

    def cmd(self) -> Cmd:
        c = self.inner.cmd()
        if self.ack_t0 is None:   # 일어나기: 선 상태가 0.3 s 이어지면 신호 시작
            if self.sim.pose()[2] < -0.9 and self.sim.t - self.t0 > 0.1:
                self._upright_since = self._upright_since or self.sim.t
                if self.sim.t - self._upright_since >= 0.3:
                    self.ack_t0 = self.sim.t
            else:
                self._upright_since = None
        w = self.ack_weight
        if w > 0:
            c.head = tuple((1 - w) * h + w * u for h, u in zip(c.head, ACK_HEAD_UP))
        return c


def make(b: Behavior, sim: DuckSim, ack: bool = True) -> Controller:
    import policies
    table = {"walk": Walk, "stop": Stop, "head": Head, "getup": GetUp, "dance": Dance,
             "follow_ball": FollowBall, "avoid": AvoidGo, **policies.KINDS}
    if b.kind != "kick":
        sim.ball_free = False
    inner = table[b.kind](b, sim)
    return Acknowledged(inner) if ack and b.params.get("ack", True) else inner
