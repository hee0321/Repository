"""행동 컨트롤러: Behavior → 매 스텝(50 Hz) 정책 명령(Cmd). 새 학습 없이 v8 정책의 명령 입력만 조종합니다."""
from __future__ import annotations
import math
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


class Walk(Controller):
    def cmd(self):
        p = self.b.params
        return Cmd(p.get("vx", 0.0), p.get("vy", 0.0), p.get("wz", 0.0), head=self.head)


class Stop(Controller):
    pass


class Head(Controller):
    def cmd(self):
        if self.b.params.get("nod"):
            return Cmd(head=(0, 0.45 * max(0.0, math.sin(2 * math.pi * 1.2 * self.tau)), 0, 0))
        # 부드럽게 0.4초에 걸쳐 목표 자세로
        a = min(1.0, self.tau / 0.4)
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


def make(b: Behavior, sim: DuckSim) -> Controller:
    return {"walk": Walk, "stop": Stop, "head": Head, "getup": GetUp, "dance": Dance,
            "follow_ball": FollowBall}[b.kind](b, sim)
