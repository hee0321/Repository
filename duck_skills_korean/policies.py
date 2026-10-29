"""Pollen 공개 정책(pollen-robotics/microduck-policies)을 명령에 따라 바꿔 끼우기. 학습 없음, CPU(onnxruntime).

모든 정책이 같은 약속을 씀 (manifest.json: obs_len 61, action_len 14, 50 Hz, action_scale 1.0, 같은 기본 자세):
  관측 = 각속도3 · 중력방향3 · 관절위치14 · 관절속도14 · 직전행동14 · twist3 · 고개4 · 몸6
명령 자리(twist)의 뜻은 정책마다 다름 — manifest 의 command 항목 그대로:
  sitstand     : twist.vx = 1 앉기 / 0 서기, 2초에 걸쳐 바꾸고(ramp_s) 1초 유지(unwind_s), 고개 명령 가능
  ground_pick  : twist.vx,vy = cos/sin(2π·위상), 주기 4초, 위상 0.7(2.8초)에서 끝
  roller_crouch: 같은 위상 방식, 주기 5초, 3.5초 — 원래는 '롤러스케이트 신은 덕'용 (mode: roller)
  roulade      : 명령 없음, 1초 (구르기 시작만 하고 나머지는 다음 정책이 받음, chain)
  kick_left/right: 명령 없음, 0.5초, 공이 차는 발 앞에 있어야 함
  alpha_stand  : 일어나기 선생 정책(넘어진 상태에서 서기), alpha_walking : 옛 걷기 정책
"""
from __future__ import annotations
import math
import numpy as np
from duck_sim import Cmd
from behaviors import Controller
from beak import BEAK_OPEN

REPO = "pollen-robotics/microduck-policies"
FILES = {
    "sitstand": "alpha_sitstand.onnx", "ground_pick": "alpha_ground_pick.onnx",
    "roller_crouch": "roller_crouch.onnx", "roulade": "roulade.onnx",
    "kick_left": "ball_kick_left.onnx", "kick_right": "ball_kick_right.onnx",
    "alpha_stand": "alpha_stand.onnx", "alpha_walking": "alpha_walking.onnx", "roller": "roller.onnx",
}
# (고개 명령 사용, 몸 명령 사용) — ONNX 메타데이터 command_names 기준
USES = {"sitstand": (True, False), "ground_pick": (False, False), "roller_crouch": (False, False),
        "roulade": (False, False), "kick_left": (False, False), "kick_right": (False, False),
        "alpha_stand": (True, True), "alpha_walking": (True, True), "roller": (False, False)}
_SESS = {}


def session(name):
    if name not in _SESS:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        _SESS[name] = ort.InferenceSession(hf_hub_download(REPO, FILES[name]), providers=["CPUExecutionProvider"])
    return _SESS[name]


class Skill(Controller):
    """한 번 하는 동작: 지정 시간 동안 해당 ONNX 정책을 돌리고, 끝나면 v8(velstand)로 돌려준다."""
    policy = ""
    duration = 1.0
    after_s = 0.0       # 끝난 뒤 v8 로 돌아와 '서 있기' 명령

    def twist(self, t):
        return (0.0, 0.0, 0.0)

    def prepare(self):
        pass

    def __init__(self, b, sim):
        super().__init__(b, sim)
        self.prepared = False
        self.done = False

    @property
    def event(self):
        t = self.tau
        if 0 <= t < self.duration:
            return f"정책 바꿈 → {FILES[self.policy]}"
        if t >= self.duration and t < self.duration + 1.5:
            return "끝 → v8 velstand 로 복귀"
        return None

    def cmd(self):
        t = self.tau
        if t < 0:                       # 알아들음 신호(고개 들기) 동안은 v8 로 서 있기
            return Cmd()
        if not self.prepared:
            self.prepare(); self.prepared = True
        if t < self.duration:
            vx, vy, wz = self.twist(t)
            return Cmd(vx, vy, wz, policy=self.policy)
        self.done = True
        return Cmd()


def _phase(t, period):
    ph = t / period
    return (math.cos(2 * math.pi * ph), math.sin(2 * math.pi * ph), 0.0)


class GroundPick(Skill):
    policy, duration = "ground_pick", 2.8

    def prepare(self):   # 바닥에 작은 물건(2 cm 파란 상자)을 입 아래 놓기 — 이 정책은 숙이기만 함 (집기 없음)
        from duck_sim import PICK_HALF
        p, yaw, _ = self.sim.pose()
        if getattr(self.sim, "pickup", None) is None:
            return
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        xy = p[:2] + R @ np.array(BeakPick.SPOT)
        self.sim.release()
        self.sim.place_pickup(float(xy[0]), float(xy[1]), PICK_HALF, (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))

    def twist(self, t):
        return _phase(t, 4.0)


class RollerCrouch(Skill):
    policy, duration = "roller_crouch", 3.5

    def twist(self, t):
        return _phase(t, 5.0)


class Roulade(Skill):
    policy, duration = "roulade", 1.0


class Kick(Skill):
    duration = 0.5

    def __init__(self, b, sim):
        super().__init__(b, sim)
        self.policy = "kick_right" if b.params.get("foot") == "right" else "kick_left"

    def prepare(self):
        # 학습 환경과 같은 자리: 차는 발 앞 (앞 0.09 m, 옆 ±0.042 m), 지름 7 cm 공
        p, yaw, _ = self.sim.pose()
        oy = -0.042 if self.policy == "kick_right" else 0.042
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        self.sim.ball_free = True
        self.sim.set_ball(p[:2] + R @ np.array([0.09, oy]), (0, 0))
        self.sim.kick_ball0 = self.sim.ball_xy.copy()   # 공이 얼마나 굴러갔는지 재기용


class AlphaStand(Skill):
    """일어나기 선생 정책으로 일어나기 (비교용): 엎어 놓고 3초 alpha_stand → v8."""
    policy, duration = "alpha_stand", 3.0

    def prepare(self):
        self.sim.knock_over("down")


class AlphaWalk(Skill):
    policy, duration = "alpha_walking", 6.0

    def twist(self, t):
        return (self.b.params.get("vx", 0.3), 0.0, 0.0)


class SitStand(Controller):
    """앉아/일어서: sitstand 정책을 계속 돌림 (앉아 있는 동안 고개 명령 가능 → 알아들음 신호도 보임)."""

    def __init__(self, b, sim):
        super().__init__(b, sim)
        self.sit = b.params.get("sit", True)
        self.from_flag = getattr(sim, "sit_flag", 0.0)

    @property
    def event(self):
        return "정책 바꿈 → alpha_sitstand.onnx" if self.tau < 2.5 else None

    def cmd(self):
        t = max(0.0, self.tau)
        target = 1.0 if self.sit else 0.0
        a = min(1.0, t / 2.0)           # ramp_s 2초
        flag = self.from_flag + (target - self.from_flag) * a
        self.sim.sit_flag = flag
        if not self.sit and t > 3.0:    # 일어선 뒤 1초 더(unwind) → v8 로 복귀
            self.sim.sit_flag = 0.0
            return Cmd(head=self.head)
        return Cmd(flag, 0.0, 0.0, head=self.head, policy="sitstand")


class CrouchSubstitute(Controller):
    """'웅크려' (걷는 덕 모델): roller_crouch.onnx 는 롤러스케이트 덕 전용이라 걷는 모델에선 넘어짐(시험함).
    그래서 비슷한 동작을 alpha_sitstand.onnx 로 대신: 2초에 걸쳐 쪼그려 앉기 → 1.5초 유지 → 일어서기 → v8.
    진짜 roller_crouch 는 roller_crouch_clip.py (롤러 덕 모델) 영상에 따로 있음."""

    def __init__(self, b, sim):
        super().__init__(b, sim)
        self.done = False

    @property
    def event(self):
        t = self.tau
        if t < 0:
            return None
        if t < 3.5:
            return "대신 alpha_sitstand.onnx 로 쪼그리기 (roller_crouch 는 롤러 모델 전용)"
        if t < 7.5:
            return "다시 일어서기 → v8 복귀"
        return None

    def cmd(self):
        t = self.tau
        if t < 0:
            return Cmd()
        if t < 3.5:
            flag = min(1.0, t / 2.0)
        elif t < 6.5:
            flag = max(0.0, 1.0 - (t - 3.5) / 2.0)
        else:
            self.done = True
            return Cmd()
        self.sim.sit_flag = flag
        return Cmd(flag, 0.0, 0.0, policy="sitstand")


class BeakPick(Controller):
    """'부리로 물어' / '물어와': 부리 벌리기 → ground_pick 정책으로 숙이기 → 부리 닫기 → v8 로 서기 (물고 있음)
    → (물어와) 제자리 돌아서 0.5 m 걸어와 멈춤.

    정직하게: 부리 경첩·액추에이터는 진짜 관절이지만, 물건을 '무는' 건 물리 마찰 집기가 아님.
    부리를 닫는 순간 물건을 부리(아랫부리) 좌표에 운동학적으로 붙임(sim.grab) — 물건은 질량 없는 보기용 물체.
    """
    OPEN_S = 0.6          # v8 로 서서 부리 벌리기
    PICK = 2.8            # ground_pick 정책 시간 (manifest)
    CLOSE_AT = 1.25       # ground_pick 안에서 부리 닫기 시작 (입이 가장 낮을 때 ≈ 1.3–1.5 s)
    CLOSE_S = 0.25
    HOLD = 0.42           # 물건(2 cm)을 문 채 부리 각도
    # 물건이 놓일 자리 (덕 기준 앞 x, 왼쪽 y) — 숙였을 때 부리 가운데가 오는 곳 (시뮬에서 잼)
    SPOT = (0.062, 0.006)

    def __init__(self, b, sim):
        super().__init__(b, sim)
        self.fetch = bool(b.params.get("fetch"))
        self.done = False
        self.grabbed = False
        self.yaw0 = None
        p, yaw, _ = sim.pose()
        from duck_sim import PICK_HALF
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        xy = p[:2] + R @ np.array(self.SPOT)
        sim.release()
        sim.place_pickup(float(xy[0]), float(xy[1]), PICK_HALF, (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))

    @property
    def event(self):
        t = self.tau
        if t < 0:
            return None
        if t < self.OPEN_S:
            return "부리 벌리기"
        tp = t - self.OPEN_S
        if tp < self.CLOSE_AT:
            return f"정책 바꿈 → {FILES['ground_pick']} (숙이기)"
        if tp < self.PICK:
            return "부리 닫기 → 물었어요"
        if tp < self.PICK + 1.5:
            return "v8 로 서기 → 물고 있어요"
        if self.fetch and not self.done:
            return "돌아서 가져오기"
        return "물고 있어요" if self.grabbed else None

    def cmd(self):
        t = self.tau
        if t < 0:
            return Cmd(beak=0.0)
        if t < self.OPEN_S:
            return Cmd(beak=BEAK_OPEN * min(1.0, t / 0.35))
        tp = t - self.OPEN_S
        if tp < self.PICK:
            if tp < self.CLOSE_AT:
                bk = BEAK_OPEN
            else:
                a = min(1.0, (tp - self.CLOSE_AT) / self.CLOSE_S)
                bk = BEAK_OPEN + (self.HOLD - BEAK_OPEN) * a
                if a >= 1.0 and not self.grabbed:
                    self.sim.grab(); self.grabbed = True
            vx, vy, _ = _phase(tp, 4.0)
            return Cmd(vx, vy, 0.0, policy="ground_pick", beak=bk)
        ts = tp - self.PICK
        head = (0.0, -0.25, 0.0, 0.0)        # 살짝 고개 들어 문 물건 보여주기
        if not self.fetch or ts < 1.5:
            self.done = not self.fetch and ts > 1.5
            return Cmd(head=head, beak=self.HOLD)
        # 물어와: 180° 돌기 → 0.5 m 걸어오기
        _, yaw, _ = self.sim.pose()
        if self.yaw0 is None:
            self.yaw0 = yaw; self.turned = False; self.p0 = None
        if not self.turned:
            err = math.atan2(math.sin(self.yaw0 + math.pi - yaw), math.cos(self.yaw0 + math.pi - yaw))
            if abs(err) > 0.12:
                return Cmd(0.0, 0.0, 1.2 if err > 0 else -1.2, head=head, beak=self.HOLD)
            self.turned = True; self.p0 = self.sim.pose()[0][:2].copy()
        if np.linalg.norm(self.sim.pose()[0][:2] - self.p0) < 0.5:
            return Cmd(0.25, 0.0, 0.0, head=head, beak=self.HOLD)
        self.done = True
        return Cmd(head=head, beak=self.HOLD)


KINDS = {"beak_pick": BeakPick, "ground_pick": GroundPick, "roller_crouch": CrouchSubstitute, "roulade": Roulade, "kick": Kick,
         "alpha_stand": AlphaStand, "alpha_walk": AlphaWalk, "sitstand": SitStand}
