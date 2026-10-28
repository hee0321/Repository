"""한국어 문장 → Microduck 명령 (dry-run).

명령 이름은 실제 코드에서 확인한 것만 씁니다 (pollen-robotics/microduck @ 5def2f3,
duck-ipc-proto/src/lib.rs):
  - robot.move  {vx, vy, vyaw}   연속 명령(notification, 20–50 Hz로 계속 보내야 함, 끊기면 deadman이 0으로)
  - robot.stop  {}               속도 0 (힘 빼기 아님)
  - robot.do    {skill}          기본 스킬: kick_left, kick_right, ground_pick, sit_toggle, roulade
  - robot.sound {tag}            alarm, greet, inquire, peck, chirp, coo, wheee
  - robot.mouth {open: 0..1}     부리 열기 (TTS 재생 중 립싱크용)
속도 범위: microduck_rl velocity env cfg 학습 범위 vx ±0.4 m/s, vy ±0.3, vyaw ±1.0 rad/s.
여기서는 안전하게 vx 0.2 (v8 문서 평가 기준값), 회전 0.7 rad/s 를 씁니다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict

FWD_VX = 0.2
BACK_VX = -0.15
SIDE_VY = 0.15
TURN_VYAW = 0.7
DEFAULT_S = 2.0
MAX_S = 10.0


@dataclass
class Command:
    action: str                    # 사람이 읽는 이름 (한국어)
    method: str                    # robotd JSON-RPC 메서드
    params: dict = field(default_factory=dict)
    duration_s: float | None = None  # robot.move 를 몇 초 동안 반복 송신할지
    note: str = ""

    def rpc_lines(self) -> list[str]:
        msg = {"jsonrpc": "2.0", "method": self.method, "params": self.params}
        if self.method != "robot.move":
            msg["id"] = 1  # 이산 명령은 request (응답 받음)
        lines = [json.dumps(msg, ensure_ascii=False)]
        if self.method == "robot.move":
            lines.append(f"# … {self.duration_s:.1f}초 동안 10 Hz로 반복 송신 후")
            lines.append(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "robot.stop", "params": {}}))
        return lines

    def as_dict(self) -> dict:
        return asdict(self)


def _duration(text: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*초", text)
    if m:
        return min(float(m.group(1)), MAX_S)
    kor = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5}
    m = re.search(r"(한|두|세|네|다섯)\s*초", text)
    return float(kor[m.group(1)]) if m else DEFAULT_S


def _move(action, vx=0.0, vy=0.0, vyaw=0.0, text=""):
    return Command(action, "robot.move", {"vx": vx, "vy": vy, "vyaw": vyaw}, _duration(text))


# 우선순위 순서. (정규식, 생성 함수)
RULES: list[tuple[str, callable]] = [
    (r"멈춰|멈춰라|정지|그만|스톱|서\s*있어", lambda t: Command("멈추기", "robot.stop", {})),
    (r"(왼|오른)?\s*(발|쪽)?.*공.*(차|킥)|킥", None),  # 아래 _kick 에서 처리
    (r"(주워|집어|줍|물어)", lambda t: Command("바닥 물건 줍기", "robot.do", {"skill": "ground_pick"},
                                         note="부리로 바닥 물건 집기")),
    (r"구르|굴러|데굴", lambda t: Command("구르기", "robot.do", {"skill": "roulade"})),
    (r"앉아|앉자|앉을래|앉으렴", lambda t: Command("앉기", "robot.do", {"skill": "sit_toggle"},
                                       note="sit_toggle은 토글: 실제로는 robot.state 로 앉아 있지 않을 때만 보냄")),
    (r"일어나|일어서|기립", lambda t: Command("일어서기", "robot.do", {"skill": "sit_toggle"},
                                     note="앉아 있을 때만 sit_toggle. 넘어진 상태의 일어나기는 velstand 정책이 자동으로 함")),
    (r"(왼쪽|좌)\s*(으로)?\s*(돌|회전)", lambda t: _move("왼쪽으로 돌기", vyaw=+TURN_VYAW, text=t)),
    (r"(오른쪽|우)\s*(으로)?\s*(돌|회전)", lambda t: _move("오른쪽으로 돌기", vyaw=-TURN_VYAW, text=t)),
    (r"돌아|돌아라|회전|빙글", lambda t: _move("제자리 돌기(왼쪽)", vyaw=+TURN_VYAW, text=t)),
    (r"뒤로|후진|물러", lambda t: _move("뒤로 걷기", vx=BACK_VX, text=t)),
    (r"왼쪽\s*(으로)?\s*(가|걸|이동|옆)", lambda t: _move("왼쪽 옆걸음", vy=+SIDE_VY, text=t)),
    (r"오른쪽\s*(으로)?\s*(가|걸|이동|옆)", lambda t: _move("오른쪽 옆걸음", vy=-SIDE_VY, text=t)),
    (r"앞으로|전진|걸어|이리\s*와|와\s*봐|가\s*자", lambda t: _move("앞으로 걷기", vx=FWD_VX, text=t)),
    (r"꽥|소리\s*내|울어|인사", lambda t: Command("꽥 소리", "robot.sound", {"tag": "chirp"})),
]


def _kick(text: str) -> Command:
    side = "kick_right" if "오른" in text else "kick_left"
    label = "오른발 공 차기" if side == "kick_right" else "왼발 공 차기"
    return Command(label, "robot.do", {"skill": side}, note="공이 발 앞에 있어야 함")


WAKE = re.compile(r"^\s*(안녕[,.!]?\s*)?(덕|더|떡|턱)\s*(아|가|야)[,.!]?\s+(?=\S)")


def strip_wake(text: str) -> str:
    """'덕아,' 호출어 제거. Whisper가 '덕아'를 '더 가'로 자주 받아써서 '가'가 명령으로 오인되지 않게."""
    return WAKE.sub(lambda m: m.group(1) or "", text, count=1).strip()


def parse(text: str) -> Command | None:
    t = strip_wake(text)
    for pat, fn in RULES:
        if re.search(pat, t):
            if fn is None:
                if "공" in t or "킥" in t:
                    return _kick(t)
                continue
            return fn(t)
    return None


if __name__ == "__main__":
    import sys
    for s in sys.argv[1:] or ["앞으로 가", "왼쪽으로 돌아", "일어나", "앉아", "오른발로 공 차", "3초 동안 뒤로 가", "멈춰", "안녕?"]:
        c = parse(s)
        print(s, "→", c.as_dict() if c else None)
