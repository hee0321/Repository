"""롤러 모드 한국어 명령 → roller 정책 명령 (규칙 기반, 무료·오프라인). 걷기 모드 파서(korean_commands.py)와 따로.

  롤러 타 / 앞으로 밀어 / 밀어      → 밀기 cmd_x = PUSH (0.4)
  더 빨리 / 세게 밀어 / 빨리        → 세게 밀기 cmd_x = FAST (0.6, 학습 범위 끝)
  천천히 / 살살                      → 살살 밀기 cmd_x = SLOW (0.3)
  미끄러져 / 힘 빼                   → cmd_x = 0 ('coast' 자리지만 공개 roller.onnx 는 1초 안에 거의 섬 — 시뮬에서 잼)
  멈춰 / 브레이크 / 서               → 브레이크 cmd_x = BRAKE (-0.4), 거의 서면 0
  왼쪽으로 / 오른쪽으로              → 목표 방향 ±TURN rad 바꾸고 밀기 (cmd[2] = 방향 오차 × 게인)
  똑바로                             → 지금 방향을 목표로
  웅크려 / 숙여                      → roller_crouch.onnx (3.5초, 낮게 미끄러지기) → 다시 미끄러지기
"""
from __future__ import annotations
import re
from dataclasses import dataclass

PUSH, FAST, SLOW, BRAKE, TURN = 0.4, 0.6, 0.3, -0.4, 0.5   # 0.2 이하는 공개 정책이 그냥 섬 (시뮬에서 잼)


@dataclass
class RollerCmd:
    kind: str        # push | coast | brake | turn | straight | crouch
    label: str
    cmd_x: float = 0.0
    turn: float = 0.0


WAKE = re.compile(r"^\s*(덕|더|떡|턱)\s*(아|가|야)[,.!]?\s*")
RULES = [
    (r"웅크|숙여|쪼그", lambda: RollerCmd("crouch", "웅크리기 (Pollen roller_crouch.onnx, 3.5초)")),
    (r"멈춰|멈추|브레이크|정지|서\s*$|그만", lambda: RollerCmd("brake", f"브레이크 (cmd_x {BRAKE}) → 거의 서면 0", BRAKE)),
    (r"천천|살살", lambda: RollerCmd("push", f"살살 밀기 (cmd_x {SLOW})", SLOW)),
    (r"미끄러|그냥\s*가|힘\s*빼", lambda: RollerCmd("coast", "힘 빼기 (cmd_x 0 → 정책이 곧 멈춤)", 0.0)),
    (r"왼쪽", lambda: RollerCmd("turn", f"왼쪽으로 {round(TURN*57.3)}° 방향 바꾸며 밀기", PUSH, +TURN)),
    (r"오른쪽", lambda: RollerCmd("turn", f"오른쪽으로 {round(TURN*57.3)}° 방향 바꾸며 밀기", PUSH, -TURN)),
    (r"똑바로|직진", lambda: RollerCmd("straight", "똑바로 (지금 방향 유지하며 밀기)", PUSH)),
    (r"빨리|세게|힘껏", lambda: RollerCmd("push", f"세게 밀기 (cmd_x {FAST})", FAST)),
    (r"롤러|스케이트|밀어|앞으로|타", lambda: RollerCmd("push", f"밀어서 타기 (cmd_x {PUSH})", PUSH)),
]


def parse_roller(text: str) -> RollerCmd | None:
    t = WAKE.sub("", text.strip())
    for pat, mk in RULES:
        if re.search(pat, t):
            return mk()
    return None


if __name__ == "__main__":
    for s in ["덕아, 롤러 타", "앞으로 밀어", "더 빨리", "천천히", "왼쪽으로", "오른쪽으로 가", "웅크려", "멈춰", "브레이크", "똑바로 가", "노래해"]:
        print(s, "→", parse_roller(s))
