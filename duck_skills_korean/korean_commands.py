"""한국어 문장 → 덕 행동(behavior) 파서. 규칙 기반이라 무료, 오프라인, 즉시 동작.

../microduck_korean_chat/duckchat/intent.py 의 규칙(실제 로봇 robotd 명령)을 확장해서,
시뮬레이션의 v8 정책 명령(속도·고개·몸 자세)과 '춤', '공 따라가기' 같은 상위 행동으로 바꿉니다.
음성은 같은 폴더의 STT(faster-whisper)로 글자로 바꾼 뒤 이 파서에 넣으면 됩니다.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field

FWD_VX = 0.3      # 명령 속도 (v8 정책은 실제로 약 0.1–0.15 m/s 로 걸음)
BACK_VX = -0.2
SIDE_VY = 0.2
TURN_WZ = 0.9     # rad/s (v8 시뮬 평가: 1.0 명령 → 0.9 내외)
HEAD_YAW = 0.9    # rad, + = 왼쪽
HEAD_UP = -0.5    # head_pitch, - = 위
HEAD_DOWN = 0.6   # head_pitch, + = 아래


@dataclass
class Behavior:
    kind: str               # walk | stop | head | getup | dance | follow_ball
    label: str              # 화면에 보여 줄 한국어 설명
    params: dict = field(default_factory=dict)


WAKE = re.compile(r"^\s*(안녕[,.!]?\s*)?(덕|더|떡|턱)\s*(아|가|야)[,.!]?\s+(?=\S)")

RULES: list[tuple[str, callable]] = [
    (r"공.*(따라|쫓|쫒|좇|찾|가)|따라\s*가.*공", lambda t: Behavior("follow_ball", "공 따라가기 (공 위치 → 방향·속도·고개 명령)")),
    (r"춤|댄스|흔들어|신나게", lambda t: Behavior("dance", "춤추기 (몸 기울이기·고개·제자리 돌기 안무)")),
    (r"일어나|일어서|기립|일어나라", lambda t: Behavior("getup", "일어나기 (넘어지면 정책이 스스로 일어남)")),
    (r"멈춰|정지|그만|스톱|가만", lambda t: Behavior("stop", "멈추기 (속도 0, 고개 정면)")),
    (r"(고개|머리|목).*(왼|좌)|(왼|좌)\S*\s*(봐|보|쳐다)", lambda t: Behavior("head", "고개 왼쪽", {"head": (0, 0, HEAD_YAW, 0)})),
    (r"(고개|머리|목).*(오른|우)|(오른|우)\S*\s*(봐|보|쳐다)", lambda t: Behavior("head", "고개 오른쪽", {"head": (0, 0, -HEAD_YAW, 0)})),
    (r"(고개|머리|목).*(위|들)|위\S*\s*(봐|보|쳐다)|하늘", lambda t: Behavior("head", "고개 위로", {"head": (-0.2, HEAD_UP, 0, 0)})),
    (r"(고개|머리|목).*(아래|숙)|아래\S*\s*(봐|보|쳐다)|바닥", lambda t: Behavior("head", "고개 아래로", {"head": (0.2, HEAD_DOWN, 0, 0)})),
    (r"끄덕|인사", lambda t: Behavior("head", "고개 끄덕", {"nod": True, "head": (0, 0, 0, 0)})),
    (r"(고개|머리|정면).*(앞|정면|바로)|앞\S*\s*봐", lambda t: Behavior("head", "고개 정면", {"head": (0, 0, 0, 0)})),
    (r"(왼쪽|좌)\s*(으로)?\s*(돌|회전)", lambda t: Behavior("walk", "왼쪽으로 돌기", {"wz": TURN_WZ})),
    (r"(오른쪽|우)\s*(으로)?\s*(돌|회전)", lambda t: Behavior("walk", "오른쪽으로 돌기", {"wz": -TURN_WZ})),
    (r"빙글|한\s*바퀴|돌아", lambda t: Behavior("walk", "제자리 돌기", {"wz": TURN_WZ})),
    (r"뒤로|후진|물러", lambda t: Behavior("walk", "뒤로 걷기", {"vx": BACK_VX})),
    (r"왼쪽\s*(으로)?\s*(가|걸|이동|옆)", lambda t: Behavior("walk", "왼쪽 옆걸음", {"vy": SIDE_VY})),
    (r"오른쪽\s*(으로)?\s*(가|걸|이동|옆)", lambda t: Behavior("walk", "오른쪽 옆걸음", {"vy": -SIDE_VY})),
    (r"앞으로|전진|걸어|이리\s*와|와\s*봐|가\s*자|가$", lambda t: Behavior("walk", "앞으로 걷기", {"vx": FWD_VX})),
]


def parse(text: str) -> Behavior | None:
    t = WAKE.sub(lambda m: m.group(1) or "", text, count=1).strip()
    for pat, fn in RULES:
        if re.search(pat, t):
            return fn(t)
    return None


CANON = ["앞으로 가", "뒤로 가", "멈춰", "왼쪽으로 돌아", "오른쪽으로 돌아", "고개 왼쪽", "고개 오른쪽", "고개 들어",
         "아래 봐", "일어나", "춤춰", "공 따라가"]


def parse_fuzzy(text: str, cutoff: float = 0.5) -> Behavior | None:
    """규칙에 안 걸린 말(음성 인식 오타 등)을 가장 비슷한 기본 명령으로. 너무 다르면 None."""
    import difflib
    t = re.sub(r"[^가-힣 ]", "", text).strip()
    if not t:
        return None
    sc = sorted(((difflib.SequenceMatcher(None, t.replace(" ", ""), c.replace(" ", "")).ratio(), c) for c in CANON),
                reverse=True)
    (r, best), (r2, _) = sc[0], sc[1]
    # 애매하면(예: '고개'만 들림 → 왼쪽/오른쪽 동점) 움직이지 않고 다시 말해 달라고 하는 게 안전
    return parse(best) if r >= cutoff and r - r2 > 0.05 else None


if __name__ == "__main__":
    import sys
    tests = sys.argv[1:] or ["덕아, 앞으로 가", "멈춰", "왼쪽으로 돌아", "오른쪽으로 돌아", "고개 왼쪽", "오른쪽 봐",
                             "고개 들어", "아래 봐", "일어나", "춤춰", "공 따라가", "덕아 공 쫓아가 봐", "안녕?"]
    for s in tests:
        b = parse(s)
        print(f"{s!r:22} → {b.kind + ' / ' + b.label + ' ' + str(b.params) if b else '(모르는 말)'}")
