"""화면 상태(어떤 패널이 크게, 어떤 패널을 보이게)와 키 조작. 그리는 건 hud.py / compose.py.

키(대화형 --interactive):
  m / Tab : 큰 화면 바꾸기 (운전석 → 지도 → 모터룸 → 위성 → 운전석)
  1       : 모터룸 작은 카드 보이기/숨기기 (오른쪽에서 밀려 들어옴)
  2       : 위성 작은 카드 보이기/숨기기
  g       : 운전석 중심 ↔ 2×2 격자
  space   : 일시 정지,  q / Esc : 끝내기
"""
from __future__ import annotations

from .hud import EXTRAS, MAINS


class UIState:
    def __init__(self, main="driver", show=(), layout="main", slide_speed=4.0):
        self.main = main
        self.layout = layout
        self.target = {n: (1.0 if n in show else 0.0) for n in EXTRAS}
        self.slide = dict(self.target)
        self.slide_speed = slide_speed  # 1/s
        self.paused = False
        self.quit = False

    @property
    def show(self):
        return [n for n in EXTRAS if self.target[n] > 0 or self.slide[n] > 0.01]

    def handle_key(self, key: str) -> None:
        k = (key or "").lower()
        if k in ("m", "tab"):
            self.main = MAINS[(MAINS.index(self.main) + 1) % len(MAINS)]
        elif k in ("1", "2"):
            n = EXTRAS[int(k) - 1]
            self.target[n] = 0.0 if self.target[n] > 0 else 1.0
        elif k == "g":
            self.layout = "grid" if self.layout == "main" else "main"
        elif k == " " or k == "space":
            self.paused = not self.paused
        elif k in ("q", "escape"):
            self.quit = True

    def tick(self, dt: float) -> None:
        for n in EXTRAS:
            step = self.slide_speed * dt
            cur, tgt = self.slide[n], self.target[n]
            self.slide[n] = min(tgt, cur + step) if tgt > cur else max(tgt, cur - step)

    def apply(self, ui) -> None:
        """DriverMainUI에 반영."""
        ui.main = self.main
        ui.show = self.show
        ui.slide = dict(self.slide)
