"""(옛 이름) 엔진룸 → 이제 전기차 모터룸. 예전 코드·명령(--view engine)이 계속 되도록 남긴 별칭."""
from .motor import MotorBayView

EngineView = MotorBayView
__all__ = ["EngineView", "MotorBayView"]
