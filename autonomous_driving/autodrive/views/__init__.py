"""주행 영상 시점(view) 모음.

- driver   : 운전석 1인칭 시점(원근 투영, 대시보드·미니맵 포함)
- top      : 위에서 내려다본 지도(기존 화면과 같은 정보)
- main     : (기본) 운전석 중심 미니멀 화면 — 큰 운전석 + 속도·제한속도·배터리 HUD + 반투명 미니맵,
             모터룸·위성은 필요할 때만 작은 카드 (hud.py, 키 조작은 ui.py)
- motor    : 전기차 모터룸 (단순 모델 기반 시각화 — 실제 차량 모델 아님). 옛 이름 engine도 됨
- satellite: 위성 영상 패널(autonomous_driving/satellite 모듈이 있으면 사용, 없으면 지도 + 안내 문구)

numpy + Pillow만 쓴다(GPU 필요 없음). 영상 저장은 imageio-ffmpeg(MP4), Pillow(GIF).
"""
from .state import StateRecorder, interpolate, pick_window  # noqa: F401
from .compose import VIEWS, make_ui, render_views  # noqa: F401
from .ui import UIState  # noqa: F401
__all__ = ["StateRecorder", "interpolate", "pick_window", "VIEWS", "make_ui", "render_views", "UIState"]
