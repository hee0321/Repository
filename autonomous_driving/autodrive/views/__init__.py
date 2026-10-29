"""주행 영상 시점(view) 모음.

- driver   : 운전석 1인칭 시점(원근 투영, 대시보드·미니맵 포함)
- top      : 위에서 내려다본 지도(기존 화면과 같은 정보)
- engine   : 엔진룸 (단순 모델 기반 시각화 — 실제 엔진 모델 아님)
- satellite: 위성 영상 패널(autonomous_driving/satellite 모듈이 있으면 사용, 없으면 지도 + 안내 문구)

numpy + Pillow만 쓴다(GPU 필요 없음). 영상 저장은 imageio-ffmpeg(MP4), Pillow(GIF).
"""
from .state import StateRecorder, interpolate, pick_window  # noqa: F401
from .compose import VIEWS, render_views  # noqa: F401
__all__ = ["StateRecorder", "interpolate", "pick_window", "VIEWS", "render_views"]
