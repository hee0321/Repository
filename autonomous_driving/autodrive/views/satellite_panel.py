"""위성 영상 패널 (끼워 넣기 방식).

위성 타일 기능은 다른 브랜치(autodrive/satellite, 폴더 autonomous_driving/satellite/)에서 만드는 중이다.
여기서는 그 코드를 복사하지 않고, 모듈이 설치돼 있으면 불러서 쓰고 없으면 지도 + '위성 영상 없음' 안내를 보여 준다.

satellite 모듈이 맞춰 주면 되는 약속(둘 중 하나):
  render_panel(env, frame, size) -> PIL.Image 또는 HxWx3 uint8 배열   # 패널 전체를 직접 그림
  background(env, frame, size, meters_per_px) -> PIL.Image             # 차 위치 중심 배경만 주면 위에 도로·차를 덧그림
frame: {"x","y","yaw","s","t","v","mode","ob",...} (월드 좌표 m, 트랙 좌표계), size: (W, H)
"""
from __future__ import annotations

import importlib

import numpy as np
from PIL import Image

from .topdown import TopDownView


def load_satellite_module():
    for name in ("satellite", "autonomous_driving.satellite"):
        try:
            return importlib.import_module(name), name
        except Exception:
            continue
    return None, None


class SatelliteView:
    def __init__(self, env, W=640, H=360, kinds=None):
        self.env, self.W, self.H = env, W, H
        self.mod, self.modname = load_satellite_module()
        self.available = self.mod is not None and (hasattr(self.mod, "render_panel") or hasattr(self.mod, "background"))
        self.reason = None if self.available else ("satellite 모듈 없음" if self.mod is None else "satellite 모듈에 render_panel/background 없음")
        self.top = TopDownView(env, W, H, kinds=kinds, note=None if self.available else f"위성 영상 없음({self.reason}) → 지도로 대신")
        self.failed = False

    def render(self, f) -> Image.Image:
        if self.available and not self.failed:
            try:
                if hasattr(self.mod, "render_panel"):
                    out = self.mod.render_panel(self.env, f, (self.W, self.H))
                    if isinstance(out, np.ndarray):
                        out = Image.fromarray(out.astype(np.uint8))
                    return out.convert("RGB").resize((self.W, self.H))
                bg = self.mod.background(self.env, f, (self.W, self.H), 1.0 / self.top.sc)
                if isinstance(bg, np.ndarray):
                    bg = Image.fromarray(bg.astype(np.uint8))
                self.top.note = "위성 영상 위에 도로·차"
                return self.top.render(f, bg=bg.convert("RGB").resize((self.W, self.H)), road_alpha=90)
            except Exception as e:  # 위성 모듈 오류 → 지도로 계속
                self.failed = True
                self.top.note = f"위성 영상 오류({type(e).__name__}) → 지도로 대신"
        return self.top.render(f)
