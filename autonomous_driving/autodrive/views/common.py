"""그림 공통: 글꼴(한글), 색, 모드 이름."""
from __future__ import annotations

import functools

from PIL import ImageFont

MODE_RGB = {"AUTONOMOUS": (42, 157, 143), "TAKEOVER": (231, 111, 81), "SAFE_STOP": (214, 40, 40), "BASELINE": (69, 123, 157)}
MODE_LABEL = {"AUTONOMOUS": "AUTONOMOUS  자율 주행(모델)", "TAKEOVER": "TAKEOVER  개입 요청 → 기준 제어기",
              "SAFE_STOP": "SAFE STOP  안전 정지", "BASELINE": "BASELINE  기준 제어기"}
# 트랙터·경운기는 시골길 시나리오에서 빠지는 중이라(일반 차로 대체) 화면에서는 일반 차로 그린다
KIND_AS = {"tractor": "car"}

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/sand-box/google/Nanum Gothic/NanumGothic-Bold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf", "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]


@functools.lru_cache(maxsize=None)
def font_path():
    import os
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    try:
        from matplotlib import font_manager as fm
        for name in ("NanumGothic", "Noto Sans CJK KR", "Malgun Gothic", "AppleGothic"):
            p = fm.findfont(name, fallback_to_default=False)
            if p:
                return p
    except Exception:
        pass
    return None


@functools.lru_cache(maxsize=None)
def font(size: int):
    p = font_path()
    size = max(6, int(size))
    if p:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size)
    except TypeError:  # 옛 Pillow
        return ImageFont.load_default()


def text_center(draw, xy, txt, size, fill, stroke=0, stroke_fill=(0, 0, 0)):
    f = font(size)
    draw.text(xy, txt, font=f, fill=fill, anchor="mm", stroke_width=stroke, stroke_fill=stroke_fill)


def scale_rgb(c, k):
    return tuple(int(max(0, min(255, v * k))) for v in c)


def mix(c1, c2, w):
    return tuple(int(a + (b - a) * w) for a, b in zip(c1, c2))
