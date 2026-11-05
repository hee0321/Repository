"""모자이크를 우리 matplotlib 렌더러(osm.to_local 좌표: 중심 기준 동/북 미터)에 정확히 겹친다."""
from __future__ import annotations

import numpy as np

from .mosaic import Mosaic


def to_local_image(m: Mosaic, lat0: float, lon0: float):
    """메르카토르 행을 위도 등간격으로 다시 뽑아(우리 좌표는 위도에 선형) imshow용 (이미지, extent) 반환."""
    from autodrive.osm import to_local
    H = m.image.shape[0]
    lat_r = m.lat_rows()                                    # 위→아래, 감소
    s, n = m.bbox[1], m.bbox[3]
    want = n - (np.arange(H) + 0.5) / H * (n - s)           # 위도 등간격
    rows = np.clip(np.round(np.interp(-want, -lat_r, np.arange(H))).astype(int), 0, H - 1)
    img = m.image[rows]
    x0, y0 = to_local(m.bbox[1], m.bbox[0], lat0, lon0)
    x1, y1 = to_local(m.bbox[3], m.bbox[2], lat0, lon0)
    return img, (float(x0), float(x1), float(y0), float(y1))


def draw_satellite(ax, m: Mosaic, lat0: float, lon0: float, alpha: float = 1.0, attribution: bool = True,
                   zorder: int = 0):
    """ax에 위성 배경을 깐다. 이후 도로/경로/차량을 같은 좌표로 그리면 그대로 맞는다."""
    img, ext = to_local_image(m, lat0, lon0)
    ax.imshow(img, extent=ext, origin="upper", alpha=alpha, zorder=zorder, interpolation="bilinear")
    if attribution:
        ax.text(0.995, 0.005, m.attribution, transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
                color="white", bbox=dict(facecolor="black", alpha=0.45, pad=1.5, lw=0), zorder=50)
    return ext
