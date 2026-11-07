"""위경도 범위(bbox) → 타일 모자이크 → PNG + 월드파일(.pgw, EPSG:3857) + 메타(JSON)."""
from __future__ import annotations

import io
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from .tiles import ATTRIBUTION, VWorldTiles, lonlat_to_merc, lonlat_to_tile, meters_per_pixel, tile_to_lonlat


@dataclass
class Mosaic:
    image: np.ndarray            # (H, W, 3) uint8, 위쪽이 북쪽
    bbox: tuple                  # (서 lon, 남 lat, 동 lon, 북 lat) 이미지 가장자리
    zoom: int
    attribution: str = ATTRIBUTION

    @property
    def merc_bounds(self):
        x0, y0 = lonlat_to_merc(self.bbox[0], self.bbox[1])
        x1, y1 = lonlat_to_merc(self.bbox[2], self.bbox[3])
        return x0, y0, x1, y1

    def lat_rows(self) -> np.ndarray:
        """각 픽셀 행 중심의 위도 (메르카토르라서 행 간격이 위도에 대해 아주 약간 비선형)."""
        H = self.image.shape[0]
        _, y0, _, y1 = self.merc_bounds
        ym = y1 - (np.arange(H) + 0.5) / H * (y1 - y0)
        return np.degrees(np.arctan(np.sinh(ym / 6378137.0)))

    def save(self, png_path: str):
        """PNG + 월드파일(.pgw, EPSG:3857) + .json(위경도 범위·출처). QGIS 등에서 바로 겹쳐짐."""
        from PIL import Image
        os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
        Image.fromarray(self.image).save(png_path)
        H, W = self.image.shape[:2]
        x0, y0, x1, y1 = self.merc_bounds
        px, py = (x1 - x0) / W, (y1 - y0) / H
        base = os.path.splitext(png_path)[0]
        with open(base + ".pgw", "w") as f:  # A, D, B, E, C, F (픽셀 중심 기준)
            f.write(f"{px:.6f}\n0.0\n0.0\n{-py:.6f}\n{x0 + px / 2:.6f}\n{y1 - py / 2:.6f}\n")
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump({"bbox_lonlat": list(self.bbox), "zoom": self.zoom, "crs": "EPSG:3857", "size": [W, H],
                       "attribution": self.attribution}, f, ensure_ascii=False, indent=1)
        return png_path


def bbox_around(lat: float, lon: float, half_w_m: float, half_h_m: float | None = None):
    """중심 ± 반폭(m) → (서, 남, 동, 북)."""
    half_h_m = half_w_m if half_h_m is None else half_h_m
    dlat = math.degrees(half_h_m / 6378137.0)
    dlon = math.degrees(half_w_m / (6378137.0 * math.cos(math.radians(lat))))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def pick_zoom(bbox, max_px: int = 4096, max_zoom: int = 19, min_zoom: int = 6) -> int:
    """긴 변이 max_px 이하가 되는 가장 큰 줌."""
    for z in range(max_zoom, min_zoom - 1, -1):
        x0, y0 = lonlat_to_tile(bbox[0], bbox[3], z)
        x1, y1 = lonlat_to_tile(bbox[2], bbox[1], z)
        if max(x1 - x0, y1 - y0) * 256 <= max_px:
            return z
    return min_zoom


def build_mosaic(bbox, zoom: int | None = None, source: VWorldTiles | None = None, max_px: int = 4096,
                 progress=None) -> Mosaic:
    """bbox(서, 남, 동, 북) 를 덮는 타일을 받아 이어 붙이고 bbox에 딱 맞게 자른다."""
    from PIL import Image
    src = source or VWorldTiles()
    z = zoom or pick_zoom(bbox, max_px)
    fx0, fy0 = lonlat_to_tile(bbox[0], bbox[3], z)  # 북서
    fx1, fy1 = lonlat_to_tile(bbox[2], bbox[1], z)  # 남동
    tx0, ty0, tx1, ty1 = int(fx0), int(fy0), int(math.ceil(fx1)) - 1, int(math.ceil(fy1)) - 1
    tiles = [(x, y) for y in range(ty0, ty1 + 1) for x in range(tx0, tx1 + 1)]
    src.check_request(z, tiles)
    canvas = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, 3), np.uint8)
    for i, (x, y) in enumerate(tiles):
        raw = src.get(z, x, y)
        if raw:
            im = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB").resize((256, 256)))
            canvas[(y - ty0) * 256:(y - ty0 + 1) * 256, (x - tx0) * 256:(x - tx0 + 1) * 256] = im
        if progress:
            progress(i + 1, len(tiles))
    # bbox에 맞게 자르기 (픽셀 경계로 반올림한 뒤, 실제 가장자리 위경도로 bbox를 다시 계산)
    c0, r0 = int(round((fx0 - tx0) * 256)), int(round((fy0 - ty0) * 256))
    c1, r1 = int(round((fx1 - tx0) * 256)), int(round((fy1 - ty0) * 256))
    img = canvas[r0:r1, c0:c1].copy()
    w, n = tile_to_lonlat(tx0 + c0 / 256, ty0 + r0 / 256, z)
    e, s = tile_to_lonlat(tx0 + c1 / 256, ty0 + r1 / 256, z)
    return Mosaic(img, (w, s, e, n), z)


def resolution_m(m: Mosaic) -> float:
    return meters_per_pixel((m.bbox[1] + m.bbox[3]) / 2, m.zoom)
