"""위성영상 바닥 텍스처: 경로 주변(통로) 브이월드 타일 → 지역 좌표(m) 래스터.

- 출처: 국토교통부 브이월드 영상지도, 공공누리 제1유형 (출처표시). 키는 VWORLD_API_KEY 환경변수로만.
- 캐시: satellite/cache/ (git 무시). 영상·래스터는 저장소에 올리지 않는다.
- 키도 캐시도 없으면 None → 카메라는 절차 생성 바닥(풀·흙)으로 그린다.
"""
from __future__ import annotations

import io
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(HERE, "satellite", "cache", "ground")
ATTRIBUTION = "위성영상 © 국토교통부 브이월드(VWorld) 공공누리 제1유형 | 도로·건물 © OpenStreetMap contributors (ODbL)"


class Raster:
    """RGB 래스터, 지역 좌표 [x0, x0+W·res] × [y0, y0+H·res], 0행 = 북쪽. valid: 영상이 있는 칸."""

    def __init__(self, img, x0, y0, res, valid=None):
        self.img = np.ascontiguousarray(img, dtype=np.uint8)
        self.x0, self.y0, self.res = float(x0), float(y0), float(res)
        self.H, self.W = img.shape[:2]
        self.y1 = self.y0 + self.H * self.res
        self.valid = valid if valid is not None else np.ones((self.H, self.W), bool)

    def sample(self, X, Y):
        """최근접 표본 → (RGB float32, 있음 여부). 카메라 해상도가 낮아 쌍선형까지는 필요 없음."""
        c = ((X - self.x0) / self.res).astype(np.int32)
        r = ((self.y1 - Y) / self.res).astype(np.int32)
        ok = (c >= 0) & (c < self.W) & (r >= 0) & (r < self.H)
        c = np.clip(c, 0, self.W - 1)
        r = np.clip(r, 0, self.H - 1)
        ok &= self.valid[r, c]
        return self.img[r, c].astype(np.float32), ok


def _to_latlon(x, y, lat0, lon0):
    R = 6378137.0
    return lat0 + np.rad2deg(np.asarray(y) / R), lon0 + np.rad2deg(np.asarray(x) / (R * math.cos(math.radians(lat0))))


def corridor_tiles(paths, lat0, lon0, zoom, margin=90.0):
    """경로들(지역 좌표 Nx2) 주변 margin m 안을 덮는 타일 집합."""
    from satellite.tiles import lonlat_to_tile
    tiles = set()
    for P in paths:
        P = np.asarray(P)
        step = max(1, int(len(P) / max(len(P) / 10, 1)))
        for x, y in P[::step]:
            for dx in np.arange(-margin, margin + 1, 30):
                for dy in np.arange(-margin, margin + 1, 30):
                    la, lo = _to_latlon(x + dx, y + dy, lat0, lon0)
                    tx, ty = lonlat_to_tile(float(lo), float(la), zoom)
                    tiles.add((int(tx), int(ty)))
    return sorted(tiles)


def build_ground(key: str, paths, lat0, lon0, zoom=18, res=0.5, margin=90.0, offline=False):
    """경로 통로 위성 래스터를 만들거나 캐시에서 읽는다. 실패하면 None."""
    path = os.path.join(CACHE, f"{key}_z{zoom}_r{res}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return Raster(d["img"], *d["box"][:2], float(d["res"]), d["valid"])
    try:
        from PIL import Image
        from satellite.tiles import VWorldTiles, VWorldError
    except ImportError:
        return None
    if offline or not os.environ.get("VWORLD_API_KEY"):
        return None
    tiles = corridor_tiles(paths, lat0, lon0, zoom, margin)
    src = VWorldTiles(min_interval=0.2, max_tiles=max(400, len(tiles)))
    src.check_request(zoom, tiles)
    P = np.concatenate([np.asarray(p) for p in paths])
    x0, y0 = np.floor(P.min(axis=0) - margin)
    x1, y1 = np.ceil(P.max(axis=0) + margin)
    W, H = int((x1 - x0) / res), int((y1 - y0) / res)
    img = np.zeros((H, W, 3), np.uint8)
    valid = np.zeros((H, W), bool)
    tset = {}
    for tx, ty in tiles:
        try:
            raw = src.get(zoom, tx, ty)
        except Exception as e:  # pragma: no cover - 네트워크(시간 초과 등): 그 타일만 빼고 계속
            print("tile fail", tx, ty, str(e)[:80])
            continue
        if raw:
            tset[(tx, ty)] = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB").resize((256, 256)))
    print(f"[ground {key}] tiles {len(tiles)} fetched {src.fetched} cached {src.cache_hits}", flush=True)
    n = 2 ** zoom
    xs = x0 + (np.arange(W) + 0.5) * res
    ys = y1 - (np.arange(H) + 0.5) * res
    la = _to_latlon(0 * ys, ys, lat0, lon0)[0]
    lo = _to_latlon(xs, 0 * xs, lat0, lon0)[1]
    fx = (lo + 180) / 360 * n
    fy = (1 - np.arcsinh(np.tan(np.radians(la))) / math.pi) / 2 * n
    for (tx, ty), t in tset.items():
        cs = np.nonzero((fx >= tx) & (fx < tx + 1))[0]
        rs = np.nonzero((fy >= ty) & (fy < ty + 1))[0]
        if not len(cs) or not len(rs):
            continue
        pc = np.clip(((fx[cs] - tx) * 256).astype(int), 0, 255)
        pr = np.clip(((fy[rs] - ty) * 256).astype(int), 0, 255)
        img[rs[0]:rs[-1] + 1, cs[0]:cs[-1] + 1] = t[pr][:, pc]
        valid[rs[0]:rs[-1] + 1, cs[0]:cs[-1] + 1] = True
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, img=img, valid=valid, box=np.array([x0, y0, x1, y1]), res=res)
    return Raster(img, x0, y0, res, valid)


_GROUND = {}


def ground_for(data_key: str, offline=False):
    """지도 데이터 이름(gangnam_station, yangpyeong_jeongbae, daechi_school_zone) → 그 지도를 쓰는 모든 시나리오
    경로를 덮는 위성 래스터 (프로세스 안 캐시)."""
    if data_key in _GROUND:
        return _GROUND[data_key]
    from korea_roads.scenarios import SCENARIOS
    from ..scenarios import korea_route
    names = [k for k, v in SCENARIOS.items() if v.get("data", k) == data_key]
    paths, center = [], None
    for k in names:
        tr, _, meta = korea_route(k)
        paths.append(tr.center)
        center = meta["center"]
    r = build_ground(data_key, paths, *center, offline=offline) if paths else None
    _GROUND[data_key] = r
    return r
