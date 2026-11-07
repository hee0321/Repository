"""브이월드(VWorld) 영상지도(Satellite) XYZ 타일 받기: 캐시, 속도 제한, 출처 표시.

URL: https://api.vworld.kr/req/wmts/1.0.0/{KEY}/Satellite/{z}/{y}/{x}.jpeg   (z/y/x 순서, EPSG:3857)
줌 6~19 (19 ≈ 0.23 m/px @ 서울 위도). 대한민국 범위만 제공.
"""
from __future__ import annotations

import math
import os
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
URL = "https://api.vworld.kr/req/wmts/1.0.0/{key}/{layer}/{z}/{y}/{x}.{ext}"
LAYERS = {"Satellite": "jpeg", "Hybrid": "png"}  # Hybrid = 투명 PNG 도로명/경계 (위에 겹쳐 그림)
MIN_ZOOM, MAX_ZOOM = 6, 19
ATTRIBUTION = "영상: © 국토교통부 브이월드(VWorld), 공공누리 제1유형 | 도로: © OpenStreetMap contributors (ODbL)"
UA = "hee0321-autodrive-study/0.1 (https://github.com/hee0321/Repository)"
# 대한민국 대략 범위 (브이월드 제공 범위 밖 요청 방지)
KOREA_BBOX = (124.0, 33.0, 132.0, 39.0)  # lon_min, lat_min, lon_max, lat_max
R = 6378137.0


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[float, float]:
    """위경도 → 타일 좌표(실수). 정수부 = 타일 번호, 소수부 = 타일 안 위치."""
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def tile_to_lonlat(x: float, y: float, z: int) -> tuple[float, float]:
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def tile_bounds_lonlat(x: int, y: int, z: int):
    """(서, 남, 동, 북)"""
    w, n = tile_to_lonlat(x, y, z)
    e, s = tile_to_lonlat(x + 1, y + 1, z)
    return w, s, e, n


def lonlat_to_merc(lon: float, lat: float) -> tuple[float, float]:
    return R * math.radians(lon), R * math.asinh(math.tan(math.radians(lat)))


def meters_per_pixel(lat: float, z: int) -> float:
    return 2 * math.pi * R * math.cos(math.radians(lat)) / (256 * 2 ** z)


class VWorldError(RuntimeError):
    pass


class VWorldTiles:
    """브이월드 타일 받기. 키는 인자 또는 VWORLD_API_KEY 환경변수.

    - 한 번 받은 타일은 cache/vworld/{layer}/{z}/{x}/{y}.{ext} 에서 재사용 (다시 요청하지 않음)
    - 요청 사이 최소 간격(min_interval, 기본 0.2 s = 초당 5개 이하), 한 번에 max_tiles 개 이하
    - 오류(키 미등록 등)는 브이월드가 XML로 돌려주므로 이미지인지 확인한다
    """

    def __init__(self, key: str | None = None, layer: str = "Satellite", cache_dir: str = CACHE_DIR,
                 min_interval: float = 0.2, max_tiles: int = 400, timeout: float = 20.0, offline: bool = False):
        if layer not in LAYERS:
            raise ValueError(f"layer는 {list(LAYERS)} 중 하나")
        self.key = key if key is not None else os.environ.get("VWORLD_API_KEY", "").strip()
        self.layer, self.ext = layer, LAYERS[layer]
        self.cache_dir = os.path.join(cache_dir, "vworld", layer)
        self.min_interval, self.max_tiles, self.timeout = min_interval, max_tiles, timeout
        self.offline = offline  # True면 캐시만 사용 (네트워크 요청 없음)
        self.domain = os.environ.get("VWORLD_DOMAIN", "").strip()  # 웹용 키를 등록 도메인으로 쓸 때만
        self._last = 0.0
        self.fetched = 0
        self.cache_hits = 0

    def url(self, z: int, x: int, y: int) -> str:
        return URL.format(key=self.key or "{VWORLD_API_KEY}", layer=self.layer, z=z, y=y, x=x, ext=self.ext)

    def path(self, z: int, x: int, y: int) -> str:
        return os.path.join(self.cache_dir, str(z), str(x), f"{y}.{self.ext}")

    def check_request(self, z: int, tiles: list[tuple[int, int]]):
        if not (MIN_ZOOM <= z <= MAX_ZOOM):
            raise ValueError(f"브이월드 영상 줌은 {MIN_ZOOM}~{MAX_ZOOM}")
        if len(tiles) > self.max_tiles:
            raise ValueError(f"타일 {len(tiles)}개 > max_tiles {self.max_tiles}: 범위를 줄이거나 줌을 낮추세요 (서버 배려)")
        lo0, la0, lo1, la1 = KOREA_BBOX
        for x, y in tiles:
            w, s, e, n = tile_bounds_lonlat(x, y, z)
            if e < lo0 or w > lo1 or n < la0 or s > la1:
                raise ValueError("브이월드는 대한민국 범위만 제공합니다")

    def get(self, z: int, x: int, y: int) -> bytes:
        p = self.path(z, x, y)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            self.cache_hits += 1
            with open(p, "rb") as f:
                return f.read()
        if self.offline:
            raise VWorldError(f"캐시에 없음(offline): {p}")
        if not self.key:
            raise VWorldError("VWORLD_API_KEY 환경변수가 없습니다. https://www.vworld.kr 오픈API → 인증키 발급(무료) 후 "
                              "export VWORLD_API_KEY=발급키")
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        headers = {"User-Agent": UA}
        if self.domain:
            headers["Referer"] = self.domain
        req = urllib.request.Request(self.url(z, x, y), headers=headers)
        last = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data, ctype = r.read(), r.headers.get("Content-Type", "")
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:  # 바다 등 영상 없는 곳
                    data, ctype = b"", "empty"
                    break
                last = e
            except urllib.error.URLError as e:
                last = e
            time.sleep(1.5 * (attempt + 1))
        else:
            raise VWorldError(f"타일 요청 실패 z{z}/{x}/{y}: {last}")
        self._last = time.monotonic()
        self.fetched += 1
        if ctype == "empty":
            return b""
        if not (ctype.startswith("image/") or data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n"):
            msg = data.decode("utf-8", "replace")
            import re
            m = re.search(r"<ExceptionText><!\[CDATA\[(.*?)\]\]>", msg)
            raise VWorldError(f"브이월드 오류: {m.group(1) if m else msg[:200]}")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
        return data
