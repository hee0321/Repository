"""위성영상 배경: 국토교통부 브이월드(VWorld) 영상지도 타일 → 모자이크 PNG + 월드파일 → matplotlib/SUMO 배경.

- 출처: 국토교통부 브이월드(VWorld) WMTS/TMS API, 공공누리 제1유형(출처표시, 상업적 이용·변형 가능)
  https://www.data.go.kr/data/3046388/openapi.do
- 인증키(무료)는 환경변수 VWORLD_API_KEY 로만 읽는다. 키는 절대 커밋하지 않는다.
- 받은 타일/모자이크는 satellite/cache/ (git 무시)에만 저장한다.
"""
from .tiles import ATTRIBUTION, VWorldTiles, lonlat_to_tile, tile_bounds_lonlat  # noqa: F401
from .mosaic import Mosaic, build_mosaic  # noqa: F401
