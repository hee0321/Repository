"""위성 배경(브이월드) 점검: 네트워크 없이 합성 타일로 모자이크·정렬·SUMO decal 확인.  python -m pytest tests -q"""
import io
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from satellite.mosaic import bbox_around, build_mosaic  # noqa: E402
from satellite.overlay import to_local_image  # noqa: E402
from satellite.sumo_decal import lonlat_to_net, read_location, utm_forward, write_gui_settings  # noqa: E402
from satellite.tiles import VWorldError, VWorldTiles, lonlat_to_tile, tile_to_lonlat  # noqa: E402

GANGNAM = (37.4979, 127.0276)


class FakeTiles(VWorldTiles):
    """각 타일을 (x, y) 로 칠한 합성 이미지. 네트워크 요청 없음."""

    def get(self, z, x, y):
        from PIL import Image
        im = np.zeros((256, 256, 3), np.uint8)
        im[..., 0], im[..., 1] = x % 256, y % 256
        b = io.BytesIO()
        Image.fromarray(im).save(b, "PNG")
        self.fetched += 1
        return b.getvalue()


def test_tile_math_roundtrip():
    x, y = lonlat_to_tile(GANGNAM[1], GANGNAM[0], 17)
    assert (int(x), int(y)) == (111785, 50789)
    lon, lat = tile_to_lonlat(x, y, 17)
    assert abs(lon - GANGNAM[1]) < 1e-9 and abs(lat - GANGNAM[0]) < 1e-9


def test_url_and_key_handling(tmp_path, monkeypatch):
    monkeypatch.delenv("VWORLD_API_KEY", raising=False)
    t = VWorldTiles(cache_dir=str(tmp_path))
    assert t.url(17, 111785, 50789).endswith("/Satellite/17/50789/111785.jpeg")  # z/y/x 순서
    with pytest.raises(VWorldError):
        t.get(17, 111785, 50789)  # 키 없으면 요청하지 않음
    with pytest.raises(ValueError):
        t.check_request(17, [(0, 0)])  # 한국 밖
    with pytest.raises(ValueError):
        VWorldTiles(cache_dir=str(tmp_path), max_tiles=3).check_request(17, [(111785, 50789)] * 4)


def test_mosaic_crop_and_worldfile(tmp_path):
    bbox = bbox_around(*GANGNAM, 300)
    m = build_mosaic(bbox, 17, FakeTiles(cache_dir=str(tmp_path)))
    w, s, e, n = m.bbox
    assert abs(w - bbox[0]) < 2e-5 and abs(n - bbox[3]) < 2e-5 and abs(e - bbox[2]) < 2e-5
    H, W = m.image.shape[:2]
    assert abs(W - 600 / 0.946) < 6  # z17 ≈ 0.95 m/px
    # 강남역 픽셀의 색 = 그 타일 번호
    fx0, fy0 = lonlat_to_tile(w, n, 17)
    x, y = lonlat_to_tile(GANGNAM[1], GANGNAM[0], 17)
    px = m.image[int((y - fy0) * 256), int((x - fx0) * 256)]
    assert (px[0], px[1]) == (111785 % 256, 50789 % 256)
    p = m.save(str(tmp_path / "m.png"))
    pgw = open(p[:-4] + ".pgw").read().split()
    assert len(pgw) == 6 and float(pgw[0]) > 0 > float(pgw[3])


def test_overlay_extent_matches_local_coords(tmp_path):
    from autodrive.osm import to_local
    lat0, lon0 = 37.5010, 127.0300
    m = build_mosaic(bbox_around(*GANGNAM, 250), 17, FakeTiles(cache_dir=str(tmp_path)))
    img, (x0, x1, y0, y1) = to_local_image(m, lat0, lon0)
    gx, gy = to_local(GANGNAM[0], GANGNAM[1], lat0, lon0)
    c = int((gx - x0) / (x1 - x0) * img.shape[1]); r = int((y1 - gy) / (y1 - y0) * img.shape[0])
    assert (img[r, c, 0], img[r, c, 1]) == (111785 % 256, 50789 % 256)


def test_utm_matches_pyproj_reference():
    x, y = utm_forward(127.0276, 37.4979, 52)  # pyproj EPSG:32652 값
    assert abs(x - 325645.5332) < 0.01 and abs(y - 4151935.6188) < 0.01


def test_sumo_decal_aligned_to_net(tmp_path):
    net = tmp_path / "gangnam.net.xml"
    net.write_text('<?xml version="1.0"?>\n<net version="1.20">\n    <location netOffset="-324928.86,-4151163.79" '
                   'convBoundary="0.00,0.00,2183.44,2337.61" origBoundary="127.019342,37.490998,127.043868,37.512078" '
                   'projParameter="+proj=utm +zone=52 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"/>\n</net>\n')
    loc = read_location(str(net))
    m = build_mosaic(bbox_around(*GANGNAM, 200), 17, FakeTiles(cache_dir=str(tmp_path)))
    r = write_gui_settings(m, str(net), str(tmp_path / "sumo"), "g")
    assert r["residual_m"] < 0.05
    xml = open(r["gui"], encoding="utf-8").read()
    assert "<decal" in xml and 'file="g.decal.png"' in xml and "브이월드" in xml
    from PIL import Image
    img = np.asarray(Image.open(r["decal"]))
    gx, gy = lonlat_to_net(GANGNAM[1], GANGNAM[0], loc)
    cx, cy = r["center"]; w, h = r["size"]
    c = int((gx - (cx - w / 2)) / w * img.shape[1]); rr = int(((cy + h / 2) - gy) / h * img.shape[0])
    assert img[rr, c, 3] == 255 and (img[rr, c, 0], img[rr, c, 1]) == (111785 % 256, 50789 % 256)
