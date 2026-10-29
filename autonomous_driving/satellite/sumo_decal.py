"""SUMO GUI 배경(decal) 만들기: 모자이크를 .net.xml 좌표(netOffset + UTM)에 맞춰 다시 뽑고 gui-settings XML 생성.

SUMO가 OSM으로 만든 망의 좌표 = UTM(projParameter) + netOffset. 위성(메르카토르)과 UTM은 서울에서 약 1.2° 돌아가
있으므로 회전 없는 decal로 쓰도록 이미지를 망 좌표 격자에 맞춰 다시 샘플링한다.
    sumo-gui -n gangnam.net.xml -g gangnam.satellite.gui.xml
"""
from __future__ import annotations

import math
import os
import re
from xml.sax.saxutils import quoteattr

import numpy as np

from .mosaic import Mosaic
from .tiles import lonlat_to_merc


def read_location(net_xml: str) -> dict:
    """.net.xml 의 <location> (파일 앞부분만 읽음)."""
    with open(net_xml, encoding="utf-8") as f:
        head = f.read(20000)
    m = re.search(r"<location\s+([^>]*)/?>", head)
    if not m:
        raise ValueError(f"<location> 없음: {net_xml}")
    attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
    off = [float(v) for v in attrs["netOffset"].split(",")]
    conv = [float(v) for v in attrs["convBoundary"].split(",")]
    orig = [float(v) for v in attrs.get("origBoundary", "0,0,0,0").split(",")]
    proj = attrs.get("projParameter", "!")
    zone = re.search(r"\+zone=(\d+)", proj)
    if not zone or "+proj=utm" not in proj:
        raise ValueError(f"UTM 투영 망만 지원합니다 (projParameter={proj!r})")
    return {"netOffset": off, "convBoundary": conv, "origBoundary": orig, "projParameter": proj,
            "zone": int(zone.group(1)), "south": "+south" in proj}


def utm_forward(lon, lat, zone: int, south: bool = False):
    """WGS84 위경도 → UTM (m). Snyder 급수 (수 mm 정확도, pyproj 없이)."""
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    phi, lam = np.radians(lat), np.radians(lon)
    lam0 = math.radians((zone - 1) * 6 - 180 + 3)
    N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    T, C = np.tan(phi) ** 2, ep2 * np.cos(phi) ** 2
    A = np.cos(phi) * (lam - lam0)
    M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * phi)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * phi)
             - (35 * e2 ** 3 / 3072) * np.sin(6 * phi))
    x = k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000.0
    y = k0 * (M + N * np.tan(phi) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                                     + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720))
    if south:
        y = y + 10000000.0
    return x, y


def lonlat_to_net(lon, lat, loc: dict):
    x, y = utm_forward(lon, lat, loc["zone"], loc["south"])
    return x + loc["netOffset"][0], y + loc["netOffset"][1]


def _merc_to_net_affine(m: Mosaic, loc: dict):
    """메르카토르 → 망 좌표 아핀 근사(2 km 안에서는 오차 수 cm). (A 2x3, 최대 잔차 m)"""
    w, s, e, n = m.bbox
    lo, la = np.meshgrid(np.linspace(w, e, 7), np.linspace(s, n, 7))
    lo, la = lo.ravel(), la.ravel()
    mx, my = np.array([lonlat_to_merc(a, b) for a, b in zip(lo, la)]).T
    nx_, ny_ = lonlat_to_net(lo, la, loc)
    X = np.c_[mx, my, np.ones_like(mx)]
    A, *_ = np.linalg.lstsq(X, np.c_[nx_, ny_], rcond=None)
    resid = float(np.max(np.hypot(*(X @ A - np.c_[nx_, ny_]).T)))
    return A.T, resid


def reproject_to_net(m: Mosaic, loc: dict, px_m: float | None = None):
    """모자이크를 망 좌표 축에 맞춘 이미지로 다시 뽑는다. 반환: (image, (xmin, ymin, xmax, ymax), 잔차 m)."""
    A, resid = _merc_to_net_affine(m, loc)
    H, W = m.image.shape[:2]
    mx0, my0, mx1, my1 = m.merc_bounds
    corners = np.array([[mx0, my0, 1], [mx1, my0, 1], [mx0, my1, 1], [mx1, my1, 1]]) @ A.T
    xmin, ymin = corners.min(axis=0)
    xmax, ymax = corners.max(axis=0)
    if px_m is None:  # 원본 해상도(망 미터 기준)와 비슷하게
        px_m = float(np.hypot(*(A[:, 0] * (mx1 - mx0) / W)))
    Wo, Ho = int(math.ceil((xmax - xmin) / px_m)), int(math.ceil((ymax - ymin) / px_m))
    gx, gy = np.meshgrid(xmin + (np.arange(Wo) + 0.5) * px_m, ymax - (np.arange(Ho) + 0.5) * px_m)
    Ainv = np.linalg.inv(np.r_[A, [[0, 0, 1]]])[:2]
    mx = Ainv[0, 0] * gx + Ainv[0, 1] * gy + Ainv[0, 2]
    my = Ainv[1, 0] * gx + Ainv[1, 1] * gy + Ainv[1, 2]
    col = np.floor((mx - mx0) / (mx1 - mx0) * W).astype(int)
    row = np.floor((my1 - my) / (my1 - my0) * H).astype(int)
    ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    out = np.zeros((Ho, Wo, 4), np.uint8)
    out[ok, :3] = m.image[row[ok], col[ok]]
    out[ok, 3] = 255  # 바깥은 투명
    return out, (float(xmin), float(ymin), float(xmin + Wo * px_m), float(ymax)), resid


def write_gui_settings(m: Mosaic, net_xml: str, out_dir: str, name: str = "satellite") -> dict:
    """decal PNG + gui-settings XML 생성. sumo-gui -n NET -g 반환된 gui 파일."""
    from PIL import Image
    loc = read_location(net_xml)
    img, (x0, y0, x1, y1), resid = reproject_to_net(m, loc)
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{name}.decal.png")
    Image.fromarray(img, "RGBA").save(png)
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    cb = loc["convBoundary"]
    vx, vy = (cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2
    gui = os.path.join(out_dir, f"{name}.gui.xml")
    with open(gui, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(f"<!-- {m.attribution} -->\n")
        f.write(f"<!-- 원본 {os.path.basename(net_xml)}: netOffset={loc['netOffset']} {loc['projParameter']} "
                f"/ 정합 잔차 최대 {resid:.3f} m -->\n")
        f.write('<viewsettings>\n    <scheme name="real world"/>\n')
        f.write(f'    <viewport zoom="100" x="{vx:.2f}" y="{vy:.2f}"/>\n')
        f.write(f'    <decal file={quoteattr(os.path.basename(png))} centerX="{cx:.2f}" centerY="{cy:.2f}" '
                f'width="{w:.2f}" height="{h:.2f}" rotation="0.00" layer="-1.00"/>\n')
        f.write("</viewsettings>\n")
    return {"gui": gui, "decal": png, "center": (cx, cy), "size": (w, h), "residual_m": resid, "location": loc}
