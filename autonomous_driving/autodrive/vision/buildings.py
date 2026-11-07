"""OSM 건물 외곽 + 높이 (Overpass, © OpenStreetMap contributors, ODbL) → satellite/cache/ground/buildings_{지도}.json.

높이: height / building:height 태그, 없으면 building:levels × 3.2 m, 없으면 종류별 기본값(주택 7 m 등).
경로 통로(경로에서 120 m 안) 건물만 저장. 무료 공개 API — 요청은 지도마다 한 번, 캐시 재사용.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np

from .ground import CACHE, _to_latlon

OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter"]
UA = "hee0321-autodrive-study/0.2 (https://github.com/hee0321/Repository)"
DEFAULT_H = {"house": 7.0, "detached": 7.0, "shed": 3.5, "garage": 3.5, "greenhouse": 3.5, "farm_auxiliary": 4.5,
             "barn": 6.0, "roof": 4.0, "apartments": 30.0, "commercial": 14.0, "office": 20.0, "church": 14.0}


def height_of(tags: dict) -> float:
    for k in ("height", "building:height"):
        v = tags.get(k)
        if v:
            try:
                return float(str(v).replace("m", "").strip())
            except ValueError:
                pass
    lv = tags.get("building:levels")
    if lv:
        try:
            return float(str(lv).split(";")[0]) * 3.2
        except ValueError:
            pass
    return DEFAULT_H.get(tags.get("building", "yes"), 11.0)


def fetch_buildings(data_key, paths, lat0, lon0, margin=120.0):
    os.makedirs(CACHE, exist_ok=True)
    out_p = os.path.join(CACHE, f"buildings_{data_key}.json")
    if os.path.exists(out_p):
        return json.load(open(out_p))
    P = np.concatenate([np.asarray(p) for p in paths])
    x0, y0 = P.min(axis=0) - margin
    x1, y1 = P.max(axis=0) + margin
    la_s, lo_w = _to_latlon(x0, y0, lat0, lon0)
    la_n, lo_e = _to_latlon(x1, y1, lat0, lon0)
    bb = f"{float(la_s):.6f},{float(lo_w):.6f},{float(la_n):.6f},{float(lo_e):.6f}"
    q = f'[out:json][timeout:120];(way["building"]({bb}););out geom;'
    body = urllib.parse.urlencode({"data": q}).encode()
    raw = None
    for attempt in range(3):
        for url in OVERPASS:
            try:
                with urllib.request.urlopen(urllib.request.Request(url, data=body, headers={"User-Agent": UA}), timeout=150) as r:
                    raw = json.load(r)
                break
            except Exception as e:  # pragma: no cover - 네트워크
                print("overpass", url, e)
        if raw:
            break
        time.sleep(15)
    if raw is None:
        return []
    R = 6378137.0
    import math
    keep = []
    sub = P[::5]
    for e in raw.get("elements", []):
        g = e.get("geometry")
        if not g or len(g) < 4:
            continue
        la = np.array([p["lat"] for p in g])
        lo = np.array([p["lon"] for p in g])
        x = np.deg2rad(lo - lon0) * R * math.cos(math.radians(lat0))
        y = np.deg2rad(la - lat0) * R
        c = np.array([x.mean(), y.mean()])
        if np.min(np.linalg.norm(sub - c, axis=1)) > margin:
            continue
        keep.append([np.c_[x, y].round(2).tolist(), round(height_of(e.get("tags", {})), 1)])
    json.dump(keep, open(out_p, "w"))
    return keep
