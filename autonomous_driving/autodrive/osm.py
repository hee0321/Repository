"""OpenStreetMap(무료, ODbL)에서 한국 실제 도로망을 받아 시뮬레이터 좌표로 바꾼다.

- 카카오맵/네이버맵 데이터는 API 키가 필요하고 약관상 도로 형상 추출이 금지라서 쓰지 않는다.
  지도 확인용 '링크'만 만든다.
- Overpass API는 무료 공용 서버라서 한 번 받은 데이터는 korea_roads/data/*.json 에 작게 저장(캐시)해 재사용한다.
- 데이터 © OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.parse
import urllib.request

import numpy as np

OVERPASS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
DRIVABLE = "^(trunk|primary|secondary|tertiary|unclassified|residential|living_street|trunk_link|primary_link|secondary_link|tertiary_link)$"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "korea_roads", "data")
UA = "hee0321-autodrive-study/0.1 (https://github.com/hee0321/Repository)"


def kakao_link(name: str, lat: float, lon: float) -> str:
    return f"https://map.kakao.com/link/map/{urllib.parse.quote(name)},{lat:.6f},{lon:.6f}"


def naver_link(name: str, lat: float, lon: float) -> str:
    # 네이버 지도 웹: 좌표 중심 + 확대 단계 (검색어 링크도 함께 제공)
    return f"https://map.naver.com/p?c={lon:.6f},{lat:.6f},16,0,0,0,dh"


def _query(lat, lon, radius):
    return (f'[out:json][timeout:90];'
            f'(way["highway"~"{DRIVABLE}"](around:{radius},{lat},{lon}););(._;>;);out body;'
            f'(nwr["amenity"~"^(school|kindergarten)$"](around:{radius + 300},{lat},{lon}););out tags center;')


def fetch(name: str, lat: float, lon: float, radius: int = 800, refresh: bool = False) -> dict:
    """캐시가 있으면 읽고, 없으면 Overpass에서 받아 작게 줄여 저장한다."""
    path = os.path.join(DATA_DIR, f"{name}.json")
    if os.path.exists(path) and not refresh:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    body = urllib.parse.urlencode({"data": _query(lat, lon, radius)}).encode()
    last = None
    for attempt in range(4):
        for url in OVERPASS:
            try:
                req = urllib.request.Request(url, data=body, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=120) as r:
                    raw = json.load(r)
                break
            except Exception as e:  # 공용 서버가 바쁘면 다른 서버/잠시 후 재시도
                last = e
                raw = None
        if raw is not None:
            break
        time.sleep(15 * (attempt + 1))
    if raw is None:
        raise RuntimeError(f"Overpass 다운로드 실패: {last}")
    data = compact(raw, name, lat, lon, radius)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    return data


KEEP_TAGS = ("highway", "name", "maxspeed", "maxspeed:type", "oneway", "lanes", "hazard",
             "school_zone", "zone:maxspeed", "maxspeed:conditional")


def compact(raw: dict, name, lat, lon, radius) -> dict:
    nodes, ways, schools = {}, [], []
    for e in raw["elements"]:
        t = e.get("tags", {})
        if e["type"] == "way" and "highway" in t and "nodes" in e:
            ways.append({"id": e["id"], "nodes": e["nodes"], "tags": {k: t[k] for k in KEEP_TAGS if k in t}})
        elif e["type"] == "node" and "lat" in e and "amenity" not in t:
            nodes[str(e["id"])] = [e["lat"], e["lon"], 1 if t.get("highway") == "traffic_signals" else 0]
        if t.get("amenity") in ("school", "kindergarten"):
            c = e.get("center") or {"lat": e.get("lat"), "lon": e.get("lon")}
            if c.get("lat") is not None:
                schools.append({"name": t.get("name", ""), "amenity": t["amenity"], "lat": c["lat"], "lon": c["lon"]})
    return {"name": name, "center": [lat, lon], "radius": radius,
            "attribution": "© OpenStreetMap contributors, ODbL 1.0",
            "osm_base": raw.get("osm3s", {}).get("timestamp_osm_base"),
            "nodes": nodes, "ways": ways, "schools": schools}


def to_local(lat, lon, lat0, lon0):
    """위경도 → 중심 기준 동(x)/북(y) 미터 (작은 지역이라 등장방형 근사로 충분)."""
    R = 6378137.0
    x = np.deg2rad(np.asarray(lon) - lon0) * R * math.cos(math.radians(lat0))
    y = np.deg2rad(np.asarray(lat) - lat0) * R
    return x, y


def to_latlon(x, y, lat0, lon0):
    R = 6378137.0
    lat = lat0 + np.rad2deg(np.asarray(y) / R)
    lon = lon0 + np.rad2deg(np.asarray(x) / (R * math.cos(math.radians(lat0))))
    return lat, lon


def default_limit_kmh(highway: str) -> int:
    """안전속도 5030 (rules/kr.yaml): 큰 길(간선) 50 km/h, 주택가·이면도로 30 km/h, 도시고속화 60."""
    from .krrules import load
    lim = load()["speed_limits_kmh"]
    return int(lim.get(highway.replace("_link", ""), 50))


def is_child_facility(s: dict) -> bool:
    from .krrules import load
    n = s.get("name", "")
    return s["amenity"] == "kindergarten" or any(f in n for f in load()["school_zone"]["facilities"])


def build_graph(data: dict):
    """방향 그래프(일방통행 반영). 간선에 길이, 제한속도, 어린이보호구역 여부."""
    import networkx as nx
    lat0, lon0 = data["center"]
    nodes = data["nodes"]
    xy = {}
    for k, (la, lo, _) in nodes.items():
        x, y = to_local(la, lo, lat0, lon0)
        xy[k] = (float(x), float(y))
    child = [to_local(s["lat"], s["lon"], lat0, lon0) for s in data["schools"] if is_child_facility(s)]
    child = np.array(child, dtype=float).reshape(-1, 2)
    from .krrules import load
    zone = load()["school_zone"]
    G = nx.DiGraph()
    for k, (la, lo, sig) in nodes.items():
        G.add_node(k, x=xy[k][0], y=xy[k][1], lat=la, lon=lo, signal=bool(sig))
    for w in data["ways"]:
        t = w["tags"]
        hw = t.get("highway", "residential")
        try:
            limit = int(str(t.get("maxspeed", "")).split()[0])
        except (ValueError, IndexError):
            limit = default_limit_kmh(hw)
        tag_school = any("school" in str(t.get(k, "")) or "children" in str(t.get(k, ""))
                         for k in ("hazard", "school_zone", "zone:maxspeed", "maxspeed:type"))
        ow = t.get("oneway") in ("yes", "1", "true")
        lanes = t.get("lanes")
        ns = [str(n) for n in w["nodes"] if str(n) in nodes]
        for a, b in zip(ns[:-1], ns[1:]):
            mid = np.add(xy[a], xy[b]) / 2
            near_child = bool(len(child)) and float(np.min(np.linalg.norm(child - mid, axis=1))) < float(zone["radius_m"])
            school = tag_school or near_child
            attrs = dict(length=float(np.hypot(*np.subtract(xy[b], xy[a]))), highway=hw, name=t.get("name", ""),
                         limit_kmh=min(limit, int(zone["limit_kmh"])) if school else limit, school_zone=school,
                         oneway=ow, lanes=lanes, school_source="tag" if tag_school else ("near_child_facility" if near_child else ""))
            G.add_edge(a, b, **attrs)
            if not ow:
                G.add_edge(b, a, **attrs)
    return G


def nearest_node(G, x, y):
    ks = list(G.nodes)
    P = np.array([[G.nodes[k]["x"], G.nodes[k]["y"]] for k in ks])
    return ks[int(np.argmin(np.hypot(P[:, 0] - x, P[:, 1] - y)))]


def nearest_nodes(G, x, y, k=8, max_dist=40.0):
    ks = [n for n in G.nodes if G.degree(n) > 0]
    P = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in ks])
    d = np.hypot(P[:, 0] - x, P[:, 1] - y)
    order = np.argsort(d)[:k]
    out = [ks[i] for i in order if d[i] <= max_dist]
    return out or [ks[int(order[0])]]
