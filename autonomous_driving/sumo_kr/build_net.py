"""캐시된 OSM(korea_roads/data/*.json) → SUMO 도로망(.net.xml) + 배경 교통(.rou.xml).

    python -m sumo_kr.build_net            # 모든 지역
SUMO(무료, EPL-2.0)는 `pip install eclipse-sumo libsumo` 로 설치. 우측 통행이 기본값.
생성 파일은 sumo_kr/build/ (git 무시, 언제든 다시 만들 수 있음).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from xml.sax.saxutils import quoteattr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BUILD = os.path.join(HERE, "build")
AREAS = {"gangnam": "gangnam_station", "daechi": "daechi_school_zone"}


def sumo_home():
    import sumo
    return sumo.SUMO_HOME


def json_to_osm(data: dict, path: str):
    """작게 줄인 캐시 JSON을 netconvert가 읽는 OSM XML로 되돌린다."""
    used = {str(n) for w in data["ways"] for n in w["nodes"]}
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6" generator="autodrive">\n')
        for nid, (lat, lon, sig) in data["nodes"].items():
            if nid not in used:
                continue
            if sig:
                f.write(f'<node id="{nid}" lat="{lat}" lon="{lon}" version="1"><tag k="highway" v="traffic_signals"/></node>\n')
            else:
                f.write(f'<node id="{nid}" lat="{lat}" lon="{lon}" version="1"/>\n')
        for w in data["ways"]:
            f.write(f'<way id="{w["id"]}" version="1">')
            for n in w["nodes"]:
                if str(n) in data["nodes"]:
                    f.write(f'<nd ref="{n}"/>')
            for k, v in w["tags"].items():
                f.write(f'<tag k={quoteattr(k)} v={quoteattr(str(v))}/>')
            f.write("</way>\n")
        f.write("</osm>\n")


def build(area: str, period: float = 2.5, end: int = 1200, seed: int = 1):
    os.makedirs(BUILD, exist_ok=True)
    data = json.load(open(os.path.join(ROOT, "korea_roads", "data", f"{AREAS[area]}.json"), encoding="utf-8"))
    osm = os.path.join(BUILD, f"{area}.osm.xml")
    net = os.path.join(BUILD, f"{area}.net.xml")
    rou = os.path.join(BUILD, f"{area}.bg.rou.xml")
    json_to_osm(data, osm)
    sh = sumo_home()
    nc = os.path.join(sh, "bin", "netconvert")
    subprocess.run([nc, "--osm-files", osm, "-o", net, "--geometry.remove", "--roundabouts.guess", "--ramps.guess",
                    "--junctions.join", "--tls.guess-signals", "--tls.discard-simple", "--tls.join",
                    "--tls.default-type", "static", "--tls.green.time", "25", "--tls.yellow.time", "3",
                    "--no-turnarounds", "--keep-edges.by-vclass", "passenger", "--remove-edges.isolated",
                    "--osm.speedlimit-none", "13.89", "--lefthand", "false", "--no-warnings", "--seed", str(seed)],
                   check=True)
    rt = os.path.join(sh, "tools", "randomTrips.py")
    subprocess.run([sys.executable, rt, "-n", net, "-r", rou, "-e", str(end), "-p", str(period), "--seed", str(seed),
                    "--fringe-factor", "5", "--min-distance", "300", "--validate",
                    "--trip-attributes", 'departLane="best" departSpeed="max"', "-o", os.path.join(BUILD, f"{area}.trips.xml")],
                   check=True, stdout=subprocess.DEVNULL)
    return net, rou


if __name__ == "__main__":
    for a in AREAS:
        print(a, build(a))
