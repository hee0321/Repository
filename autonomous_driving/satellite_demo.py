"""강남역 실제 도로(OSM) 경로를 브이월드 위성영상 위에 겹쳐 그리고, SUMO 배경 설정도 만든다.

    export VWORLD_API_KEY=발급받은키          # https://www.vworld.kr → 오픈API → 인증키 발급 (무료)
    python satellite_demo.py --out media/satellite_gangnam.png
    python satellite_demo.py --zoom 18 --net sumo_kr/build/gangnam.net.xml   # SUMO decal도 생성
    sumo-gui -n sumo_kr/build/gangnam.net.xml -g satellite/cache/sumo/gangnam_station.gui.xml

키가 없으면 위성 대신 타일 격자 자리표시(회색)로 그려서 정렬/범위만 확인한다 (네트워크 요청 없음).
영상 © 국토교통부 브이월드(공공누리 제1유형, 출처표시). 타일·모자이크는 satellite/cache/ (git 무시)에만 저장.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autodrive import osm  # noqa: E402
from autodrive.roads import build_route  # noqa: E402
from korea_roads.scenarios import SCENARIOS  # noqa: E402
from satellite.mosaic import Mosaic, build_mosaic, resolution_m  # noqa: E402
from satellite.overlay import draw_satellite  # noqa: E402
from satellite.tiles import ATTRIBUTION, CACHE_DIR, VWorldTiles, lonlat_to_tile, tile_to_lonlat  # noqa: E402

plt.rcParams["font.family"] = ["NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"]


def placeholder(bbox, z):
    """키 없을 때: 같은 범위·줌의 타일 경계만 그린 회색 이미지 (정렬 확인용)."""
    fx0, fy0 = lonlat_to_tile(bbox[0], bbox[3], z)
    fx1, fy1 = lonlat_to_tile(bbox[2], bbox[1], z)
    W, H = int(round((fx1 - fx0) * 256)), int(round((fy1 - fy0) * 256))
    img = np.full((H, W, 3), 58, np.uint8)
    for x in range(int(fx0) + 1, int(fx1) + 1):
        img[:, max(0, min(W - 1, int(round((x - fx0) * 256))))] = 90
    for y in range(int(fy0) + 1, int(fy1) + 1):
        img[max(0, min(H - 1, int(round((y - fy0) * 256)))), :] = 90
    w, n = tile_to_lonlat(fx0, fy0, z)
    e, s = tile_to_lonlat(fx1, fy1, z)
    return Mosaic(img, (w, s, e, n), z, attribution="VWORLD_API_KEY 없음: 타일 격자 자리표시 | 도로 © OpenStreetMap contributors (ODbL)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="gangnam_station", choices=list(SCENARIOS))
    ap.add_argument("--zoom", type=int, default=17, help="17 ≈ 0.95 m/px, 18 ≈ 0.47 m/px, 19 ≈ 0.24 m/px")
    ap.add_argument("--pad", type=float, default=120.0, help="경로 바깥 여백 (m)")
    ap.add_argument("--max-tiles", type=int, default=400)
    ap.add_argument("--net", default=None, help="SUMO .net.xml (주면 decal + gui-settings 생성)")
    ap.add_argument("--out", default=os.path.join("media", "satellite_preview.png"))
    a = ap.parse_args()

    sc = SCENARIOS[a.scenario]
    track, rules, meta = build_route(a.scenario, sc)
    lat0, lon0 = meta["center"]
    xs, ys = track.center[:, 0], track.center[:, 1]
    la_s, lo_w = osm.to_latlon(xs.min() - a.pad, ys.min() - a.pad, lat0, lon0)
    la_n, lo_e = osm.to_latlon(xs.max() + a.pad, ys.max() + a.pad, lat0, lon0)
    bbox = (float(lo_w), float(la_s), float(lo_e), float(la_n))

    src = VWorldTiles(max_tiles=a.max_tiles)
    if src.key:
        m = build_mosaic(bbox, a.zoom, src, progress=lambda i, n: print(f"\r타일 {i}/{n}", end="", flush=True))
        print(f"\n새로 받은 타일 {src.fetched}개, 캐시 {src.cache_hits}개, 해상도 {resolution_m(m):.2f} m/px")
        mos_png = m.save(os.path.join(CACHE_DIR, "mosaic", f"{a.scenario}_z{m.zoom}.png"))
        print("모자이크(+.pgw 월드파일, EPSG:3857):", mos_png)
    else:
        print("VWORLD_API_KEY 없음 → 타일 격자 자리표시로 정렬만 확인합니다. 실제 영상은 키를 넣고 다시 실행하세요.")
        m = placeholder(bbox, a.zoom)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
    draw_satellite(ax, m, lat0, lon0)
    G = meta["graph"]
    for u, v, e in G.edges(data=True):
        ax.plot([G.nodes[u]["x"], G.nodes[v]["x"]], [G.nodes[u]["y"], G.nodes[v]["y"]],
                color="#ffb000" if e["school_zone"] else "white", lw=0.7, alpha=0.55, zorder=2)
    ax.plot(xs, ys, color="#00e5ff", lw=3, zorder=3, label="경로 (OSM)")
    for sg in rules.signals:
        ax.plot(*track.point_at(sg.s), "s", color="red", ms=6, zorder=4)
    ax.plot(xs[0], ys[0], "o", color="lime", ms=9, zorder=5)
    ax.plot(xs[-1], ys[-1], "X", color="yellow", ms=10, zorder=5)
    ax.set_xlim(xs.min() - a.pad, xs.max() + a.pad)
    ax.set_ylim(ys.min() - a.pad, ys.max() + a.pad)
    ax.set_aspect("equal")
    bg = "브이월드 영상" if src.key else "자리표시(키 없음)"
    ax.set_title(f"{meta['title']}\n{bg} z{m.zoom} (~{resolution_m(m):.2f} m/px) + OSM 경로 {meta['length_m']} m, "
                 f"신호 {meta['n_signals']}곳", fontsize=9)
    ax.set_xlabel("동쪽 (m)"); ax.set_ylabel("북쪽 (m)")
    ax.legend(loc="upper left", fontsize=8)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out); plt.close(fig)
    print("미리보기:", a.out)

    if a.net:
        from satellite.sumo_decal import write_gui_settings
        r = write_gui_settings(m, a.net, os.path.join(CACHE_DIR, "sumo"), a.scenario)
        print(f"SUMO: sumo-gui -n {a.net} -g {r['gui']}  (decal {r['size'][0]:.0f}×{r['size'][1]:.0f} m, "
              f"정합 잔차 {r['residual_m'] * 100:.1f} cm)")


if __name__ == "__main__":
    main()
