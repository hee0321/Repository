"""OSM 도로망 → 경로(Track) + 한국 교통 규칙(TrafficRules)."""
from __future__ import annotations

import numpy as np

from . import osm
from .rules import KMH, LANE_W, Signal, TrafficRules
from .track import Track


def _resample(P, step=1.0):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    keep = np.r_[True, np.diff(d) > 1e-6]
    P, d = P[keep], d[keep]
    s = np.arange(0, d[-1], step)
    return np.stack([np.interp(s, d, P[:, 0]), np.interp(s, d, P[:, 1])], axis=1), s


def _smooth(P, win=9, iters=3):
    """교차로의 각진 꺾임을 둥글게 (끝점 고정 이동평균)."""
    Q = P.copy()
    k = win // 2
    for _ in range(iters):
        R = Q.copy()
        for i in range(k, len(Q) - k):
            R[i] = Q[i - k:i + k + 1].mean(axis=0)
        Q = R
    return Q


def build_route(name: str, scenario: dict, signal_seed: int = 0):
    """시나리오(웨이포인트) → (Track, TrafficRules, meta). 데이터는 캐시에서 읽는다(없으면 받는다)."""
    import networkx as nx
    data = osm.fetch(scenario.get("data", name), *scenario["center"], scenario["radius"])
    G = osm.build_graph(data)
    lat0, lon0 = data["center"]
    # 웨이포인트마다 가까운 노드 후보 여러 개(40 m 안) 중에서 전체 경로가 가장 짧은 조합을 고른다.
    # (왕복 분리 도로에서 반대편 차로 노드에 붙어 빙 도는 것을 막음)
    cands = [osm.nearest_nodes(G, *osm.to_local(la, lo, lat0, lon0), k=8, max_dist=40.0)
             for la, lo in scenario["waypoints"]]
    best = {c: (0.0, [c]) for c in cands[0]}
    for nxt in cands[1:]:
        new = {}
        for b in nxt:
            for a, (cost, pth) in best.items():
                try:
                    L, seg = nx.single_source_dijkstra(G, a, b, weight="length")
                except nx.NetworkXNoPath:
                    continue
                if b not in new or cost + L < new[b][0]:
                    new[b] = (cost + L, pth + seg[1:])
        best = new
    path = min(best.values(), key=lambda v: v[0])[1]
    raw = np.array([[G.nodes[k]["x"], G.nodes[k]["y"]] for k in path])
    # 원래 노드별 누적 거리 → 간선 속성을 1 m 표본에 칠하기
    node_d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
    P, s = _resample(raw, 1.0)
    P0 = P.copy()
    # 교차로 회전 반경이 너무 작으면(차 최소 회전 반경 약 4.6 m + 차로 오프셋) 못 도니까,
    # 최대 곡률이 0.12(반경 약 8 m)보다 작아질 때까지 꺾인 곳만 조금씩 더 둥글게 한다.
    P = _smooth(P, win=9, iters=3)
    for _ in range(60):
        seg = np.diff(P, axis=0)
        h = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
        kap = np.abs(np.diff(h)) / np.maximum(np.linalg.norm(seg[1:], axis=1), 1e-6)
        bad = np.nonzero(kap > 0.12)[0]
        if not len(bad):
            break
        mask = np.zeros(len(P), bool)
        for b in bad:
            mask[max(1, b - 12):min(len(P) - 1, b + 14)] = True
        Q = P.copy()
        Q[1:-1] = (P[:-2] + P[1:-1] + P[2:]) / 3
        P = np.where(mask[:, None], Q, P)
        P, _ = _resample(P, 1.0)
    # 둥글게 하면서 길이가 조금 줄어드니 원래 노드 거리 비율을 맞춘다
    s = np.linspace(0, node_d[-1], len(P))
    edges = [G.edges[a, b] for a, b in zip(path[:-1], path[1:])]
    ei = np.clip(np.searchsorted(node_d, s, side="right") - 1, 0, len(edges) - 1)
    limit = np.array([edges[i]["limit_kmh"] for i in ei], float) * KMH
    school = np.array([edges[i]["school_zone"] for i in ei], bool)
    twoway = np.array([not edges[i]["oneway"] for i in ei], bool)
    lane_ref = np.full(len(s), -LANE_W / 2)  # 우측 통행: 오른쪽 차로 가운데
    track = Track(P, width=2 * LANE_W, closed=False, name=name)
    # 신호등: 경로 위 신호 노드 → 30 m 안에 몰린 것은 한 교차로로 합치고 정지선은 3 m 앞
    rng = np.random.default_rng(signal_seed)
    sig_s = sorted(node_d[i] for i, k in enumerate(path) if G.nodes[k]["signal"])
    merged = []
    for x in sig_s:
        if not merged or x - merged[-1] > 30:
            merged.append(x)
    from .krrules import load
    cfg = load()
    before = float(cfg["traffic_signal"]["stop_line_before_m"])
    sigs = []
    for k, x in enumerate(merged):
        if not (15 < x < track.length - 5):
            continue
        # 우회전 판단: 신호 앞뒤 30 m 사이에서 진행 방향이 오른쪽(시계 방향, 음수)으로 꺾이면 우회전
        dh = np.angle(np.exp(1j * (track.heading_at(x + 30) - track.heading_at(x - 30))))
        sigs.append(Signal(s=float(max(x - before, 1.0)), offset=float(rng.uniform(0, 48)), name=f"signal{k}",
                           right_turn=bool(dh < -cfg["right_turn_stop"]["detect_turn_rad"])))
    rules = TrafficRules(1.0, limit, school, twoway, lane_ref, sigs, closed=False, length=track.length)
    # 메타: 지도 링크, 지나는 길 이름
    names = []
    for e in edges:
        if e["name"] and (not names or names[-1] != e["name"]):
            names.append(e["name"])
    mid_lat, mid_lon = osm.to_latlon(*P[len(P) // 2], lat0, lon0)
    sig_ll = [tuple(map(float, osm.to_latlon(*track.point_at(sg.s), lat0, lon0))) for sg in sigs]
    meta = {
        "title": scenario["title"], "center": (lat0, lon0), "streets": names,
        "length_m": round(track.length, 1), "right_turn_signals": sum(sg.right_turn for sg in sigs), "n_signals": len(sigs),
        "school_zone_m": int(school.sum()), "limits_kmh": sorted(set(int(round(x / KMH)) for x in limit)),
        "kakao": osm.kakao_link(scenario["title"].split(":")[0], float(mid_lat), float(mid_lon)),
        "naver": osm.naver_link(scenario["title"], float(mid_lat), float(mid_lon)),
        "signals_latlon": sig_ll, "graph": G, "data": data,
        "school_sources": sorted(set(e["school_source"] for e in edges if e["school_zone"])),
    }
    return track, rules, meta
