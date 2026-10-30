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


def apply_geometry_fixes(G, fixes, lat0, lon0, tol=3.0):
    """위성사진으로 확인한 도로 모양 보정(korea_roads.scenarios.GEOMETRY_FIXES): A-B 간선을 A-v1-..-B 로 쪼갬.
    간선 속성(제한속도·보호구역·일방통행 등)은 그대로, 길이만 다시 계산. 적용한 개수를 돌려줌."""
    n_done = 0
    for fx in fixes or []:
        (la_a, lo_a), (la_b, lo_b) = fx["between"]
        a = osm.nearest_node(G, *osm.to_local(la_a, lo_a, lat0, lon0))
        b = osm.nearest_node(G, *osm.to_local(la_b, lo_b, lat0, lon0))
        ok = all(np.hypot(G.nodes[n]["x"] - x, G.nodes[n]["y"] - y) <= tol for n, (x, y) in
                 ((a, osm.to_local(la_a, lo_a, lat0, lon0)), (b, osm.to_local(la_b, lo_b, lat0, lon0))))
        if not ok or not (G.has_edge(a, b) or G.has_edge(b, a)):
            continue
        if fx.get("move_b"):
            la, lo = fx["move_b"]
            x, y = osm.to_local(la, lo, lat0, lon0)
            G.nodes[b].update(x=float(x), y=float(y), lat=la, lon=lo)
            for u, v in list(G.in_edges(b)) + list(G.out_edges(b)):
                G.edges[u, v]["length"] = float(np.hypot(G.nodes[v]["x"] - G.nodes[u]["x"], G.nodes[v]["y"] - G.nodes[u]["y"]))
        vias = []
        for i, (la, lo) in enumerate(fx["via"]):
            k = f"fix_{a}_{b}_{i}"
            x, y = osm.to_local(la, lo, lat0, lon0)
            G.add_node(k, x=float(x), y=float(y), lat=la, lon=lo, signal=False)
            vias.append(k)
        for u, v in ((a, b), (b, a)):
            if not G.has_edge(u, v):
                continue
            attrs = dict(G.edges[u, v])
            G.remove_edge(u, v)
            chain = [u] + (vias if u == a else vias[::-1]) + [v]
            for p, q in zip(chain[:-1], chain[1:]):
                G.add_edge(p, q, **{**attrs, "length": float(np.hypot(G.nodes[q]["x"] - G.nodes[p]["x"],
                                                                        G.nodes[q]["y"] - G.nodes[p]["y"]))})
        n_done += 1
    return n_done


def remove_detours(G, path, near_m=35.0, slack_m=30.0):
    """경로 다듬기: 경유점 노드에 '찍고 돌아 나오는' 군더더기(교차로 안 U자·고리)를 없앤다.
    경로 위 두 노드가 직선으로 near_m 안에 있는데 경로로는 slack_m 넘게 더 돌아가면, 둘 사이를
    실제 도로망 최단 경로(일방통행 지킴)로 바꾼다. 예: 강남역 역방향(역삼→강남역→신논현)에서 경유점이
    교차로 남서쪽 노드에 붙어 교차로 안에서 한 바퀴(270°) 돌던 것."""
    import networkx as nx
    path = list(path)
    for _ in range(10):
        P = np.array([[G.nodes[k]["x"], G.nodes[k]["y"]] for k in path])
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        changed = False
        for i in range(len(path)):
            # 가장 멀리 떨어진 j 부터 (가장 큰 고리를 먼저 자름)
            for j in range(len(path) - 1, i + 1, -1):
                e = float(np.linalg.norm(P[j] - P[i]))
                if e > near_m or cum[j] - cum[i] < e + slack_m:
                    continue
                try:
                    L, seg = nx.single_source_dijkstra(G, path[i], path[j], weight="length")
                except nx.NetworkXNoPath:
                    continue
                if L < cum[j] - cum[i] - slack_m / 2:
                    path = path[:i] + seg + path[j + 1:]
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    return path


def route_turning(P):
    """경로 전체 회전량(도, 왼쪽 +) 과 1 m 표본 사이 최대 꺾임. 고리가 있으면 |회전량| 이 크게 나옴."""
    seg = np.diff(np.asarray(P), axis=0)
    h = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    return float(np.degrees(h[-1] - h[0])), float(np.degrees(np.ptp(h)))


def build_route(name: str, scenario: dict, signal_seed: int = 0):
    """시나리오(웨이포인트) → (Track, TrafficRules, meta). 데이터는 캐시에서 읽는다(없으면 받는다)."""
    import networkx as nx
    if scenario.get("bbox"):   # 시골길: OSM 공식 API 작은 영역 캐시
        data = osm.fetch_osm_api(scenario.get("data", name), tuple(scenario["bbox"]))
    else:
        data = osm.fetch(scenario.get("data", name), *scenario["center"], scenario["radius"])
    G = osm.build_graph(data)
    lat0, lon0 = data["center"]
    from korea_roads.scenarios import GEOMETRY_FIXES
    n_fix = apply_geometry_fixes(G, GEOMETRY_FIXES.get(scenario.get("data", name), []), lat0, lon0)
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
    path = remove_detours(G, path)
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
    sig_s = [node_d[i] for i, k in enumerate(path) if G.nodes[k]["signal"]]
    # 경로가 교차로 한가운데 신호 노드를 안 지나고 우회전 차로(연결로)로 돌아도, 15 m 안의 신호 노드는 그 교차로 신호로 본다
    # (remove_detours 로 강남역 역방향 고리를 없앤 뒤 우회전 신호가 사라지지 않게)
    for k in G.nodes:
        if G.nodes[k]["signal"] and k not in path:
            dd = np.hypot(P[:, 0] - G.nodes[k]["x"], P[:, 1] - G.nodes[k]["y"])
            j = int(np.argmin(dd))
            if dd[j] < 15.0:
                sig_s.append(float(s[j]))
    sig_s = sorted(sig_s)
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
    if scenario.get("rural"):
        # 시골길은 노드 간격이 길어서(간선 중점 기준이면 놓침) 1 m 표본마다 학교·유치원까지 거리로 보호구역을 다시 칠함
        from .krrules import load as _load
        zr = float(_load()["school_zone"]["radius_m"])
        ch = [osm.to_local(c["lat"], c["lon"], lat0, lon0) for c in data["schools"] if osm.is_child_facility(c)]
        if ch:
            C = np.array(ch, float).reshape(-1, 2)
            school = school | (np.min(np.linalg.norm(P[:, None, :] - C[None], axis=2), axis=1) < zr)
        track, rules, rmeta = _rural_rules(P, s, edges, ei, school, sigs, scenario, signal_seed)
    else:
        rules = TrafficRules(1.0, limit, school, twoway, lane_ref, sigs, closed=False, length=track.length)
        rmeta = {}
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
        "school_zone_m": int(np.asarray(rules.school).sum()), "limits_kmh": sorted(set(int(round(x / KMH)) for x in rules.limit_mps)),
        "kakao": osm.kakao_link(scenario["title"].split(":")[0], float(mid_lat), float(mid_lon)),
        "naver": osm.naver_link(scenario["title"], float(mid_lat), float(mid_lon)),
        "signals_latlon": sig_ll, "graph": G, "data": data,
        "school_sources": sorted(set(e["school_source"] for e in edges if e["school_zone"])),
        "attribution": data.get("attribution"), "geometry_fixes": n_fix, **rmeta,
    }
    return track, rules, meta


GRAVEL = ("unpaved", "gravel", "fine_gravel", "dirt", "compacted", "ground", "earth", "mud", "sand")


def _rural_rules(P, s, edges, ei, school, sigs, scenario, seed):
    """실제 OSM 시골길 → 시골길 규칙(rules/kr.yaml rural):
    - 제한속도: OSM maxspeed 우선, 없으면 중앙선 있는 2차로(secondary·tertiary, lanes≥2) 60, 좁은 길
      (residential·unclassified·track·lanes=1) 40, 어린이보호구역(초등학교·유치원 300 m) 30
    - 노면: surface 가 비포장 계열이거나 track(포장 정보 없음)이면 자갈/비포장(마찰 0.45~0.6)
    - 폭: 경로 길이의 절반 이상이 좁은 길이면 1차로 폭(4.6~5.0 m, 중앙선 없음), 아니면 2차로(중앙선, 앞지르기 허용 구간 계산)
    - 밤/차량 흐름: 시나리오의 night, traffic 시드로 rural_traffic() (농기계 없음, 일반 차량 흐름)
    - 언덕 꼭대기: OSM 에 고도가 없어 넣지 않음(굽이 시야는 곡률로 계산)"""
    from .krrules import load
    from .rural import passing_zones, rural_traffic
    cfg = load()["rural"]
    rng = np.random.default_rng(seed + 7)
    m = len(s)
    narrow_e = []
    for e in edges:
        hw = e["highway"].replace("_link", "")
        lanes = str(e.get("lanes") or "")
        narrow_e.append(hw in ("residential", "unclassified", "track", "living_street") or lanes == "1")
    narrow_s = np.array([narrow_e[i] for i in ei], bool)
    narrow = bool(narrow_s.mean() >= 0.5)
    lim = np.empty(m)
    for j, i in enumerate(ei):
        e = edges[i]
        try:
            v = float(str(e.get("maxspeed_tag") or "").split()[0])
        except (ValueError, IndexError):
            v = cfg["narrow_limit_kmh"] if narrow_e[i] else cfg["general_limit_kmh"]
        lim[j] = v
    lim = np.where(school, np.minimum(lim, cfg["village_school_limit_kmh"]), lim) * KMH
    surf = np.array([(str(edges[i].get("surface") or "") in GRAVEL) or (edges[i]["highway"] == "track" and not edges[i].get("surface"))
                     for i in ei], bool)
    mu = np.where(surf, rng.uniform(0.45, 0.6), 0.9)
    width = float(rng.uniform(4.6, 5.0)) if narrow else 2 * LANE_W
    track = Track(P, width=width, closed=False, name=scenario.get("title", "rural"))
    lane_ref = -min(0.5, width / 2 - 0.9 - 0.6) if narrow else -LANE_W / 2
    night = bool(scenario.get("night", False))
    rules = TrafficRules(1.0, lim, school, np.full(m, not narrow), np.full(m, lane_ref), sigs, closed=False,
                         length=track.length, mu=mu, gravel=surf, light=0.15 if night else 1.0,
                         roadside_clear_m=float(scenario.get("roadside_clear_m", 3.0)), crests=[], rural=True)
    rules.passing_ok = passing_zones(track, school, [], narrow)
    traffic = rural_traffic(np.random.default_rng(scenario.get("traffic_seed", seed)), track, width, narrow, lane_ref,
                            scenario.get("difficulty", 1.0)) if scenario.get("traffic", True) else []
    return track, rules, {"rural_narrow": narrow, "gravel_m": int(surf.sum()), "traffic": traffic, "night": night,
                          "passing_ok_m": int(rules.passing_ok.sum())}
