"""SUMO(실제 교통 시뮬레이터) 위에서 내 차(ego)를 PPO가 운전하는 환경. 한국 교통 규칙을 감시·보상에 반영.

- 도로: OSM 강남역/대치동 → netconvert (우측 통행). 배경 차량은 SUMO 기본 운전자(Krauss)가 몰아요.
- 내 차: 차선 따라가기는 SUMO가 하고, 에이전트는 **가감속 + 차로 변경(왼/유지/오른)** 을 정해요(전술 수준).
  speedMode=0 이라 신호·앞차·제한속도를 SUMO가 대신 지켜 주지 않아요 → 에이전트가 직접 지켜야 해요.
- 규칙(rules/kr.yaml): 안전속도 5030(도로 종류별) + OSM maxspeed, 어린이보호구역 30 km/h(초등학교·유치원·어린이집 300 m),
  빨간불 정지, 우회전 일시정지(빨간불 우회전은 정지선 앞에서 멈춘 뒤만), 충돌·급정거.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym, spaces = None, None

from autodrive import osm
from autodrive.krrules import load as load_kr

from .build_net import AREAS, BUILD, ROOT, build

SCENARIOS = {
    "gangnam": dict(area="gangnam", waypoints=[(37.5046, 127.0250), (37.4979, 127.0276), (37.5006, 127.0364)]),
    "gangnam_reverse": dict(area="gangnam", waypoints=[(37.5006, 127.0364), (37.4979, 127.0276), (37.5046, 127.0250)]),
    "daechi_school": dict(area="daechi", waypoints=[(37.4945, 127.0570), (37.4875, 127.0675)]),
}
DT = 0.5
OBS_DIM = 14


def _traci():
    try:
        import libsumo as traci  # 빠름 (프로세스당 1개)
    except ImportError:  # pragma: no cover
        import traci
    return traci


class SumoKREnv(gym.Env if gym else object):
    metadata = {"render_modes": []}

    def __init__(self, scenario="gangnam", seed=0, max_steps=700, warmup=120, ego_driver="agent", gui=False):
        self.sc = SCENARIOS[scenario]
        self.name = scenario
        self.seed_ = seed
        self.max_steps = max_steps
        self.warmup = warmup
        self.ego_driver = ego_driver  # "agent" | "krauss"(SUMO 기본 운전자 = 기준)
        self.cfg = load_kr()
        self.traci = _traci()
        area = self.sc["area"]
        self.net_file = os.path.join(BUILD, f"{area}.net.xml")
        self.rou_file = os.path.join(BUILD, f"{area}.bg.rou.xml")
        if not os.path.exists(self.net_file):
            build(area)
        import sumolib
        self.net = sumolib.net.readNet(self.net_file, withPrograms=True)
        self.data = json.load(open(os.path.join(ROOT, "korea_roads", "data", f"{AREAS[area]}.json"), encoding="utf-8"))
        self.route = self._ego_route()
        self.route_len = sum(self.net.getEdge(e).getLength() for e in self.route)
        self.limits = self._edge_limits()
        if spaces is not None:
            self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.running = False

    # ---------- 준비 ----------
    def _ego_route(self):
        """웨이포인트 근처 엣지 후보들 중에서 이어지는(끊기지 않는) 가장 짧은 경로를 고른다."""
        cands = [self._near_edges(la, lo) for la, lo in self.sc["waypoints"]]
        best = {e.getID(): (0.0, [e]) for e in cands[0]}
        for nxt in cands[1:]:
            new = {}
            for eb in nxt:
                for ea_id, (cost, pth) in best.items():
                    path, c = self.net.getShortestPath(self.net.getEdge(ea_id), eb, vClass="passenger")
                    if path and (eb.getID() not in new or cost + c < new[eb.getID()][0]):
                        new[eb.getID()] = (cost + c, pth + list(path[1:]))
            best = new
        return [e.getID() for e in min(best.values(), key=lambda v: v[0])[1]]

    def _near_edges(self, lat, lon, r=150):
        x, y = self.net.convertLonLat2XY(lon, lat)
        cand = sorted(self.net.getNeighboringEdges(x, y, r), key=lambda t: t[1])
        out = [e for e, _ in cand if e.allows("passenger")][:6]
        return out

    def _edge_limits(self):
        """엣지별 제한속도(m/s)와 어린이보호구역 여부."""
        lim_cfg = self.cfg["speed_limits_kmh"]
        zone = self.cfg["school_zone"]
        lat0, lon0 = self.data["center"]
        child = [osm.to_local(s["lat"], s["lon"], lat0, lon0) for s in self.data["schools"] if osm.is_child_facility(s)]
        child = np.array(child, float).reshape(-1, 2)
        tagged = {str(w["id"]): w["tags"].get("maxspeed") for w in self.data["ways"]}
        out = {}
        for e in self.net.getEdges():
            if e.getFunction() == "internal":
                continue
            wid = e.getID().lstrip("-").split("#")[0]
            hw = (e.getType() or "highway.residential").split(".")[-1]
            tag = tagged.get(wid)
            try:
                lim = float(str(tag).split()[0])
            except (ValueError, TypeError, IndexError):
                lim = float(lim_cfg.get(hw.replace("_link", ""), 50))
            shp = e.getShape()
            mx, my = shp[len(shp) // 2]
            lon, lat = self.net.convertXY2LonLat(mx, my)
            p = np.array(osm.to_local(lat, lon, lat0, lon0))
            school = bool(len(child)) and float(np.min(np.linalg.norm(child - p, axis=1))) < zone["radius_m"]
            if school:
                lim = min(lim, float(zone["limit_kmh"]))
            out[e.getID()] = (lim / 3.6, school)
        return out

    # ---------- 시뮬레이션 ----------
    def _start(self, seed):
        t = self.traci
        if self.running:
            t.close()
        import sumo
        args = [os.path.join(sumo.SUMO_HOME, "bin", "sumo"), "-n", self.net_file, "-r", self.rou_file,
                "--step-length", str(DT), "--seed", str(seed), "--no-step-log", "true", "--no-warnings", "true",
                "--collision.action", "warn", "--collision.check-junctions", "true", "--time-to-teleport", "120",
                "--ignore-route-errors", "true"]
        t.start(args)
        self.running = True
        for eid, (lim, _) in self.limits.items():
            for k in range(self.net.getEdge(eid).getLaneNumber()):
                t.lane.setMaxSpeed(f"{eid}_{k}", lim)

    def reset(self, *, seed=None, options=None):
        seed = int(self.seed_ if seed is None else seed)
        self.rng = np.random.default_rng(seed)
        self._start(seed)
        t = self.traci
        warm = self.warmup + float(self.rng.uniform(0, 48))  # 신호 위상이 매번 달라지게
        for _ in range(int(warm / DT)):
            t.simulationStep()
        t.route.add("ego_route", self.route)
        t.vehicle.add("ego", "ego_route", departLane="best", departSpeed="0")
        t.vehicle.setColor("ego", (0, 180, 255, 255))
        t.simulationStep()
        if self.ego_driver == "agent":
            t.vehicle.setSpeedMode("ego", 0)          # 안전 기능 끔: 에이전트가 직접 책임
            # 경로상 꼭 필요한 차로 변경(교차로 회전 차로 등)은 SUMO 기본 전략이 하고, 에이전트는 추가로 왼/오른 차로 변경을 요청할 수 있어요
            t.vehicle.setLaneChangeMode("ego", 1621)
        self.steps = 0
        self.stats = dict(speeding_s=0.0, school_speeding_s=0.0, max_over_kmh=0.0, red_runs=0, right_turn_no_stop=0,
                          right_turn_ok=0, collisions=0, harsh_brake=0, signal_stops=0, lane_changes=0, dist_m=0.0)
        self._tls_prev = None
        self._stopped_at = set()
        self.done_reason = ""
        return self._obs(), {}

    def _ego_ok(self):
        return "ego" in self.traci.vehicle.getIDList()

    def _next_tls(self):
        t = self.traci
        nt = t.vehicle.getNextTLS("ego")
        if not nt:
            return None
        tls, link, dist, state = nt[0]
        # 우회전인지: 내 경로의 다음 연결(link) 방향이 r 이면 우회전
        right = False
        try:
            nl = t.vehicle.getNextLinks("ego")
            if nl:
                right = nl[0][6] in ("r", "R")
        except Exception:
            pass
        return tls, link, dist, state, right

    def _obs(self):
        t = self.traci
        if not self._ego_ok():
            return np.zeros(OBS_DIM, np.float32)
        v = t.vehicle.getSpeed("ego")
        edge = t.vehicle.getRoadID("ego")
        lim, school = self.limits.get(edge, (self._last_lim if hasattr(self, "_last_lim") else 13.9, False))
        self._last_lim = lim
        lead = t.vehicle.getLeader("ego", 100)
        gap, lv = (100.0, v) if not lead else (lead[1], t.vehicle.getSpeed(lead[0]))
        tl = self._next_tls()
        tdist, st, right = (200.0, "G", False) if tl is None else (tl[2], tl[3], tl[4])
        red = st in "rR"
        yellow = st in "yY"
        ln = t.vehicle.getLaneIndex("ego")
        nl = max(1, t.edge.getLaneNumber(edge)) if not edge.startswith(":") else 1
        # 현재 차로가 경로에 맞는지(다음 엣지로 이어지는지)
        best = t.vehicle.getBestLanes("ego")
        want = 0
        for b in best:
            if b[0].endswith(f"_{ln}"):
                want = int(np.clip(b[3], -2, 2))  # bestLaneOffset: +면 왼쪽, -면 오른쪽으로 가야 함
        self._cache = dict(v=v, lim=lim, school=school, gap=gap, lv=lv, tdist=tdist, st=st, right=right)
        return np.array([v / 20, lim / 20, min(gap, 100) / 100, (v - lv) / 20, min(tdist, 200) / 200,
                         float(red), float(yellow), float(right), float(school), ln / max(nl - 1, 1),
                         float(nl > ln + 1), float(ln > 0), want / 2, min(1.0, self.stats["dist_m"] / self.route_len)],
                        np.float32)

    def limit_ahead(self, horizon):
        """경로 앞 horizon m 안의 (거리, 제한속도) 목록 — 어린이보호구역 앞에서 미리 감속하려고."""
        t = self.traci
        out = []
        try:
            idx = t.vehicle.getRouteIndex("ego")
            edge = t.vehicle.getRoadID("ego")
            d = (t.lane.getLength(t.vehicle.getLaneID("ego")) - t.vehicle.getLanePosition("ego")) if not edge.startswith(":") else 5.0
            for e in self.route[max(idx, 0) + 1:]:
                if d > horizon:
                    break
                out.append((d, self.limits.get(e, (13.9, False))[0]))
                d += self.net.getEdge(e).getLength()
        except Exception:
            pass
        return out

    def expert_action(self):
        """기준 운전자(규칙 기반): 제한속도(앞 구간 미리 보기)·앞차 간격·빨간불 정지·우회전 일시정지를 지키는 가감속.
        차로는 SUMO 기본 전략에 맡겨요."""
        c = self._cache
        v, lim = c["v"], c["lim"]
        b = 2.5  # 편안한 감속
        vt = lim - 0.5
        for d, l in self.limit_ahead(v * v / (2 * b) + 15):
            vt = min(vt, math.sqrt((l - 0.5) ** 2 + 2 * b * max(d - 3 - v, 0)))
        gap = c["gap"] - 2.0
        vt = min(vt, math.sqrt(max(0, c["lv"] ** 2 + 2 * b * max(gap - 6, 0))))
        self.rule_note = "제한속도"
        if c["st"] in "rRyY":
            key = self._tls_key()
            d = c["tdist"] - 2.5
            if c["right"] and c["st"] in "rR" and key in self._stopped_at:
                vt = min(vt, 15 / 3.6)
                self.rule_note = "우회전 일시정지 후 서행"
                if self.right_turn_must_yield():  # 멈춘 뒤에도 교차로에 들어오는 차가 있으면 양보(정지선에서 대기)
                    vt = min(vt, math.sqrt(2 * b * max(d, 0)))
                    self.rule_note = "우회전 양보 대기"
            elif c["st"] in "rR" or v * v / (2 * 4.5) < d:
                vt = min(vt, math.sqrt(2 * b * max(d, 0)))
                self.rule_note = "우회전 일시정지" if c["right"] else "신호 정지"
        a = (vt - v) / DT
        if vt < 0.1:
            a = -4.5
        return np.array([np.clip(a / (4.5 if a < 0 else 2.6), -1, 1), 0.0], np.float32)

    def junction_conflicts(self, horizon=60.0):
        """교차로 안에서 내 경로와 겹치는 차(SUMO junction foes) → [(foe, 내 진입 거리, 내 진출 거리, foe 도착 s, foe 빠져나감 s)]."""
        t = self.traci
        out = []
        try:
            foes = t.vehicle.getJunctionFoes("ego", horizon)
        except Exception:  # pragma: no cover (SUMO < 1.15)
            return out
        for f in foes:
            fid, ed, fd, ex, fx = f[0], f[1], f[2], f[3], f[4]
            if fid == "ego" or fx < 0:
                continue
            fv = t.vehicle.getSpeed(fid)
            fa = fd / fv if fv > 0.5 else (0.0 if fd < 1.0 else np.inf)
            fe = fx / fv if fv > 0.5 else (np.inf if fd < 1.0 else np.inf)
            out.append((fid, ed, ex, fa, fe))
        return out

    def right_turn_must_yield(self, crawl=3.0, margin=1.5):
        """빨간불 우회전(일시정지 뒤): 내가 서행(crawl m/s)으로 교차로를 빠져나가기 전에 foe가 충돌 지점에 오면 기다림."""
        for _, ed, ex, fa, _ in self.junction_conflicts():
            if fa < ex / crawl + margin:
                return True
        return False

    def _obs_want(self):
        t = self.traci
        ln = t.vehicle.getLaneIndex("ego")
        for b in t.vehicle.getBestLanes("ego"):
            if b[0].endswith(f"_{ln}"):
                return int(b[3])
        return 0

    def _tls_key(self):
        tl = self._next_tls()
        return None if tl is None else (tl[0], int(self.traci.simulation.getTime() // 48))

    def step(self, action):
        t = self.traci
        a = np.clip(np.asarray(action, float).reshape(-1)[:2], -1, 1)
        if not self._ego_ok():
            return np.zeros(OBS_DIM, np.float32), 0.0, True, False, {"stats": self.stats, "reason": "gone"}
        c = self._cache
        v = c["v"]
        if self.ego_driver == "agent":
            acc = a[0] * (2.6 if a[0] >= 0 else 4.5)
            t.vehicle.setSpeed("ego", max(0.0, v + acc * DT))
            if a[1] > 0.5 or a[1] < -0.5:
                ln = t.vehicle.getLaneIndex("ego")
                edge = t.vehicle.getRoadID("ego")
                if not edge.startswith(":"):
                    nl = t.edge.getLaneNumber(edge)
                    tgt = ln + (1 if a[1] > 0.5 else -1)
                    if 0 <= tgt < nl:
                        t.vehicle.changeLane("ego", tgt, 2.0)
                        self.stats["lane_changes"] += 1
        tl_before = self._next_tls()
        d0 = t.vehicle.getDistance("ego")
        t.simulationStep()
        self.steps += 1
        r = 0.0
        term = False
        if "ego" in t.simulation.getCollidingVehiclesIDList():
            self.stats["collisions"] += 1
            r -= 30.0
            term = True
            self.done_reason = "collision"
        if not self._ego_ok():
            arrived = "ego" in t.simulation.getArrivedIDList()
            self.done_reason = self.done_reason or ("arrived" if arrived else "removed")
            self.stats["dist_m"] = self.route_len if arrived else self.stats["dist_m"]
            return np.zeros(OBS_DIM, np.float32), r + (10.0 if arrived else 0.0), True, False, {"stats": self.stats, "reason": self.done_reason}
        d1 = t.vehicle.getDistance("ego")
        prog = max(0.0, d1 - d0)
        self.stats["dist_m"] += prog
        obs = self._obs()
        c = self._cache
        v = c["v"]
        tol = self.cfg["speeding_tolerance_kmh"] / 3.6
        over = v - c["lim"]
        if over > tol:
            self.stats["speeding_s"] += DT
            self.stats["max_over_kmh"] = max(self.stats["max_over_kmh"], over * 3.6)
            if c["school"]:
                self.stats["school_speeding_s"] += DT
        r += prog / 5.0 - (0.3 * max(0, over) * (2 if c["school"] else 1)) * DT
        acc = t.vehicle.getAcceleration("ego")
        if acc < -4.0:
            self.stats["harsh_brake"] += 1
            r -= 0.2
        # 신호 정지/통과 판정
        if tl_before is not None:
            tls, link, dist, st, right = tl_before
            key = (tls, int(t.simulation.getTime() // 48))
            if dist < 6 and v < 0.3 and st in "rRyY":
                if key not in self._stopped_at:
                    self.stats["signal_stops"] += 1
                self._stopped_at.add(key)
            nt = self._next_tls()
            passed = nt is None or nt[0] != tls or nt[2] > dist + 1
            if passed and dist < 15 and st in "rR":
                if right and key in self._stopped_at:
                    self.stats["right_turn_ok"] += 1
                elif right:
                    self.stats["right_turn_no_stop"] += 1
                    r -= 10.0
                else:
                    self.stats["red_runs"] += 1
                    r -= 20.0
        trunc = self.steps >= self.max_steps
        if trunc:
            self.done_reason = "timeout"
        return obs, float(r), term, trunc, {"stats": self.stats, "reason": self.done_reason}

    def close(self):
        if self.running:
            self.traci.close()
            self.running = False
