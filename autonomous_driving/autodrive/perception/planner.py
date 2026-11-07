"""인식·예측 결과로 속도를 줄이거나 멈추는 안전 계획기 (조향은 비전 정책 그대로).

규칙 (모두 카메라 인식 결과와 내 차 계기만 사용):
1. 예측 충돌: 추적 물체의 3초 예측 위치(± 2σ 와 물체 크기)가 내 차 예상 경로(지금 조향·속도로 3초)와 겹치면
   그 지점 앞 3 m 에 멈출 감속도를 요구.
2. 빈 공간(free space): 내 경로 띠 위의 바닥 픽셀이 도로가 아닌 물체(차·사람·물체) 또는 '모름'(확신도 < 0.5)이면
   그 거리 안에 멈출 수 있는 속도로 제한 — 분류 못 한 물체도 막힘으로 봄(보수적).
3. 신호: 빨강(또는 멈출 수 있는 노랑) 등이 보이면 정지선(신호등 22 m 앞, 장면 가정)에 멈춤. 등이 잠깐 안 보여도 1초 기억.
4. 가려짐: 경로 옆 1.2~5 m 에 서 있는 차(속도 < 1 m/s)가 25 m 안에 있으면 그 뒤에서 사람이 나올 수 있어 30 km/h 제한.
"""
from __future__ import annotations

import numpy as np

from ..vision.camera import LBL
from .track import SIZE

A_MAX = 6.0          # 차 최대 감속
A_COMF = 3.0
HALF_W = 0.95        # 내 차 반폭
FRONT = 3.25         # 뒷바퀴 축 → 앞 범퍼


def ego_path(v, steer, wheelbase=2.5, horizon=3.0, dt=0.25):
    """지금 조향각·속도로 굴린 내 차(뒷바퀴 축) 경로. 반환 [(t, 앞, 왼, 진행 거리)]."""
    x = y = yaw = s = 0.0
    out = []
    vv = max(v, 1.0)
    for k in range(1, int(horizon / dt) + 1):
        yaw += vv / wheelbase * np.tan(steer) * dt
        x += vv * np.cos(yaw) * dt
        y += vv * np.sin(yaw) * dt
        s += vv * dt
        out.append((k * dt, x, y, s, yaw))
    return out


class SafetyPlanner:
    def __init__(self, geo, odo):
        self.geo, self.odo = geo, odo
        self.sig_mem = None   # (상태, 정지선 거리, 남은 시간)
        self.reason = ""

    def free_distance(self, cls_map, conf, path, max_d=60.0):
        """경로 띠(폭 1.9 m) 위로 1 m 마다 바닥 픽셀을 보고 처음 막힌 거리(앞 범퍼 기준)."""
        H, W = cls_map.shape
        g = self.geo
        cam_x = 2.45
        pts = []
        # 경로를 1 m 간격으로 늘려 씀 (3초 경로가 짧으면 직선 연장)
        P = np.array([[0.0, 0.0]] + [[p[1], p[2]] for p in path])
        seg = np.diff(P, axis=0)
        L = np.r_[0, np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))]
        heading = np.arctan2(seg[-1, 1], seg[-1, 0]) if len(seg) else 0.0
        for d in np.arange(FRONT + 1, FRONT + max_d, 1.0):
            if d <= L[-1]:
                x, y = np.interp(d, L, P[:, 0]), np.interp(d, L, P[:, 1])
                i = min(np.searchsorted(L, d), len(seg)) - 1
                hd = np.arctan2(seg[max(i, 0), 1], seg[max(i, 0), 0])
            else:
                x = P[-1, 0] + (d - L[-1]) * np.cos(heading)
                y = P[-1, 1] + (d - L[-1]) * np.sin(heading)
                hd = heading
            blocked = 0
            for off in (-0.8, 0.0, 0.8):
                fx, ly = x - off * np.sin(hd) - cam_x, y + off * np.cos(hd)
                if fx < 1:
                    continue
                # 바닥 점 → 픽셀
                p = np.radians(g.m.pitch_deg)
                up = -g.m.height
                zc = fx * np.cos(p) - up * np.sin(p)
                yc = -(fx * np.sin(p) + up * np.cos(p))
                u = g.cx + g.f * (-ly) / zc
                v = g.cy + g.f * yc / zc
                if not (0 <= u < W and 0 <= v < H):
                    continue
                c = cls_map[int(v), int(u)]
                if c in (LBL["car"], LBL["ped"], LBL["object"]) or (conf[int(v), int(u)] < 0.5 and c != LBL["other"]):
                    blocked += 1
            if blocked >= 1:
                return d - FRONT, (x, y)
            pts.append((x, y))
        return max_d, None

    def plan(self, v, steer, tracks, lamps, cls_map, conf, dt):
        path = ego_path(v, steer)
        a_req, why = 0.0, []
        # 1) 예측 충돌
        conflicts = []
        for tr in tracks:
            if not tr.confirmed and tr.cls != "ped":
                continue
            L_, W_ = SIZE.get(tr.cls, (2, 2))
            for (t, pw, Pc), (_, ex, ey, es, eyaw) in zip(tr.predict_traj(), path):
                pe = self.odo.to_ego(pw)
                sig = float(np.sqrt(max(np.linalg.eigvalsh(Pc).max(), 0)))
                r = min(W_ / 2 + 2 * sig, 2.0) + HALF_W + 0.2   # 불확실성은 2 m 까지만 넓힘 (옆 차로 마주 오는 차 오경보 방지)
                # 내 차 몸통(뒷축~앞범퍼) 을 경로 위 점 3개로
                hit = False
                ext = (L_ / 2 - W_ / 2) if tr.cls == "car" else 0.0
                for back in (-0.5, 1.6, FRONT):   # 내 차 몸통 점 3개 (뒷축 기준)
                    bx, by = ex + back * np.cos(eyaw), ey + back * np.sin(eyaw)
                    if np.hypot(pe[0] - bx, pe[1] - by) < r + ext:
                        hit = True
                        break
                if hit:
                    gap = max(np.hypot(pe[0], pe[1]) - FRONT - (L_ / 2 if tr.cls == "car" else W_ / 2) - 3.0, 0.3)
                    conflicts.append((gap, tr.tid, t))
                    break
        if conflicts:
            gap, tid, t = min(conflicts)
            a = v * v / (2 * gap)
            if a > a_req:
                a_req = a
            why.append(f"예측 충돌 #{tid} {t:.1f}s 뒤, {gap:.0f} m")
        # 2) 빈 공간
        free, _ = self.free_distance(cls_map, conf, path)
        self.free = free
        v_cap = np.sqrt(max(2 * A_COMF * max(free - 3.0, 0), 0))
        if v > v_cap + 0.5:
            a = min((v * v - v_cap * v_cap) / (2 * max(free - 3.0, 0.5)), A_MAX)
            if a > a_req:
                a_req = a
                why.append(f"빈 공간 {free:.0f} m")
        # 3) 신호
        red = [l for l in lamps if l.extra["state"] in ("red", "yellow") and 5 < l.fwd < 100 and abs(l.left) < 8]
        grn = [l for l in lamps if l.extra["state"] == "green" and 5 < l.fwd < 100 and abs(l.left) < 8]
        if red:
            l = min(red, key=lambda d: d.fwd)
            self.sig_mem = [l.extra["state"], l.fwd + 2.45 - 22.0 - FRONT, 1.0]
        elif grn:
            self.sig_mem = None
        elif self.sig_mem is not None:
            self.sig_mem[1] -= v * dt
            self.sig_mem[2] -= dt
            if self.sig_mem[2] <= 0:
                self.sig_mem = None
        if self.sig_mem is not None:
            st, d_stop, _ = self.sig_mem
            if d_stop > -1.0:
                a = v * v / (2 * max(d_stop - 1.0, 0.3))
                if st == "red" or a <= A_COMF:
                    if a > a_req or (d_stop < 3 and v < 1):
                        a_req = max(a, a_req, 0.5 if v < 1 else 0)
                    why.append(f"{'빨간' if st == 'red' else '노란'}불, 정지선 {d_stop:.0f} m")
        # 4) 가려짐
        v_lim = None
        for tr in tracks:
            if tr.cls != "car" or not tr.confirmed:
                continue
            pe = self.odo.to_ego(tr.x[:2])
            if np.hypot(*tr.x[2:]) < 1.0 and 3 < pe[0] < 25 and 1.2 < abs(pe[1]) < 5:
                v_lim = 8.3
        if v_lim is not None and v > v_lim:
            a = min((v - v_lim) / 1.0, A_COMF)
            if a > a_req:
                a_req = a
            why.append("서 있는 차 뒤 가려짐 → 30 km/h")
        self.reason = ", ".join(why)
        return a_req, path
