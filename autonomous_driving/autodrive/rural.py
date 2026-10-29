"""시골길(농로·마을길·지방도) 절차 생성기.

특징: 좁은 1차로(중앙선 없음, 마주 오면 오른쪽으로 비켜 서로 양보), 비포장·자갈 구간(마찰↓·미끄러짐),
굽이 많고 앞이 잘 안 보임(굽이·언덕 꼭대기), 가로등 없는 밤(센서 거리↓·잡음↑),
느린 트랙터, 보행자(길가·길 건너기), 길가 위험(농수로·낭떠러지 = 길 밖으로 나가면 사고, 돌·물건).
"""
from __future__ import annotations

import numpy as np

from .krrules import load
from .rules import KMH, LANE_W, TrafficRules
from .track import Track


def make_rural_road(seed: int, difficulty: float = 1.0, night: bool | None = None, length: float = 900.0,
                    traffic: bool = True):
    rng = np.random.default_rng(seed)
    cfg = load()["rural"]
    # 1) 굽이: 길이 25~110 m 구간마다 곡률을 바꾸고(최소 반경 약 18 m), 부드럽게 이음
    n = int(length)
    kap = np.zeros(n)
    i = 0
    while i < n:
        seg = int(rng.uniform(25, 110))
        k = float(np.clip(rng.normal(0, 0.018 * difficulty), -1 / 18, 1 / 18))
        if rng.random() < 0.3:
            k = 0.0
        kap[i:i + seg] = k
        i += seg
    kap = np.convolve(kap, np.ones(15) / 15, mode="same")
    h = np.cumsum(kap)
    P = np.stack([np.cumsum(np.cos(h)), np.cumsum(np.sin(h))], axis=1)
    narrow = rng.random() < 0.7
    width = float(rng.uniform(4.6, 5.4)) if narrow else 2 * LANE_W
    track = Track(P, width=width, closed=False, name=f"rural_{seed}")
    m = int(np.ceil(track.length)) + 1
    # 2) 제한속도: 중앙선 있는 지방도 60, 좁은 1차로 40, 마을 어린이보호구역 30
    base = cfg["narrow_limit_kmh"] if narrow else cfg["general_limit_kmh"]
    limit = np.full(m, base * KMH)
    school = np.zeros(m, bool)
    if rng.random() < 0.5:
        a = int(rng.uniform(0.3, 0.6) * m)
        school[a:a + int(rng.uniform(120, 220))] = True
        limit[school] = cfg["village_school_limit_kmh"] * KMH
    # 3) 노면: 비포장·자갈 구간
    mu = np.full(m, 0.9)
    gravel = np.zeros(m, bool)
    for _ in range(int(rng.integers(0, 3) + (1 if narrow else 0))):
        a = int(rng.uniform(0, 0.8) * m)
        b = a + int(rng.uniform(80, 250))
        gravel[a:b] = True
        mu[a:b] = rng.uniform(0.45, 0.6)
    # 4) 시야: 길옆이 막힌 정도, 언덕 꼭대기, 밤
    night = bool(rng.random() < 0.4) if night is None else night
    light = float(rng.uniform(0.05, 0.25)) if night else 1.0
    crests = [(float(rng.uniform(0.1, 0.9) * track.length), float(rng.uniform(35, 70)))
              for _ in range(int(rng.integers(0, 3)))]
    lane_ref = -min(0.5, width / 2 - 0.9 - 0.6) if narrow else -LANE_W / 2
    rules = TrafficRules(1.0, limit, school, np.full(m, not narrow), np.full(m, lane_ref), [], closed=False,
                         length=track.length, mu=mu, gravel=gravel, light=light,
                         roadside_clear_m=float(rng.uniform(2.0, 5.0)), crests=crests, rural=True)
    # 5) 길 위의 것들
    obs = []
    if traffic:
        half = width / 2
        L = track.length
        for _ in range(int(rng.integers(0, 2) + difficulty)):        # 트랙터·경운기 (느림)
            obs.append(dict(kind="tractor", s=float(rng.uniform(80, 0.7 * L)), d=lane_ref, v=float(rng.uniform(2.5, 4.5)),
                            r=0.9, l=4.0))
        for _ in range(int(rng.integers(0, 3))):                     # 마주 오는 차 (서로 오른쪽으로 비킴)
            obs.append(dict(kind="oncoming", s=float(rng.uniform(0.4, 1.2) * L), d=(half - 1.0) if narrow else LANE_W / 2,
                            v=-float(rng.uniform(5.0, 8.0)), r=0.9, l=4.2))
        for _ in range(int(rng.integers(0, 3))):                     # 길가 보행자 / 길 건너는 보행자
            side = -1 if rng.random() < 0.7 else 1
            s0 = float(rng.uniform(60, 0.9 * L))
            if rng.random() < 0.3:
                obs.append(dict(kind="ped", s=s0, d=side * (half - 0.3), vd=1.2, d_to=-side * (half - 0.3), r=0.35, l=0.6))
            else:
                obs.append(dict(kind="ped", s=s0, d=side * (half - 0.3), r=0.35, l=0.6))
        for _ in range(int(rng.integers(0, 2))):                     # 길가 돌·농기구 같은 물건
            obs.append(dict(kind="object", s=float(rng.uniform(60, 0.9 * L)), d=-(half - 0.4), r=0.4, l=0.8))
    return track, rules, obs
