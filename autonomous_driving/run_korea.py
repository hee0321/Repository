"""실제 한국 도로(OpenStreetMap)에서 기준 제어기(또는 PPO)를 달려 보고, 지도 위 그림/GIF와 결과를 남긴다.

예) python run_korea.py                         # 모든 시나리오, Pure Pursuit
    python run_korea.py --scenario gangnam_station --gif
    python run_korea.py --policy weights/best.zip
데이터 © OpenStreetMap contributors (ODbL). 카카오맵/네이버맵 링크는 '보기'용일 뿐, 데이터는 쓰지 않는다.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from autodrive import osm  # noqa: E402
from autodrive.bench import make_policy, run_episode  # noqa: E402
from autodrive.roads import build_route  # noqa: E402
from autodrive.scenarios import make_env  # noqa: E402
from korea_roads.scenarios import SCENARIOS  # noqa: E402

plt.rcParams["font.family"] = ["NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"]


def draw_map(ax, G, track, rules, meta, lim=None):
    for a, b, e in G.edges(data=True):
        major = e["highway"].startswith(("trunk", "primary", "secondary"))
        ax.plot([G.nodes[a]["x"], G.nodes[b]["x"]], [G.nodes[a]["y"], G.nodes[b]["y"]],
                color="#f4a259" if e["school_zone"] else "#b0b0b0", lw=2.2 if major else 0.8, zorder=1)
    ax.plot(track.center[:, 0], track.center[:, 1], color="#3d5a80", lw=5, alpha=0.25, zorder=2)
    for sg in rules.signals:
        ax.plot(*track.point_at(sg.s), "s", color="crimson", ms=7, zorder=6)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if lim:
        ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--policy", default="pure_pursuit")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--out", default="media")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    act = make_policy(a.policy)
    tag = "ppo" if a.policy != "pure_pursuit" else "pure_pursuit"
    results = {}
    for name in ([a.scenario] if a.scenario else SCENARIOS):
        track, rules, meta = build_route(name, SCENARIOS[name])
        env = make_env({"korea": name})
        r, fr = run_episode(env, act, record=True)
        results[name] = {"title": meta["title"], "streets": meta["streets"], "length_m": meta["length_m"],
                         "signals": meta["n_signals"], "school_zone_m": meta["school_zone_m"],
                         "limits_kmh": meta["limits_kmh"], "kakao": meta["kakao"], "naver": meta["naver"], **r}
        print(f"\n[{name}] {meta['title']}")
        print(f"  경로: {' → '.join(meta['streets'])} ({meta['length_m']} m, 신호 {meta['n_signals']}곳, 어린이보호구역 {meta['school_zone_m']} m)")
        print(f"  카카오맵: {meta['kakao']}\n  네이버지도: {meta['naver']}")
        print(f"  결과: 완주 {r['completion'] * 100:.0f}%, 이탈 {r['crashed']}, {r['time_s']} s, 평균 {r['mean_speed_kmh']} km/h, "
              f"차선 편차 평균 {r['mean_lane_dev_m']} m / 최대 {r['max_lane_dev_m']} m")
        print(f"  위반: 과속 {r['speeding_s']} s (최대 +{r['max_over_kmh']} km/h), 어린이보호구역 과속 {r['school_speeding_s']} s, "
              f"신호 위반 {r['red_runs']}회 (신호 정지 {r['signal_stops']}회), 중앙선 침범 {r['centerline_s']} s")
        # --- 그림: 지도 + 속도/제한속도 ---
        xy = np.array([[f[0], f[1]] for f in fr]); v = np.array([f[3] for f in fr]) * 3.6
        limk = np.array([f[8] for f in fr]) * 3.6; s = np.array([f[9] for f in fr]); t = np.array([f[6] for f in fr])
        pad = 120
        lim = ((track.center[:, 0].min() - pad, track.center[:, 0].max() + pad),
               (track.center[:, 1].min() - pad, track.center[:, 1].max() + pad))
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=100, gridspec_kw={"width_ratios": [1.2, 1]})
        draw_map(ax, meta["graph"], track, rules, meta, lim)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=v, cmap="viridis", s=4, zorder=4, vmin=0, vmax=60)
        fig.colorbar(sc, ax=ax, shrink=0.7, label="speed (km/h)")
        ax.plot(*xy[0], "o", color="green", ms=9, zorder=7); ax.plot(*xy[-1], "X", color="black", ms=9, zorder=7)
        ax.set_title(f"{name} (OSM)  orange = school zone, red = signal", fontsize=9)
        ax2.plot(t, v, label="speed")
        ax2.plot(t, limk, "--", label="limit")
        ax2.set_xlabel("time (s)"); ax2.set_ylabel("km/h"); ax2.legend(loc="lower right")
        ax2.set_title(f"done {r['completion'] * 100:.0f}%  crash={r['crashed']}  red-run={r['red_runs']}  "
                      f"over-limit {r['speeding_s']}s", fontsize=9)
        fig.text(0.01, 0.01, "Map data © OpenStreetMap contributors (ODbL)", fontsize=7)
        fig.tight_layout()
        png = os.path.join(a.out, f"korea_{name}_{tag}.png")
        fig.savefig(png); plt.close(fig)
        print("  그림:", png)
        if a.gif:
            fig, ax = plt.subplots(figsize=(5.5, 5), dpi=75)
            draw_map(ax, meta["graph"], track, rules, meta)
            car, = ax.plot([], [], "o", color="tab:blue", ms=7, zorder=8)
            trail, = ax.plot([], [], color="tab:green", lw=2, zorder=7)
            title = ax.set_title("", fontsize=8)
            fig.text(0.01, 0.01, "© OpenStreetMap contributors", fontsize=6)
            idx = list(range(0, len(fr), 20))

            def upd(k):
                i = idx[k]
                x, y = xy[i]
                car.set_data([x], [y]); trail.set_data(xy[:i + 1, 0], xy[:i + 1, 1])
                ax.set_xlim(x - 160, x + 160); ax.set_ylim(y - 140, y + 140)
                sig = env.rules.signals
                title.set_text(f"{name} t={t[i]:5.1f}s v={v[i]:4.1f} km/h limit {limk[i]:.0f}")
                return car, trail, title
            anim = FuncAnimation(fig, upd, frames=len(idx))
            gif = os.path.join(a.out, f"korea_{name}_{tag}.gif")
            anim.save(gif, writer=PillowWriter(fps=12)); plt.close(fig)
            print("  GIF:", gif, f"({os.path.getsize(gif) / 1e6:.2f} MB)")
    os.makedirs("results", exist_ok=True)
    with open(f"results/korea_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
