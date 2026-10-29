"""한 에피소드를 그려서 PNG(주행 궤적) / GIF(애니메이션)로 저장한다. 화면 없이(headless) 동작.

예) python render_episode.py                       # Pure Pursuit, media/ 에 저장
    python render_episode.py --policy weights/ppo_autodrive.zip   # 학습한 PPO 정책
    python render_episode.py --view driver             # 운전석 1인칭 시점 MP4/GIF
    python render_episode.py --view main               # 운전석 중심 화면(HUD·미니맵, 전기차)
    python render_episode.py --view all                # 2×2: 운전석·지도·모터룸(단순 모델 시각화)·위성
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from autodrive import AutoDriveEnv, PurePursuitController  # noqa: E402


def rollout(policy_path=None, laps=1, seed=0, recorder=False):
    env = AutoDriveEnv(target_laps=laps, max_steps=6000, seed=seed)
    if policy_path:
        from stable_baselines3 import PPO
        model = PPO.load(policy_path, device="cpu")
        act = lambda o: model.predict(o, deterministic=True)[0]
        name = "PPO"
    else:
        ctrl = PurePursuitController(env)
        act = ctrl.act
        name = "Pure Pursuit"
    obs, info = env.reset(seed=seed)
    frames = []
    rec = None
    if recorder:
        from autodrive.views import StateRecorder
        rec = StateRecorder(env)
    done = False
    while not done:
        a = act(obs)
        obs, r, term, trunc, info = env.step(a)
        if rec is not None:
            rec.record(env, a, {"mode": "BASELINE" if not policy_path else "AUTONOMOUS"})
        c = env.car
        frames.append((c.x, c.y, c.yaw, c.v, env.lidar_ranges.copy(), info["laps"], info["t"], info["crashed"]))
        done = term or trunc
    if recorder:
        return env, frames, name, rec
    return env, frames, name


def draw_track(ax, track):
    # 반시계 트랙: 오른쪽 경계 = 바깥, 왼쪽 경계 = 안쪽. 바깥을 회색(도로)으로 칠하고 안쪽을 비운다.
    ax.fill(np.r_[track.right[:, 0], track.right[0, 0]], np.r_[track.right[:, 1], track.right[0, 1]],
            color="#d9d9d9", zorder=0)
    ax.fill(np.r_[track.left[:, 0], track.left[0, 0]], np.r_[track.left[:, 1], track.left[0, 1]],
            color="white", zorder=1)
    for b in (track.left, track.right):
        ax.plot(np.r_[b[:, 0], b[0, 0]], np.r_[b[:, 1], b[0, 1]], "k-", lw=1.2, zorder=2)
    ax.plot(track.center[:, 0], track.center[:, 1], "--", color="#999", lw=0.6, zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")


def car_poly(x, y, yaw, L=4.0, W=1.8):
    pts = np.array([[-0.6, -W / 2], [L - 0.6, -W / 2], [L - 0.6, W / 2], [-0.6, W / 2]])
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return pts @ R.T + [x, y]


def save_png(env, frames, name, path):
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=110)
    draw_track(ax, env.track)
    xy = np.array([[f[0], f[1]] for f in frames])
    v = np.array([f[3] for f in frames])
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=v, cmap="viridis", s=3, zorder=3)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="speed (m/s)")
    x, y, yaw, _, rays, *_ = frames[len(frames) // 3]
    for ang, r in zip(env.lidar.angles, rays):
        ax.plot([x, x + r * np.cos(yaw + ang)], [y, y + r * np.sin(yaw + ang)], color="tab:red", lw=0.6, zorder=4)
    ax.add_patch(plt.Polygon(car_poly(x, y, yaw), color="tab:blue", zorder=5))
    ax.set_title(f"{name}: {frames[-1][5]:.2f} laps, {frames[-1][6]:.1f} s, crashed={frames[-1][7]}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_gif(env, frames, name, path, every=6, fps=15):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
    draw_track(ax, env.track)
    rays = [ax.plot([], [], color="tab:red", lw=0.6, zorder=4)[0] for _ in env.lidar.angles]
    trail, = ax.plot([], [], color="tab:green", lw=1.0, zorder=3)
    car = plt.Polygon(car_poly(0, 0, 0), color="tab:blue", zorder=5)
    ax.add_patch(car)
    title = ax.set_title("")
    idx = list(range(0, len(frames), every))
    xs = np.array([f[0] for f in frames]); ys = np.array([f[1] for f in frames])

    def upd(k):
        i = idx[k]
        x, y, yaw, v, rr, laps, t, crashed = frames[i]
        car.set_xy(car_poly(x, y, yaw))
        for ln, ang, r in zip(rays, env.lidar.angles, rr):
            ln.set_data([x, x + r * np.cos(yaw + ang)], [y, y + r * np.sin(yaw + ang)])
        trail.set_data(xs[max(0, i - 120):i + 1], ys[max(0, i - 120):i + 1])
        title.set_text(f"{name}  t={t:4.1f}s  v={v * 3.6:4.1f}km/h  lap {laps:.2f}")
        return [car, trail, title, *rays]

    anim = FuncAnimation(fig, upd, frames=len(idx), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None, help="PPO 가중치(.zip). 없으면 Pure Pursuit")
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--out", default="media")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--view", default="top", choices=["top", "main", "driver", "engine", "motor", "satellite", "both", "all"])
    ap.add_argument("--clip-seconds", type=float, default=18.0)
    ap.add_argument("--fps", type=float, default=25.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.view != "top":
        from autodrive.views import pick_window, render_views
        env, frames, name, rec = rollout(a.policy, a.laps, recorder=True)
        t0, t1 = pick_window(rec.rows, a.clip_seconds)
        tag = "ppo" if a.policy else "pure_pursuit"
        for o in render_views(env, rec.rows, rec.kinds, a.view, os.path.join(a.out, f"{tag}_{a.view}"), fps=a.fps,
                              t0=t0, t1=t1, gif=not a.no_gif):
            print("저장:", o, f"({os.path.getsize(o) / 1e6:.2f} MB)")
        raise SystemExit(0)
    env, frames, name = rollout(a.policy, a.laps)
    tag = "ppo" if a.policy else "pure_pursuit"
    png = os.path.join(a.out, f"{tag}_trajectory.png")
    save_png(env, frames, name, png)
    print("저장:", png)
    if not a.no_gif:
        gif = os.path.join(a.out, f"{tag}_lap.gif")
        save_gif(env, frames, name, gif)
        print("저장:", gif, f"({os.path.getsize(gif) / 1e6:.2f} MB)")
