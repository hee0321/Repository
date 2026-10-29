"""Pure Pursuit 기준 제어기로 트랙을 여러 바퀴 돌고 결과를 출력한다.

예) python run_baseline.py --laps 3
"""
import argparse
import time

import numpy as np

from autodrive import AutoDriveEnv, PurePursuitController


def run(laps: int = 3, v_max: float = 15.0, lidar_noise: float = 0.0, seed: int = 0, verbose=True):
    env = AutoDriveEnv(target_laps=laps, max_steps=20000, lidar_noise=lidar_noise, seed=seed)
    ctrl = PurePursuitController(env, v_max=v_max)
    obs, info = env.reset(seed=seed)
    offsets, speeds, min_lidar = [], [], np.inf
    lap_times, next_lap = [], 1
    t0 = time.time()
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(ctrl.act(obs))
        offsets.append(abs(info["offset"]))
        speeds.append(info["speed"])
        min_lidar = min(min_lidar, float(env.lidar_ranges.min()))
        if info["laps"] >= next_lap:
            lap_times.append(info["t"])
            next_lap += 1
        done = term or trunc
    res = {
        "laps_completed": int(np.floor(info["laps"] + 1e-9)),
        "crashed": info["crashed"],
        "sim_time_s": info["t"],
        "lap_times_s": [round(b - a, 2) for a, b in zip([0.0] + lap_times[:-1], lap_times)],
        "mean_speed_mps": float(np.mean(speeds)),
        "max_speed_mps": float(np.max(speeds)),
        "mean_abs_offset_m": float(np.mean(offsets)),
        "max_abs_offset_m": float(np.max(offsets)),
        "allowed_offset_m": env.track.width / 2 - env.params.car_width / 2,
        "min_lidar_m": min_lidar,
        "track_length_m": env.track.length,
        "wall_time_s": time.time() - t0,
    }
    if verbose:
        print(f"트랙 길이 {res['track_length_m']:.1f} m, 목표 {laps}바퀴")
        print(f"완주 바퀴: {res['laps_completed']} / 이탈(충돌): {res['crashed']}")
        print(f"바퀴별 시간(s): {res['lap_times_s']}")
        print(f"평균 속도 {res['mean_speed_mps']:.2f} m/s, 최고 {res['max_speed_mps']:.2f} m/s")
        print(f"차선 오프셋 평균 {res['mean_abs_offset_m']:.3f} m, 최대 {res['max_abs_offset_m']:.3f} m"
              f" (허용 {res['allowed_offset_m']:.2f} m)")
        print(f"벽까지 최소 라이다 거리 {res['min_lidar_m']:.2f} m, 계산 시간 {res['wall_time_s']:.1f} s")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--v-max", type=float, default=15.0)
    ap.add_argument("--lidar-noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.laps, a.v_max, a.lidar_noise, a.seed)
