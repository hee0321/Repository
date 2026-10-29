"""(선택) stable-baselines3 PPO로 CPU에서 짧게 강화학습. 가중치는 weights/ (git 무시)에 저장.

예) python train_ppo.py --steps 200000      # CPU 몇 분
    python train_ppo.py --eval weights/ppo_autodrive.zip
"""
import argparse
import os
import time

import numpy as np

from autodrive import AutoDriveEnv


def make_env(seed):
    def _f():
        return AutoDriveEnv(random_start=True, max_steps=1500, lidar_noise=0.05, seed=seed)
    return _f


def evaluate(model, episodes=3, laps=2):
    out = []
    for ep in range(episodes):
        env = AutoDriveEnv(target_laps=laps, max_steps=8000, seed=100 + ep)
        obs, info = env.reset(seed=100 + ep)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
            done = term or trunc
        out.append((info["laps"], info["crashed"], info["t"]))
        print(f"  평가 {ep}: {info['laps']:.2f}바퀴, 이탈={info['crashed']}, {info['t']:.1f}s")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--out", default="weights/ppo_autodrive.zip")
    ap.add_argument("--eval", default=None, help="학습 없이 이 가중치만 평가")
    a = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    if a.eval:
        evaluate(PPO.load(a.eval, device="cpu"))
        raise SystemExit
    venv = DummyVecEnv([make_env(i) for i in range(a.n_envs)])
    model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=256, learning_rate=3e-4, gamma=0.99,
                ent_coef=0.0, policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0, seed=0)
    t0 = time.time()
    model.learn(total_timesteps=a.steps, progress_bar=False)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    model.save(a.out)
    print(f"학습 {a.steps} 스텝, {time.time() - t0:.0f}s, 저장: {a.out}")
    evaluate(model)
