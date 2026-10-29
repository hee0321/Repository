"""Render the trained velstand policy walking forward + turning (CPU, offscreen)."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
from dataclasses import asdict
import numpy as np, torch, imageio
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
import mjlab_microduck.tasks  # noqa
import mjlab_microduck.tasks.mdp  # noqa

TASK = "Mjlab-VelStand-Rough-Backlash-MicroDuck"
ckpt, out = sys.argv[1], sys.argv[2]
env_cfg = load_env_cfg(TASK, play=True)
agent_cfg = load_rl_cfg(TASK)
env_cfg.scene.num_envs = 1
for k in ("push_robot", "random_prone_init", "topple_push"):
    env_cfg.events.pop(k, None)
for k in [k for k, c in env_cfg.curriculum.items() if any(e in repr(c) for e in ("push_robot", "random_prone_init", "topple_push", "terrain_levels"))]:
    print("drop curriculum", k); env_cfg.curriculum.pop(k)
tg = env_cfg.scene.terrain.terrain_generator
tg.sub_terrains = {"flat": tg.sub_terrains["flat"]}
tg.sub_terrains["flat"].proportion = 1.0
env_cfg.viewer.width, env_cfg.viewer.height = 960, 540
env_cfg.viewer.distance = 0.9
env_cfg.viewer.elevation = -18.0
env_cfg.viewer.azimuth = 135.0
env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode="rgb_array")
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(venv, asdict(agent_cfg), device="cpu")
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cpu")
policy = runner.get_inference_policy(device="cpu")
dt = env.step_dt
term = env.command_manager.get_term("twist")
# (seconds, vx, vy, wz)
plan = [(2.0, 0.0, 0.0, 0.0), (5.0, 0.25, 0.0, 0.0), (4.0, 0.15, 0.0, 1.0), (4.0, 0.0, 0.0, -1.2), (3.0, 0.25, 0.0, 0.0)]
fps = int(round(1 / dt)); frames = []
obs = venv.get_observations()
for secs, vx, vy, wz in plan:
    for _ in range(int(secs / dt)):
        term.vel_command_b[:] = torch.tensor([vx, vy, wz])
        for a in ("is_standing_env", "is_heading_env"):
            if hasattr(term, a): getattr(term, a)[:] = (vx == 0 and wz == 0) if a == "is_standing_env" else False
        obs = venv.get_observations()
        with torch.no_grad():
            act = policy(obs)
        obs, _, dones, _ = venv.step(act)
        frames.append(env.render())
        if len(frames) % 50 == 0:
            r = env.scene["robot"].data
            print(f"t={len(frames)*dt:.0f}s cmd=({vx},{vy},{wz}) z={r.root_link_pos_w[0,2].item():.3f} vx_b={r.root_link_lin_vel_b[0,0].item():.2f} wz={r.root_link_ang_vel_b[0,2].item():.2f} g_z={r.projected_gravity_b[0,2].item():.2f} done={bool(dones[0])}", flush=True)
print("dt", dt, "frames", len(frames), "fps", fps)
imageio.mimwrite(out, frames, fps=fps, codec="libx264", quality=7, macro_block_size=8)
