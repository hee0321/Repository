"""Probe forward-speed response of a policy checkpoint (CPU, flat + backlash, no pushes/prone).
usage: vx_probe.py <ckpt.pt> [fed vx list comma-separated]"""
import sys, torch
from dataclasses import asdict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
import mjlab_microduck.tasks  # noqa
TASK = "Mjlab-VelStand-Flat-Backlash-MicroDuck"
ckpt = sys.argv[1]
feds = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0.1,0.2,0.3,0.4,0.5,0.6,0.8").split(",")]
REP = int(sys.argv[3]) if len(sys.argv) > 3 else 2
env_cfg = load_env_cfg(TASK, play=True); agent_cfg = load_rl_cfg(TASK)
n = len(feds) * REP
env_cfg.scene.num_envs = n
for k in ("push_robot", "random_prone_init", "topple_push"):
    env_cfg.events.pop(k, None)
for k in [k for k, c in env_cfg.curriculum.items() if any(e in repr(c) for e in ("push_robot", "random_prone_init", "topple_push"))]:
    env_cfg.curriculum.pop(k)
env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(venv, asdict(agent_cfg), device="cpu")
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cpu")
policy = runner.get_inference_policy(device="cpu")
term = env.command_manager.get_term("twist")
cmd = torch.zeros(n, 3); cmd[:, 0] = torch.tensor(feds).repeat_interleave(REP)
dt = env.step_dt; T = int(6.0 / dt); t0 = int(2.0 / dt)
acc = torch.zeros(n); cnt = torch.zeros(n); falls = torch.zeros(n)
for t in range(T):
    term.vel_command_b[:] = cmd
    if hasattr(term, "is_standing_env"): term.is_standing_env[:] = False
    if hasattr(term, "is_heading_env"): term.is_heading_env[:] = False
    obs = venv.get_observations()
    with torch.no_grad(): act = policy(obs)
    _, _, dones, _ = venv.step(act)
    r = env.scene["robot"].data
    up = r.projected_gravity_b[:, 2] < -0.8
    falls += (~up).float() * (t >= t0)
    if t >= t0:
        acc += r.root_link_lin_vel_b[:, 0] * up; cnt += up.float()
for i, f in enumerate(feds):
    s = slice(i * REP, (i + 1) * REP)
    print(f"fed vx={f:.2f} -> achieved {(acc[s].sum() / cnt[s].sum().clamp_min(1)).item():+.3f} m/s  tilted_frac={(falls[s].sum() / (REP * (T - t0))).item():.2f}", flush=True)
