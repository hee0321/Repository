"""Re-run the 18 s walk rollout (same plan as render_walk.py) on CPU and record the
actor network's per-layer activity + a synced video. Output: activity.json + walk.mp4"""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl")
from dataclasses import asdict
import numpy as np, torch, imageio
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
import mjlab_microduck.tasks  # noqa
import mjlab_microduck.tasks.mdp  # noqa

TASK = "Mjlab-VelStand-Rough-Backlash-MicroDuck"
ckpt, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)
env_cfg = load_env_cfg(TASK, play=True)
agent_cfg = load_rl_cfg(TASK)
env_cfg.scene.num_envs = 1
for k in ("push_robot", "random_prone_init", "topple_push"):
    env_cfg.events.pop(k, None)
for k in [k for k, c in env_cfg.curriculum.items() if any(e in repr(c) for e in ("push_robot", "random_prone_init", "topple_push", "terrain_levels"))]:
    env_cfg.curriculum.pop(k)
tg = env_cfg.scene.terrain.terrain_generator
tg.sub_terrains = {"flat": tg.sub_terrains["flat"]}
tg.sub_terrains["flat"].proportion = 1.0
env_cfg.viewer.width, env_cfg.viewer.height = 640, 360
env_cfg.viewer.distance = 0.9
env_cfg.viewer.elevation = -18.0
env_cfg.viewer.azimuth = 135.0
env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode="rgb_array")
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(venv, asdict(agent_cfg), device="cpu")
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cpu")
policy = runner.get_inference_policy(device="cpu")

sd = torch.load(ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
mean = sd["obs_normalizer._mean"][0].numpy().astype(np.float64)
std = sd["obs_normalizer._std"][0].numpy().astype(np.float64)
Ws = [sd[f"mlp.{i}.weight"].numpy().astype(np.float64) for i in (0, 2, 4, 6)]
bs = [sd[f"mlp.{i}.bias"].numpy().astype(np.float64) for i in (0, 2, 4, 6)]
elu = lambda x: np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1)

def forward(o):
    x = (o - mean) / (std + 1e-2)
    layers = [x]
    h = x
    for i in range(4):
        z = Ws[i] @ h + bs[i]
        h = elu(z) if i < 3 else z
        layers.append(h)
    return layers

om = env.observation_manager
terms = list(zip(om.active_terms["actor"], [d[0] for d in om.group_obs_term_dim["actor"]]))
robot = env.scene["robot"]
act_term = env.action_manager.get_term("joint_pos")
act_joints = None
for a in ("_joint_names", "joint_names", "target_names", "_target_names"):
    if hasattr(act_term, a):
        act_joints = list(getattr(act_term, a)); break
if act_joints is None:
    act_joints = [j for j in robot.joint_names if not j.startswith("passive_")]
print("terms", terms, "act joints", act_joints, flush=True)

dt = env.step_dt
term = env.command_manager.get_term("twist")
plan = [(2.0, 0.0, 0.0, 0.0), (5.0, 0.25, 0.0, 0.0), (4.0, 0.15, 0.0, 1.0), (4.0, 0.0, 0.0, -1.2), (3.0, 0.25, 0.0, 0.0)]
STRIDE = 2  # 50 Hz control -> 25 fps record
K = 28
frames, rec = [], {"t": [], "cmd": [], "L": [[], [], [], [], []], "edges": [[], [], [], []], "base": []}
maxerr = 0.0
step = 0
for secs, vx, vy, wz in plan:
    for _ in range(int(round(secs / dt))):
        term.vel_command_b[:] = torch.tensor([vx, vy, wz])
        for a in ("is_standing_env", "is_heading_env"):
            if hasattr(term, a): getattr(term, a)[:] = (vx == 0 and wz == 0) if a == "is_standing_env" else False
        obs = venv.get_observations()
        with torch.no_grad():
            act = policy(obs)
        if step % STRIDE == 0:
            o = (obs["actor"] if not torch.is_tensor(obs) else obs)[0].numpy().astype(np.float64)
            layers = forward(o)
            maxerr = max(maxerr, float(np.abs(layers[4] - act[0].numpy()).max()))
            rec["t"].append(round(step * dt, 3))
            rec["cmd"].append([vx, vy, wz])
            for li, h in enumerate(layers):
                rec["L"][li].append(np.clip(np.round(h * 100), -32000, 32000).astype(int).tolist())
            for li in range(4):
                c = Ws[li] * layers[li][None, :]  # contribution dst<-src
                flat = np.argsort(-np.abs(c), axis=None)[:K]
                d, s = np.unravel_index(flat, c.shape)
                rec["edges"][li].append([[int(a), int(b), int(np.sign(c[b, a]))] for a, b in zip(s, d)])
            r = robot.data
            p = r.root_link_pos_w[0].numpy(); q = r.root_link_quat_w[0].numpy()
            rec["base"].append([round(float(x), 4) for x in list(p) + list(q)] + [round(float(r.root_link_lin_vel_b[0, 0]), 3), round(float(r.root_link_ang_vel_b[0, 2]), 3)])
        obs2, _, dones, _ = venv.step(act)
        if step % STRIDE == 0:
            frames.append(env.render())
        step += 1
print("max |manual - policy| =", maxerr, "frames", len(frames), flush=True)
meta = {
    "ckpt": os.path.basename(ckpt), "task": TASK, "fps": 50 / STRIDE, "dt": dt * STRIDE,
    "obs_terms": terms, "act_joints": act_joints,
    "layer_sizes": [len(rec["L"][i][0]) for i in range(5)],
    "activation": "elu", "scale": 100, "topk_edges_per_layer": K,
    "plan": plan, "forward_check_max_abs_err": maxerr,
}
rec["meta"] = meta
json.dump(rec, open(os.path.join(outdir, "activity.json"), "w"), separators=(",", ":"))
imageio.mimwrite(os.path.join(outdir, "walk.mp4"), frames, fps=50 // STRIDE, codec="libx264", quality=6, macro_block_size=8,
                 output_params=["-movflags", "+faststart", "-pix_fmt", "yuv420p"])
print("done")
