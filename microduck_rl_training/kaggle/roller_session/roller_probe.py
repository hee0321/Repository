"""롤러 정책 속도·직진·자세 측정 (학습 없음). 새로 학습한 ONNX 와 Pollen 공개 roller.onnx 를 같은 조건으로 비교.

  python roller_probe.py NEW.onnx [--scale 1.0] [--ref roller.onnx --ref-scale 0.8] [--device cpu|cuda:0]
측정: 시작 방향 0 고정, 무작위화 끔, 1초 서 있기 → cmd_x(0.2/0.4/0.6) 8초 → 마지막 5초 평균 속도, 방향 흔들림, 넘어짐.
주의: 이 저장소에서 내보낸 ONNX 는 변환 때 0.8 을 마지막 층에 접어 넣었으므로 --scale 1.0 (Pollen 원본은 0.8).
"""
import argparse, math, json
import numpy as np, torch, onnxruntime as ort
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import mjlab_microduck.tasks  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("onnx"); ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--ref"); ap.add_argument("--ref-scale", type=float, default=0.8)
ap.add_argument("--device", default="cpu"); ap.add_argument("--out")
a = ap.parse_args()

cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck-Rollers", play=True); cfg.scene.num_envs = 1
for k in [k for k in cfg.events if any(s in k for s in ("push", "prone", "topple", "randomize", "encoder_bias", "base_com"))]: cfg.events.pop(k)
for k in list(cfg.terminations):
    if k != "nan_state": cfg.terminations.pop(k)
cfg.episode_length_s = 1000.0
for k in list(cfg.curriculum): cfg.curriculum.pop(k)
for c in cfg.commands.values(): c.resampling_time_range = (1e9, 1e9); c.debug_vis = False
env = ManagerBasedRlEnv(cfg=cfg, device=a.device)
r = env.scene["robot"]; dt = env.step_dt


def run(sess, scale, cx, T=9.0):
    env.reset()
    st = torch.zeros(1, 13, device=env.device); st[0, :3] = r.data.root_link_pos_w[0]; st[0, 3] = 1.0
    r.write_root_state_to_sim(st)
    vx, yaw, gz_max = [], [], -1.0
    for k in range(int(T / dt)):
        o = env.observation_manager.compute()["actor"].cpu().numpy().astype(np.float32)
        o[:, 48:51] = (cx if k * dt > 1.0 else 0.0, 0.0, 0.0); o[:, 51:61] = 0.0
        act = sess.run(None, {sess.get_inputs()[0].name: o})[0]
        env.step(torch.from_numpy(act).to(env.device) * scale)
        q = r.data.root_link_quat_w[0].cpu().numpy()
        yaw.append(math.degrees(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))))
        vx.append(float(r.data.root_link_lin_vel_b[0, 0])); gz_max = max(gz_max, float(r.data.projected_gravity_b[0, 2]))
    n = int(4.0 / dt)
    return dict(cmd_x=cx, mean_vx_last5s=round(float(np.mean(vx[n:])), 3), max_abs_yaw_deg=round(float(np.max(np.abs(yaw[n:]))), 1),
                upright=bool(gz_max < -0.8), max_tilt_gz=round(gz_max, 2))


res = {}
for name, path, sc in (("new", a.onnx, a.scale), ("pollen_roller", a.ref, a.ref_scale)):
    if not path:
        continue
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    res[name] = [run(s, sc, cx) for cx in (0.2, 0.4, 0.6)]
    for x in res[name]: print(name, x, flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
