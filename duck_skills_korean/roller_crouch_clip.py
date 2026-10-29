"""'웅크려' — Pollen roller_crouch.onnx 를 원래 학습 환경(롤러스케이트 신은 덕, Mjlab-RollerCrouch-Flat)에서 돌린 영상.

roller_crouch 는 바퀴 달린 발(수동 바퀴 4개)을 전제로 학습돼서, 걷는 덕 모델에서는 넘어집니다(시험해 봄).
그래서 이 클립만 롤러 덕 모델로 따로 찍어요: roller.onnx(롤러 위 서 있기) → 웅크려(roller_crouch, 3.5초) → roller.onnx.
명령 약속은 manifest 그대로: 위상 [cos 2πφ, sin 2πφ], 주기 5초, 위상 0.7 에서 끝.
"""
from __future__ import annotations
import os, sys, math
os.environ.setdefault("MUJOCO_GL", "egl")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch, imageio.v2 as imageio
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import mjlab_microduck.tasks  # noqa
from policies import session
from run_demo import overlay
import behaviors

TASK = "Mjlab-RollerCrouch-Flat-MicroDuck"
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "videos", "clip_roller_crouch.mp4")
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = 1
for k in [k for k in cfg.events if any(s in k for s in ("push", "prone", "topple", "randomize", "encoder_bias", "base_com"))]:  # 무작위화 끔 (기본 모델 그대로)
    cfg.events.pop(k)
for k in list(cfg.terminations):
    if k != "nan_state":
        cfg.terminations.pop(k)
cfg.episode_length_s = 1000.0
for k in list(cfg.curriculum):   # 커리큘럼 끔 (무작위화를 껐으니)
    cfg.curriculum.pop(k)
for c in cfg.commands.values():
    c.resampling_time_range = (1e9, 1e9); c.debug_vis = False
cfg.viewer.width, cfg.viewer.height = 960, 544
cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 0.5, -12.0, 120.0
env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode="rgb_array")
env.reset()
dt = env.step_dt
ROLLER_SCALE = 0.8
_jp = env.action_manager.get_term("joint_pos") if "joint_pos" in env.action_manager.active_terms else None
ENV_SCALE = float(np.mean(np.asarray(_jp._scale).reshape(-1))) if _jp is not None and hasattr(_jp, "_scale") else 1.0
print("ENV action scale", ENV_SCALE, list(cfg.events))
robot = env.scene["robot"]
HP = list(robot.joint_names).index("head_pitch")


def act(name, cmd, head=(0, 0, 0, 0)):
    o = env.observation_manager.compute()["actor"].clone().numpy().astype(np.float32)
    o[:, 48:51] = cmd
    o[:, 51:55] = 0.0; o[:, 55:61] = 0.0
    a = session(name).run(None, {"obs": o})[0]
    # manifest: 롤러 정책은 action_scale 0.8 (환경 설정은 1.0) → 행동에 0.8 곱해서 넣음
    env.step(torch.from_numpy(a) * ROLLER_SCALE / ENV_SCALE)


# 대본: (시작초, 끝초, 정책, 설명)
plan = [(0.0, 1.5, "roller", None), (1.5, 2.4, "roller", "ack"), (2.4, 5.9, "roller_crouch", "crouch"), (5.9, 9.0, "roller", "back")]
w = imageio.get_writer(out, fps=25, codec="libx264", quality=None, macro_block_size=8,
                       output_params=["-crf", "27", "-pix_fmt", "yuv420p"])
zs, gz_max = [], -1.0
n = int(9.0 / dt)
for k in range(n):
    t = k * dt
    seg = [p for p in plan if p[0] <= t < p[1]][0]
    name, what = seg[2], seg[3]
    if name == "roller_crouch":
        ph = (t - seg[0]) / 5.0
        act(name, (math.cos(2 * math.pi * ph), math.sin(2 * math.pi * ph), 0.0))
    else:
        act(name, (0.0, 0.0, 0.0))
    z = robot.data.root_link_pos_w[0, 2].item(); g = robot.data.projected_gravity_b[0, 2].item()
    zs.append(z); gz_max = max(gz_max, g)
    if k % 2 == 0:
        extra = {"ack": "알아들었어요 (롤러 정책은 고개 명령을 안 받아 신호 생략)", "crouch": "정책 바꿈 → roller_crouch.onnx",
                 "back": "끝 → roller.onnx 로 복귀"}.get(what)
        said = "웅크려" if t >= 1.5 else ""
        status = f"{t:4.1f}s · 몸통 높이 {z:.3f} m · {'서 있음' if g < -0.8 else '기울어짐'}"
        fr = overlay(env.render(), said, "웅크리기 (Pollen roller_crouch.onnx, 롤러스케이트 덕 모델)", status, extra, tag="MuJoCo · 롤러 덕 모델 · Pollen 공개 정책 (추가 학습 없음)")
        w.append_data(fr)
w.close()
print(f"ROLLER z start {zs[0]:.3f} min {min(zs):.3f} end {zs[-1]:.3f} max_gz {gz_max:+.2f}")
