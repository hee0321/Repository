"""롤러스케이트 덕 시뮬 (CPU, 학습 없음): Pollen 공개 roller.onnx(타기) + roller_crouch.onnx(웅크려 미끄러지기).

모델 = robot_groundcontact_rollers.xml (scene_rollers 계열, 발 밑 수동 바퀴 4개, 작업 Mjlab-Velocity-Flat-MicroDuck-Rollers).
명령(관측 48:51, twist 자리) — Pollen roller 약속 그대로:
  cmd_x : 0 = 미끄러지기(coast), + = 밀어서 가속, - = 브레이크 (학습 범위 -0.5 … 0.6)
  cmd[2]: 방향 오차(rad). 공개 roller.onnx 학습 설정에서는 0 으로 고정(직진 위주)이라, 여기선 방향 보정에 시험적으로 씀
roller_crouch: twist 자리 = [cos 2πφ, sin 2πφ, 0], 주기 5초, 3.5초에 끝 (manifest)
manifest: 롤러 정책은 action_scale 0.8 (환경 설정은 1.0) → 행동 × 0.8.
새 머리(head.py)도 씌움 — 보이는 메쉬 색·턱판 경첩만, 충돌·질량 거의 그대로.
"""
from __future__ import annotations
import os, math
os.environ.setdefault("MUJOCO_GL", "egl")
from dataclasses import replace
import numpy as np, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import mjlab_microduck.tasks  # noqa: F401

TASK = "Mjlab-Velocity-Flat-MicroDuck-Rollers"
ROLLER_SCALE = 0.8
CROUCH_PERIOD, CROUCH_LEN = 5.0, 3.5


class RollerSim:
    def __init__(self, width=960, height=544, with_head=True, render=True, yaw0=0.0):
        cfg = load_env_cfg(TASK, play=True)
        cfg.scene.num_envs = 1
        for k in [k for k in cfg.events if any(s in k for s in ("push", "prone", "topple", "randomize", "encoder_bias", "base_com"))]:
            cfg.events.pop(k)   # 시연: 무작위화 끔 (기본 모델 그대로)
        for k in list(cfg.terminations):
            if k != "nan_state":
                cfg.terminations.pop(k)
        cfg.episode_length_s = 10_000.0
        for k in list(cfg.curriculum):
            cfg.curriculum.pop(k)
        for c in cfg.commands.values():
            c.resampling_time_range = (1e9, 1e9); c.debug_vis = False
        if with_head:
            for k in list(cfg.rewards):   # 보상은 시연에 불필요 + 일부가 액추에이터 14개를 가정
                cfg.rewards.pop(k)
            from head import with_head as _wh
            rc = cfg.scene.entities["robot"]
            cfg.scene.entities = {**cfg.scene.entities, "robot": replace(rc, spec_fn=_wh(rc.spec_fn))}
        cfg.viewer.width, cfg.viewer.height = width, height
        cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 0.9, -18.0, 135.0
        self.env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode="rgb_array" if render else None)
        self.env.reset()
        self.robot = self.env.scene["robot"]
        self.dt = self.env.step_dt
        jp = self.env.action_manager.get_term("joint_pos")
        self.env_scale = float(np.mean(np.asarray(jp._scale).reshape(-1))) if hasattr(jp, "_scale") else 1.0
        self.t = 0.0
        self.set_yaw(yaw0)

    def set_yaw(self, yaw):
        """시작 방향 고정 (학습 땐 무작위) — 영상 카메라·직진 측정용."""
        st = torch.zeros(1, 13)
        st[0, 0:3] = self.robot.data.root_link_pos_w[0]
        st[0, 3:7] = torch.tensor([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        self.robot.write_root_state_to_sim(st)
        self.yaw_ref = yaw

    def pose(self):
        d = self.robot.data
        q = d.root_link_quat_w[0].numpy()
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        return d.root_link_pos_w[0].numpy().copy(), yaw, d.projected_gravity_b[0, 2].item()

    def vel(self):
        d = self.robot.data
        return d.root_link_lin_vel_b[0].numpy().copy(), d.root_link_ang_vel_b[0].numpy().copy()

    def step(self, policy: str, cmd):
        from policies import session
        o = self.env.observation_manager.compute()["actor"].clone().numpy().astype(np.float32)
        o[:, 48:51] = cmd
        o[:, 51:61] = 0.0   # 롤러 정책: 고개·몸 자리는 0 (학습 때 zero-padding)
        a = session(policy).run(None, {"obs": o})[0]
        self.env.step(torch.from_numpy(a) * ROLLER_SCALE / self.env_scale)
        self.t += self.dt

    def render(self):
        return self.env.render()
