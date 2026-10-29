"""MuJoCo(mjlab) 시뮬레이션 래퍼: v8 velstand 정책 하나로 덕을 움직이고 공을 띄운다 (CPU, 학습 없음).

정책 입력 명령 = twist(vx, vy, wz) + head_pose(neck_pitch, head_pitch, head_yaw, head_roll)
               + body_pose(x, y, z, roll, pitch, yaw; 서 있을 때만 유효)
"""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import math
from dataclasses import asdict, dataclass, field
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.robot.microduck_constants import MICRODUCK_BALL_CFG

TASK = "Mjlab-VelStand-Rough-Backlash-MicroDuck"
BALL_R = 0.035
HIDE = (30.0, 30.0)  # 공을 안 쓸 때 치워 두는 곳


@dataclass
class Cmd:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    head: tuple = (0.0, 0.0, 0.0, 0.0)            # neck_pitch, head_pitch, head_yaw, head_roll (rad, HOME 기준)
    body: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # x, y, z, roll, pitch, yaw (서 있을 때만)

    @property
    def standing(self) -> bool:
        return abs(self.vx) < 1e-6 and abs(self.vy) < 1e-6 and abs(self.wz) < 1e-6


class DuckSim:
    def __init__(self, ckpt: str, width=960, height=544, distance=0.9, with_ball=True):
        cfg = load_env_cfg(TASK, play=True)
        agent = load_rl_cfg(TASK)
        cfg.scene.num_envs = 1
        # 시연용: 무작위 밀기/넘어진 채 시작/지형 난이도 끄기, 시간 제한 없애기
        for k in ("push_robot", "random_prone_init", "topple_push"):
            cfg.events.pop(k, None)
        for k in [k for k, c in cfg.curriculum.items()
                  if any(e in repr(c) for e in ("push_robot", "random_prone_init", "topple_push", "terrain_levels"))]:
            cfg.curriculum.pop(k)
        tg = cfg.scene.terrain.terrain_generator
        tg.sub_terrains = {"flat": tg.sub_terrains["flat"]}
        tg.sub_terrains["flat"].proportion = 1.0
        for k in list(cfg.terminations):
            if k != "nan_state":
                cfg.terminations.pop(k)
        cfg.episode_length_s = 10_000.0
        for c in cfg.commands.values():
            c.resampling_time_range = (1e9, 1e9)
            c.debug_vis = False  # 명령 화살표(파란 표시) 숨기기
        if with_ball:
            cfg.scene.entities = {**cfg.scene.entities, "ball": MICRODUCK_BALL_CFG}
            cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 50)
        cfg.viewer.width, cfg.viewer.height = width, height
        cfg.viewer.distance = distance
        cfg.viewer.elevation = -18.0
        cfg.viewer.azimuth = 135.0
        self.env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode="rgb_array")
        self.venv = RslRlVecEnvWrapper(self.env, clip_actions=agent.clip_actions)
        runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(self.venv, asdict(agent), device="cpu")
        runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cpu")
        self.policy = runner.get_inference_policy(device="cpu")
        self.dt = self.env.step_dt
        self.robot = self.env.scene["robot"]
        self.ball = self.env.scene["ball"] if with_ball else None
        self.twist = self.env.command_manager.get_term("twist")
        self.head = self.env.command_manager.get_term("head_pose")
        self.body = self.env.command_manager.get_term("body_pose")
        self.t = 0.0
        self.ball_xy = np.array(HIDE)
        self.ball_v = np.zeros(2)
        self.venv.reset() if hasattr(self.venv, "reset") else None
        self.set_ball(HIDE, (0, 0))

    # ── 상태 ────────────────────────────────────────────────────────────────
    def pose(self):
        d = self.robot.data
        p = d.root_link_pos_w[0].numpy()
        q = d.root_link_quat_w[0].numpy()  # w x y z
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        g_z = d.projected_gravity_b[0, 2].item()
        return p, yaw, g_z

    def roll_pitch(self):
        g = self.robot.data.projected_gravity_b[0].numpy()
        return math.atan2(-g[1], -g[2]), math.atan2(g[0], -g[2])

    def upright(self) -> bool:
        return self.pose()[2] < -0.8

    # ── 공 (스크립트로 굴리는 운동학적 공) ─────────────────────────────────
    def set_ball(self, xy, v=(0.0, 0.0)):
        if self.ball is None:
            return
        self.ball_xy = np.array(xy, dtype=float)
        self.ball_v = np.array(v, dtype=float)
        st = torch.zeros(1, 13)
        st[0, 0:2] = torch.tensor(self.ball_xy, dtype=torch.float32)
        st[0, 2] = BALL_R
        # 굴러가는 모습: 각속도 = z × v / r
        ang = np.array([-self.ball_v[1], self.ball_v[0], 0.0]) / BALL_R
        if not hasattr(self, "_ball_q"):
            self._ball_q = np.array([1.0, 0, 0, 0])
        w = np.linalg.norm(ang)
        if w > 1e-9:
            a = ang / w * w * self.dt
            th = np.linalg.norm(a)
            dq = np.concatenate([[math.cos(th / 2)], a / th * math.sin(th / 2)])
            q0 = self._ball_q
            self._ball_q = np.array([
                dq[0]*q0[0]-dq[1]*q0[1]-dq[2]*q0[2]-dq[3]*q0[3],
                dq[0]*q0[1]+dq[1]*q0[0]+dq[2]*q0[3]-dq[3]*q0[2],
                dq[0]*q0[2]-dq[1]*q0[3]+dq[2]*q0[0]+dq[3]*q0[1],
                dq[0]*q0[3]+dq[1]*q0[2]-dq[2]*q0[1]+dq[3]*q0[0]])
            self._ball_q /= np.linalg.norm(self._ball_q)
        st[0, 3:7] = torch.tensor(self._ball_q, dtype=torch.float32)
        st[0, 7:9] = torch.tensor(self.ball_v, dtype=torch.float32)
        st[0, 10:13] = torch.tensor(ang, dtype=torch.float32)
        self.ball.write_root_state_to_sim(st)

    # ── 넘어뜨리기 (일어나기 시연용) ───────────────────────────────────────
    def knock_over(self, face="down"):
        p, yaw, _ = self.pose()
        pitch = math.pi / 2 if face == "down" else -math.pi / 2
        # q = yaw(z) * pitch(y)
        cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
        q = [cy * cp, -sy * sp, cy * sp, sy * cp]
        st = torch.zeros(1, 13)
        st[0, 0], st[0, 1], st[0, 2] = float(p[0]), float(p[1]), 0.07
        st[0, 3:7] = torch.tensor(q)
        self.robot.write_root_state_to_sim(st)
        dj = self.robot.data.default_joint_pos.clone()
        self.robot.write_joint_state_to_sim(dj, torch.zeros_like(dj))

    # ── 한 스텝 ────────────────────────────────────────────────────────────
    def step(self, c: Cmd):
        tw = self.twist
        tw.vel_command_b[:] = torch.tensor([c.vx, c.vy, c.wz])
        if hasattr(tw, "is_standing_env"):
            tw.is_standing_env[:] = c.standing
        if hasattr(tw, "is_heading_env"):
            tw.is_heading_env[:] = False
        self.head._command[:] = torch.tensor(c.head, dtype=torch.float32)
        b = torch.tensor(c.body, dtype=torch.float32) * (1.0 if c.standing else 0.0)
        if hasattr(self.body, "_raw"):
            self.body._raw[:] = torch.tensor(c.body, dtype=torch.float32)
        self.body._command[:] = b
        if self.ball is not None:
            self.set_ball(self.ball_xy + self.ball_v * self.dt, self.ball_v)
        obs = self.venv.get_observations()
        with torch.no_grad():
            act = self.policy(obs)
        self.venv.step(act)
        self.t += self.dt

    def render(self) -> np.ndarray:
        return self.env.render()
