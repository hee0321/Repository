"""MuJoCo(mjlab) 시뮬레이션 래퍼: v8 velstand 정책 하나로 덕을 움직이고 공을 띄운다 (CPU, 학습 없음).

정책 입력 명령 = twist(vx, vy, wz) + head_pose(neck_pitch, head_pitch, head_yaw, head_roll)
               + body_pose(x, y, z, roll, pitch, yaw; 서 있을 때만 유효)
"""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import math
from dataclasses import replace, asdict, dataclass, field
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.robot.microduck_constants import MICRODUCK_BALL_CFG
from mjlab.entity import EntityCfg
import mujoco

TASK = "Mjlab-VelStand-Rough-Backlash-MicroDuck"
BALL_R = 0.035
HIDE = (30.0, 30.0)  # 공을 안 쓸 때 치워 두는 곳
OBST_GROUP = 1       # 장애물 geom 그룹 (로봇은 2·3, 가상 거리 센서 광선은 이 그룹만 봄)

# 장애물: (이름, 종류, 크기) — 덕(키 약 0.2 m, 발 들어 올림 2 cm 안팎)이 넘을 수 없는 높이
OBSTACLES = [
    ("obs_box1", "box", (0.08, 0.10, 0.08)),      # 반너비 x,y,z → 16×20×16 cm
    ("obs_cyl1", "cylinder", (0.07, 0.10)),       # 반지름, 반높이 → 지름 14, 높이 20 cm
    ("obs_box2", "box", (0.10, 0.07, 0.09)),
    ("obs_cyl2", "cylinder", (0.06, 0.09)),
]
OBST_COLORS = [(0.25, 0.55, 0.95, 1), (0.35, 0.75, 0.40, 1), (0.85, 0.35, 0.55, 1), (0.95, 0.75, 0.20, 1)]


PICK_HALF = 0.016   # 집을 물건: 지름 3.2 cm 말랑 공 (영상의 양말처럼 덮개와 턱판 사이에 끼움; 보기용 운동학적 물체, 질량 없음)
PICK_HALF_Z = PICK_HALF


def _pickup_spec():
    s = mujoco.MjSpec()
    b = s.worldbody.add_body(name="root", mocap=True)
    g = b.add_geom(name="geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[PICK_HALF, 0, 0],
                   rgba=[0.96, 0.45, 0.65, 1], group=0, contype=0, conaffinity=0)
    g.pos = [0, 0, 0]
    return s


def _obstacle_spec(kind, size, rgba):
    def fn():
        s = mujoco.MjSpec()
        b = s.worldbody.add_body(name="root", mocap=True)
        g = b.add_geom(name="geom", type=mujoco.mjtGeom.mjGEOM_BOX if kind == "box" else mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=list(size) + [0] * (3 - len(size)), rgba=list(rgba), group=OBST_GROUP)
        g.pos = [0, 0, size[-1] + 0.003]  # 바닥과 3 mm 띄워 바닥-장애물 접촉 계산을 없앰
        return s
    return fn


@dataclass
class Cmd:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    head: tuple = (0.0, 0.0, 0.0, 0.0)            # neck_pitch, head_pitch, head_yaw, head_roll (rad, HOME 기준)
    body: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # x, y, z, roll, pitch, yaw (서 있을 때만)
    policy: str | None = None   # None = v8 velstand, 아니면 Pollen 공개 ONNX 이름 (policies.py)
    beak: float | None = None   # 아래 턱판(passive_jaw 경첩) 목표 각도 rad, None = 그대로 유지

    @property
    def standing(self) -> bool:
        return abs(self.vx) < 1e-6 and abs(self.vy) < 1e-6 and abs(self.wz) < 1e-6


class DuckSim:
    def __init__(self, ckpt: str, width=960, height=544, distance=0.9, with_ball=True, with_obstacles=True, with_beak=True):
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
        if with_obstacles:
            for (name, kind, size), rgba in zip(OBSTACLES, OBST_COLORS):
                cfg.scene.entities = {**cfg.scene.entities, name: EntityCfg(spec_fn=_obstacle_spec(kind, size, rgba))}
            cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 1000); cfg.sim.njmax = max(cfg.sim.njmax or 0, 4000)
        if with_beak:
            # 보상 항은 시연에 필요 없고, 일부가 '모든 액추에이터=14개'를 가정해서 턱판(15번째)와 충돌 → 끔
            for k in list(cfg.rewards):
                cfg.rewards.pop(k)
            for k in list(cfg.curriculum):   # 보상 가중치 커리큘럼도 같이 끔 (시연엔 영향 없음)
                cfg.curriculum.pop(k)
        if with_beak:   # 시연용 모델 사본에 새 머리(색 + 아래 판 경첩·액추에이터) — 학습 XML 은 그대로 (head.py)
            from head import with_head as _wb
            rc = cfg.scene.entities["robot"]
            cfg.scene.entities = {**cfg.scene.entities, "robot": replace(rc, spec_fn=_wb(rc.spec_fn)),
                                  "pickup": EntityCfg(spec_fn=_pickup_spec)}
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
        self.obstacles = [self.env.scene[n] for n, _, _ in OBSTACLES] if with_obstacles else []
        self.obst_xy = {}
        for i, n in enumerate(self.obstacles):
            self.place_obstacle(i, (HIDE[0] + 2 * i, -HIDE[1]))
        # 가상 거리 센서·접촉 검사용 CPU 사본 (정책/물리는 그대로 warp 에서 돎)
        self.mjm = self.env.sim.mj_model
        self.mjd = mujoco.MjData(self.mjm)
        self.obst_geoms = set(int(g) for g in range(self.mjm.ngeom)
                              if self.mjm.body(int(self.mjm.geom_bodyid[g])).name.startswith("obs_"))
        rb = set(int(b) for b in range(self.mjm.nbody) if self.mjm.body(b).name.startswith("robot/"))
        self.robot_geoms = set(int(g) for g in range(self.mjm.ngeom) if int(self.mjm.geom_bodyid[g]) in rb)
        self.beak_aid = None
        self.pickup = None
        self.carry = None       # (턱판 좌표계 오프셋) — 물고 있으면 물건을 턱판에 붙여 같이 움직임
        self.beak_target = 0.0
        if with_beak:
            self.pickup = self.env.scene["pickup"]
            self.beak_aid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot/jaw_motor")
            self.beak_jid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_JOINT, "robot/passive_jaw")
            self.beak_site = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_SITE, "robot/jaw_mid")
            self.place_pickup(HIDE[0], HIDE[1] + 3)
        self.robot_coll_geoms = [g for g in self.robot_geoms if self.mjm.geom_contype[g] or self.mjm.geom_conaffinity[g]]

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

    # ── 장애물 ─────────────────────────────────────────────────────────────
    def place_obstacle(self, i, xy):
        pose = torch.tensor([[float(xy[0]), float(xy[1]), 0.0, 1.0, 0.0, 0.0, 0.0]])
        self.obstacles[i].write_mocap_pose_to_sim(pose)
        self.obst_xy[i] = np.array(xy, dtype=float)

    def place_course(self, layout):
        """layout: 덕 기준 (앞 x, 왼쪽 y) 목록. 덕의 현재 위치·방향을 기준으로 장애물을 놓는다."""
        p, yaw, _ = self.pose()
        self.course_heading = yaw
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        for i in range(len(self.obstacles)):
            xy = p[:2] + R @ np.array(layout[i]) if i < len(layout) else np.array((HIDE[0] + 2 * i, -HIDE[1]))
            self.place_obstacle(i, xy)

    def sync_cpu(self):
        d = self.env.sim.data
        self.mjd.qpos[:] = d.qpos[0].cpu().numpy()
        self.mjd.qvel[:] = d.qvel[0].cpu().numpy()
        if self.mjm.nmocap:
            self.mjd.mocap_pos[:] = d.mocap_pos[0].cpu().numpy()
            self.mjd.mocap_quat[:] = d.mocap_quat[0].cpu().numpy()
        mujoco.mj_forward(self.mjm, self.mjd)

    def range_scan(self, angles_deg=range(-75, 76, 15), height=0.09, max_range=1.0):
        """가상 거리 센서: 몸통에서 수평으로 광선을 쏴서(MuJoCo mj_ray) 장애물까지 거리를 잰다.
        반환: [(덕 기준 각도 rad, 거리 m)], 안 맞으면 max_range."""
        self.sync_cpu()
        p, yaw, _ = self.pose()
        mask = np.zeros(6, np.uint8); mask[OBST_GROUP] = 1
        gid = np.zeros(1, np.int32)
        out = []
        for a in angles_deg:
            th = yaw + math.radians(a)
            v = np.array([math.cos(th), math.sin(th), 0.0])
            org = np.array([p[0], p[1], height])
            dist = mujoco.mj_ray(self.mjm, self.mjd, org, v, mask, 1, -1, gid)
            out.append((math.radians(a), dist if 0 <= dist < max_range else max_range))
        return out

    def obstacle_contacts_and_clearance(self, distmax=0.5):
        """(로봇-장애물 접촉 수, 로봇 충돌 geom 과 장애물 사이 최소 거리 m). sync_cpu 후 호출."""
        n = 0
        for c in self.mjd.contact[: self.mjd.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in self.obst_geoms and g2 in self.robot_geoms) or (g2 in self.obst_geoms and g1 in self.robot_geoms):
                n += 1
        # 최소 거리: 로봇 충돌 메쉬 꼭짓점들과 장애물(상자·원기둥) 사이의 기하학적 거리 (numpy)
        # (mj_geomDistance 는 메쉬-상자 쌍에서 가끔 0 을 돌려줘서 쓰지 않음)
        m, dd_ = self.mjm, self.mjd
        if not hasattr(self, "_rverts"):
            self._rverts = {}
            for g in self.robot_coll_geoms:
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = m.geom_dataid[g]
                    v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                    self._rverts[g] = v[:: max(1, len(v) // 300)].copy()
                else:
                    r = m.geom_size[g]
                    self._rverts[g] = np.array([[sx * r[0], sy * r[1], sz * r[2]] for sx in (-1, 0, 1) for sy in (-1, 0, 1) for sz in (-1, 0, 1)])
        pts = np.concatenate([dd_.geom_xpos[g] + self._rverts[g] @ dd_.geom_xmat[g].reshape(3, 3).T for g in self.robot_coll_geoms])
        best = distmax
        for og in self.obst_geoms:
            c = dd_.geom_xpos[og]; R = dd_.geom_xmat[og].reshape(3, 3); sz = m.geom_size[og]
            q = (pts - c) @ R  # 장애물 좌표계
            if m.geom_type[og] == mujoco.mjtGeom.mjGEOM_BOX:
                out = np.maximum(np.abs(q) - sz[:3], 0.0)
                dist = np.linalg.norm(out, axis=1)
            else:  # cylinder: 반지름 sz[0], 반높이 sz[1]
                dr = np.maximum(np.linalg.norm(q[:, :2], axis=1) - sz[0], 0.0)
                dz = np.maximum(np.abs(q[:, 2]) - sz[1], 0.0)
                dist = np.hypot(dr, dz)
            best = min(best, float(dist.min()))
        return n, best

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

    # ── 턱판·집을 물건 ─────────────────────────────────────────────────────
    def place_pickup(self, x, y, z=PICK_HALF_Z, quat=(1.0, 0.0, 0.0, 0.0)):
        self.pickup_pos = np.array([x, y, z], dtype=float)
        self.pickup.write_mocap_pose_to_sim(torch.tensor([[float(x), float(y), float(z), *map(float, quat)]]))

    def beak_mid(self):
        """(턱판 앞 가운데 위치, 턱판 회전행렬) — warp 데이터에서 직접."""
        d = self.env.sim.data
        return d.site_xpos[0, self.beak_site].numpy().astype(float), d.site_xmat[0, self.beak_site].numpy().reshape(3, 3).astype(float)

    def beak_angle(self):
        qa = self.mjm.jnt_qposadr[self.beak_jid]
        return float(self.env.sim.data.qpos[0, qa])

    GRAB_OFFSET = (0.009, 0.0, -0.006)   # 턱판 좌표: 판 앞쪽 가운데 위(+x, 덮개 쪽)에 물건 가운데

    def grab(self):
        """운동학적 '붙이기' (물리 집기 아님): 물건을 턱판 위 정해진 자리에 붙여 같이 움직이게 함.
        붙이는 순간 물건이 옮겨 가는 거리(snap)를 기록."""
        p, R = self.beak_mid()
        self.carry = np.array(self.GRAB_OFFSET)
        self.grab_snap = float(np.linalg.norm(p + R @ self.carry - self.pickup_pos))
        if os.environ.get("DUCK_GRAB_DEBUG"):
            bp, yaw, _ = self.pose(); tgt = p + R @ self.carry
            c_, s_ = math.cos(yaw), math.sin(yaw); dx, dy = tgt[0] - bp[0], tgt[1] - bp[1]
            e = tgt - self.pickup_pos
            print("GRAB_DEBUG target-minus-object (fwd, left, z)", round(c_ * e[0] + s_ * e[1], 4), round(-s_ * e[0] + c_ * e[1], 4), round(float(e[2]), 4))

    def release(self):
        self.carry = None

    def _update_carry(self):
        if self.carry is None:
            return
        p, R = self.beak_mid()
        q = np.zeros(4); mujoco.mju_mat2Quat(q, R.reshape(-1))
        pos = p + R @ self.carry
        self.pickup_pos = pos
        self.pickup.write_mocap_pose_to_sim(torch.tensor([[*map(float, pos), *map(float, q)]]))

    # ── 한 스텝 ────────────────────────────────────────────────────────────
    def step(self, c: Cmd):
        tw = self.twist
        if c.policy:   # ONNX 정책은 명령을 관측에 직접 넣음; env 명령은 '서 있음'으로
            c_env = Cmd()
        else:
            c_env = c
        tw.vel_command_b[:] = torch.tensor([c_env.vx, c_env.vy, c_env.wz])
        if hasattr(tw, "is_standing_env"):
            tw.is_standing_env[:] = bool(c_env.standing)
        if hasattr(tw, "is_heading_env"):
            tw.is_heading_env[:] = False
        self.head._command[:] = torch.tensor(c.head, dtype=torch.float32)
        b = torch.tensor(c.body, dtype=torch.float32) * (1.0 if c.standing else 0.0)
        if hasattr(self.body, "_raw"):
            self.body._raw[:] = torch.tensor(c.body, dtype=torch.float32)
        self.body._command[:] = b
        if self.ball is not None:
            if getattr(self, "ball_free", False):   # 공 차기: 공을 물리에 맡김 (위치만 읽어 옴)
                self.ball_xy = self.ball.data.root_link_pos_w[0, :2].numpy().astype(float)
            else:
                self.set_ball(self.ball_xy + self.ball_v * self.dt, self.ball_v)
        if self.beak_aid is not None:
            if c.beak is not None:
                self.beak_target = float(c.beak)
            self.env.sim.data.ctrl[0, self.beak_aid] = self.beak_target
        obs = self.venv.get_observations()
        if c.policy:
            act = self.onnx_act(c, obs["actor"])
        else:
            with torch.no_grad():
                act = self.policy(obs)
        self.venv.step(act)
        if self.beak_aid is not None:
            self._update_carry()
            if os.environ.get("DUCK_GRAB_DEBUG") and int(round(self.t / self.dt)) % 5 == 0:
                p_, R_ = self.beak_mid(); bp, yaw, _ = self.pose(); tgt = p_ + R_ @ np.array(self.GRAB_OFFSET)
                c_, s_ = math.cos(yaw), math.sin(yaw); dx, dy = tgt[0] - bp[0], tgt[1] - bp[1]
                print(f"JAWZ t={self.t:.2f} fwd={c_*dx+s_*dy:.3f} left={-s_*dx+c_*dy:.3f} z={tgt[2]:.3f} ang={self.beak_angle():.2f}")
        self.t += self.dt

    # ── Pollen 공개 ONNX 정책 실행 (같은 61차원 관측·14차원 행동 약속) ──────────
    def onnx_act(self, c: "Cmd", actor_obs: torch.Tensor) -> torch.Tensor:
        from policies import session, USES
        sess = session(c.policy)
        o = actor_obs.clone().numpy().astype(np.float32)
        uses_head, uses_body = USES[c.policy]
        o[:, 48:51] = (c.vx, c.vy, c.wz)                      # twist 자리 = 정책별 명령(자세 플래그, 위상 등)
        o[:, 51:55] = c.head if uses_head else (0.0,) * 4      # 학습 때 0으로 채운 자리는 0
        o[:, 55:61] = c.body if uses_body else (0.0,) * 6
        a = sess.run(None, {"obs": o})[0]
        return torch.from_numpy(a)

    def render(self) -> np.ndarray:
        return self.env.render()
