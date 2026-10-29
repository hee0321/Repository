"""새 머리 (YouTube Short 'VN40buqdA_s' 의 실제 마이크로덕 모양) — 시연용 모델 *사본* 에만 적용, 학습 XML 은 그대로.

예전 beak.py(튀어나온 주황 오리 부리)는 지움. 시뮬 모델 머리 메쉬가 원래 실제 마이크로덕 머리와 같은 모양
(둥근 덮개 top_head_shell · 앞면 face_part · 노란 눈(카메라) 고리 noenoeil · 아래 주황 판 jaw)이라,
  1) 색을 영상처럼 바꿈: 덮개·몸 판 = 하늘색, 앞면 = 흰색, 눈 고리 = 노랑, 아래 판·발 = 주황
  2) 아래 주황 판(jaw + 아랫입술 jaw_soft 의 *보이는* 메쉬)을 경첩 관절 `passive_jaw` 에 달아 벌렸다 닫음
     (덮개와 판 사이에 물건을 끼움). 위치 액추에이터 `jaw_motor` 를 시연 코드가 직접 씀.
관절 이름이 `passive_` 로 시작 → 정책의 관측·행동(14개 관절)에서 자동 제외 → v8 / Pollen ONNX 정책 입력·출력 그대로.
충돌용 메쉬(group 3)는 원래 자리에 그대로 둠 = 정책이 학습한 몸 물리는 안 바뀜. 움직이는 판은 보이기용(충돌 없음).
물건 집기는 물리 마찰 집기가 아니라 '붙이기'(duck_sim.grab) — 정직 표기 유지.

jaw_soft 바디 좌표: +x = 위, -z = 앞, y = 옆 (서 있을 때).
"""
import mujoco

JAW_JOINT = "passive_jaw"
JAW_ACT = "jaw_motor"
JAW_SITE = "jaw_mid"
JAW_OPEN = 0.75      # rad (앞쪽이 아래로 약 43°)
JAW_CLOSED = 0.0
JAW_HOLD = 0.42      # 물건을 덮개와 판 사이에 끼운 채 각도
PIVOT = (-0.010, 0.0, -0.012)   # 판 뒤쪽(목 쪽) 경첩
MOVING = ("jaw", "jaw_soft")     # 같이 움직이는 보이는 메쉬

HOOD = [0.56, 0.80, 0.88, 1]      # 하늘색 덮개
PANEL = [0.50, 0.76, 0.84, 1]     # 몸·다리 판
FACE = [0.94, 0.94, 0.92, 1]      # 흰 앞면
EYE = [0.99, 0.74, 0.08, 1]       # 노란 눈 고리
ORANGE = [1.0, 0.40, 0.12, 1]
COLORS = {
    "top_head_shell_material": HOOD, "face_part_material": FACE, "noenoeil_material": EYE,
    "jaw_material": ORANGE, "bottom_head_shell_material": ORANGE,
    "jaw_soft_material": ORANGE, "soft_mouth_top_material": ORANGE,
    "left_shell_material": PANEL, "right_shell_material": PANEL,
    "upper_leg_left_material": PANEL, "upper_leg_right_material": PANEL,
}


def add_head(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    for m in spec.materials:
        if m.name in COLORS:
            m.rgba = COLORS[m.name]
    head = spec.body("jaw_soft")
    plate = head.add_body(name="jaw_plate", pos=list(PIVOT))
    plate.add_joint(name=JAW_JOINT, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0],
                    range=[-0.05, 0.8], damping=0.02, armature=0.0005)
    moved = 0
    for g in list(head.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.meshname in MOVING and g.contype == 0 and g.conaffinity == 0:
            plate.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=g.meshname, material=g.material,
                           pos=[g.pos[i] - PIVOT[i] for i in range(3)], quat=list(g.quat),
                           contype=0, conaffinity=0, group=2, mass=0.0015)
            spec.delete(g)
            moved += 1
    assert moved == len(MOVING), f"moved {moved} jaw meshes (visual class changed?)"
    # 판 앞끝 가운데 (위 = +x 방향이 덮개 쪽). 숙이면 머리 앞끝이 가장 낮아서 여기서 물음
    plate.add_site(name=JAW_SITE, pos=[-0.003 - PIVOT[0], 0.0, -0.074 - PIVOT[2]], group=3)
    a = spec.add_actuator(name=JAW_ACT, target=JAW_JOINT, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    kp = 1.5
    a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    a.gainprm[0] = kp
    a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    a.biasprm[1] = -kp
    a.biasprm[2] = -0.01
    a.ctrlrange = [-0.05, 0.8]
    a.ctrllimited = True
    return spec


def with_head(spec_fn):
    def fn():
        return add_head(spec_fn())
    return fn
