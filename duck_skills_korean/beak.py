"""오리 부리 (시연용 모델 사본에만 추가) — 학습용 XML 파일은 건드리지 않음.

원래 모델(robot_allcollisions_backlash.xml)은 머리 아래 주황 테두리(jaw 메시)만 있고 턱 관절이 없음.
여기서는 로드한 MjSpec *사본* 에 다음을 더함:
  - 윗부리: 머리 앞쪽에 붙은 납작한 주황·노랑 타원체 (jaw_soft 바디에 고정)
  - 아랫부리: 경첩(hinge) 관절 `passive_beak` + 위치 액추에이터 `beak_motor`
관절 이름이 `passive_` 로 시작하므로 정책의 관측(joint_pos/vel, 14개)과 행동(14개) 정규식에서 자동 제외됨
→ v8 / Pollen ONNX 정책 입력·출력 배열은 그대로. 부리는 시연 코드가 ctrl 을 직접 씀.
부리 geom 은 contype/conaffinity = 0 (몸·바닥과 충돌 없음 — 보이는 용도). 물건 집기는 beak_pick 참고.

jaw_soft 바디 좌표: +x = 위, -z = 앞, y = 옆 (서 있을 때).
"""
import mujoco

BEAK_JOINT = "passive_beak"
BEAK_ACT = "beak_motor"
BEAK_OPEN = 0.65    # rad (아랫부리 아래로 약 37°)
BEAK_CLOSED = 0.0
UPPER_RGBA = [1.0, 0.62, 0.08, 1]   # 주황
LOWER_RGBA = [1.0, 0.78, 0.15, 1]   # 노랑빛 주황
# 부리 크기 (반지름: 두께, 반폭, 반길이)
UP_SIZE = (0.0065, 0.026, 0.032)
LO_SIZE = (0.005, 0.022, 0.028)
PIVOT = (-0.013, 0.0, -0.066)        # 아랫부리 경첩 위치 (머리 앞 아래)
UP_POS = (-0.006, 0.0, -0.092)


def add_beak(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    head = spec.body("jaw_soft")
    # 원래 머리 아래 주황 테두리(jaw 메시)를 부리와 같은 색으로 맞춤
    for m in spec.materials:
        if m.name in ("jaw_material",):
            m.rgba = UPPER_RGBA
    g = head.add_geom(name="beak_upper", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=list(UP_SIZE),
                      pos=list(UP_POS), rgba=UPPER_RGBA, contype=0, conaffinity=0, group=2, mass=0.002)
    lo = head.add_body(name="beak_lower_body", pos=list(PIVOT))
    lo.add_joint(name=BEAK_JOINT, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0],
                 range=[-0.05, 0.8], damping=0.02, armature=0.0005)
    lo.add_geom(name="beak_lower", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=list(LO_SIZE),
                pos=[0.0, 0.0, -LO_SIZE[2] + 0.002], rgba=LOWER_RGBA, contype=0, conaffinity=0, group=2, mass=0.002)
    lo.add_site(name="beak_mid", pos=[0.004, 0.0, -0.022], group=3)
    a = spec.add_actuator(name=BEAK_ACT, target=BEAK_JOINT, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    kp = 1.5
    a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    a.gainprm[0] = kp
    a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    a.biasprm[1] = -kp
    a.biasprm[2] = -0.01
    a.ctrlrange = [-0.05, 0.8]
    a.ctrllimited = True
    return spec


def with_beak(spec_fn):
    def fn():
        return add_beak(spec_fn())
    return fn
