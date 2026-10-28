# Microduck-inspired biped — design brief (free prep)

## Goal
Cute ~25 cm biped for sim RL (MuJoCo), compatible with microduck_rl-style 14-servo layout so we can reuse Velocity-Flat training recipes later.

## Form
- Soft yellow/cream ABS shell, orange beak & feet, rounded duck silhouette
- Visible XL330-class servos at joints (honest mechanism, not fully covered)
- Foot pads for ground contact; battery bay in rear trunk; status LEDs on chest

## Actuation (14 DoF)
| # | Joint | Notes |
|---|--------|--------|
| 0–4 | L hip_yaw, hip_roll, hip_pitch, knee, ankle | walking chain |
| 5–8 | neck_pitch, head_pitch, head_yaw, head_roll | look / balance cues |
| 9–13 | R leg (mirror of 0–4) | |

## Sensors
- IMU on trunk (orientation / angvel for actor obs)
- Camera in head (optional vision later; walking policy can stay proprio-only)

## Sim assets path (existing Microduck)
Already in repo: `src/mjlab_microduck/robot/microduck/` (MJCF + STL).
This brief is a **custom skin / product concept**; default training still uses official MJCF until we export a new mesh set.

## Next free steps
1. Keep using official `robot_walk.xml` for training when GPU appears
2. Optional: remodel shell in CAD → onshape-to-robot → new MJCF (heavy)
3. Optional: Hub upload of trained ONNX once a free GPU run exists
