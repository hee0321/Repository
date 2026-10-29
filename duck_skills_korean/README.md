# 덕 한국어 명령 행동 (시뮬레이션, 무료, 추가 학습 없음)

"앞으로 가", "고개 왼쪽", "춤춰", "일어나", "공 따라가"처럼 **한국어로 말하면** Microduck이 MuJoCo
시뮬레이션 안에서 그대로 움직이는 데모예요. 새 강화학습은 하지 않았어요. 덕을 움직이는 신경망은 Pollen이
공개한 **velstand v8 정책**(`0045i2si` @ 2749, 실제 로봇에서 검증된 것) 그대로이고, 이 폴더는 그 정책에
넣어 주는 **명령**을 한국어로 만들어 주는 층이에요.

```
말(음성) ─ faster-whisper small(CPU) ─▶ 글자 ─ korean_commands.py(규칙) ─▶ 행동 ─ behaviors.py ─▶ 정책 명령(50 Hz)
                                                                                    │
            v8 velstand 정책: [속도 vx·vy·회전 wz] + [고개 4축] + [몸 기울기 6축] ──▶ 관절 14개 ──▶ MuJoCo 덕
```

## 영상 (`videos/`)

| 파일 | 내용 |
|---|---|
| `demo_korean_commands.mp4` | **전체 데모 66초.** 명령 자막이 나올 때마다 덕이 바뀌어요 |
| `clip_walk.mp4` | 앞으로 가 → 왼쪽으로 돌아 → 오른쪽으로 돌아 → 멈춰 |
| `clip_head.mp4` | 고개 왼쪽 / 오른쪽 / 위 / 아래 |
| `clip_dance.mp4` | 춤춰 (몸 흔들기 · 고개 갸웃·끄덕 · 제자리 돌기 · 옆걸음) |
| `clip_getup.mp4` | 시뮬레이터가 덕을 엎어 놓고 → 일어나 |
| `clip_ball.mp4` | 공 따라가 (공이 굴러가면 방향을 틀고 고개로 공을 봄) → 멈춰 |
| `demo_all_followups.mp4` | **추가 작업 모아 보기 143초** = 아래 5개 영상을 이어 붙인 것 |
| `demo_ack_head_raise.mp4` | 알아들음 신호 34초: 명령을 알아들으면 고개를 0.9초 들었다가 명령 실행 |
| `demo_obstacle_avoid.mp4` | 장애물 피하기 33초: "장애물 피해서 가" → 상자·원기둥 피해서 3 m 가서 "도착!" |
| `demo_pollen_skills.mp4` | Pollen 공개 정책 바꿔 끼우기 52초 (굴러 · 웅크려(대체) · 앉아/일어서 · 주워 · 왼발/오른발 차기 · 선생님처럼 일어나 · 부리로 물어) |
| `clip_roulade.mp4`, `clip_crouch_substitute.mp4`, `clip_sitstand.mp4`, `clip_ground_pick.mp4`, `clip_kick_left.mp4`, `clip_kick_right.mp4`, `clip_alpha_stand.mp4`, `clip_beak_pick.mp4` | 위 영상을 행동별로 자른 것 |
| `clip_roller_crouch.mp4` | 진짜 `roller_crouch.onnx` — **롤러스케이트 덕 모델**에서 (걷는 덕에서는 넘어져서 따로 찍음) |
| `demo_beak_fetch_closeup.mp4` | 부리 가까이 15초: "물어와" → 부리 벌리기 → 숙여서 물기 → 서서 돌아 걸어오기 |
| `*_stats.json` / `demo_korean_commands_stats.json` | 구간별 실제 측정값 (넘어짐 여부, 속도, 회전, 몸 기울기, 공까지 거리) |

## 행동별 결과 (전부 같은 v8 정책, 추가 학습 0시간)

| 말 | 어떻게 | 결과 |
|---|---|---|
| 앞으로 가 / 멈춰 / 왼쪽·오른쪽으로 돌아 | 속도·회전 명령 | **됨.** 0.3 m/s 를 시키면 실제 약 0.15 m/s (v8 한계, 지금 Kaggle 세션 2가 고치는 중). 회전 0.9 rad/s 명령 → 왼쪽 1.3, 오른쪽 0.8 rad/s, 넘어짐 0 |
| 고개 왼쪽/오른쪽/위/아래 | 고개 명령(neck_pitch, head_pitch, head_yaw, head_roll) | **됨.** 정책이 원래 배운 기능. 몸은 거의 안 흔들림(기울기 1° 미만) |
| 일어나 | 시뮬레이터가 덕을 엎어 놓음 → 정책이 스스로 일어남 | **됨.** 엎드린 상태에서 약 1초 안에 일어섬. 다만 '일어나'라는 말 때문이 아니라 v8 정책이 넘어지면 **항상 스스로** 일어나요. 말은 시연용 신호예요 |
| 춤춰 | 정해진 안무 = 몸 기울기(±12° 학습 범위 안) + 고개 + 제자리 돌기 + 옆걸음을 박자에 맞춰 바꿈 | **부분적으로 됨.** 넘어지지 않고 몸이 실제로 10°까지 좌우·8°까지 앞뒤로 기울어요. 하지만 몸을 위아래로 튕기기(앉았다 서기)는 v8이 못 배워서 빠졌고, 동작이 작고 '춤'이라기보다 율동에 가까워요 |
| 공 따라가 | 시뮬레이터가 아는 공 위치 → 공 쪽으로 회전·전진 + 고개로 공 보기, 22 cm 안이면 멈춤 | **시뮬레이션에서는 됨.** 0.72 m 떨어진 공에 다가가서 굴러가는 공을 20초 동안 평균 0.28 m 거리로 따라가요. 마지막에 공이 덕 쪽으로 굴러올 때는 발 앞 0.16 m까지 붙어요. 단 실제 로봇은 **공을 볼 카메라/인식이 아직 없어서** 그대로는 못 해요 |
| 음성 → 행동 | `voice_to_behavior.py`: Whisper small + 덕 명령 어휘 힌트 + 비슷한 말 보정 | 합성 음성으로 4번 시험, 48개 중 44개 맞음(약 92%). 짧은 말("춤춰", "고개 들어")이 가끔 틀림. 애매하면 움직이지 않음. 진짜 사람 목소리로는 아직 시험 안 함 |

## 추가 1 — 알아들음 신호 (고개 들기)

명령을 알아들으면 **모든 행동 공통으로**(`behaviors.Acknowledged`) 고개를 0.9초 동안 부드럽게(코사인 경사) 들어요
(head_pitch 명령 −0.75 rad, 이 정책은 +가 아래). '고개 …'·춤·Pollen 정책 행동은 신호가 끝난 뒤 시작,
'일어나'는 일어선 직후, '공 따라가'는 들었다가 바로 따라가기. 실제 관절 head_pitch 는 −0.67 ~ −0.78 rad 까지
움직였고 신호 중 넘어짐 0 (`demo_ack_head_raise_stats.json`).

## 추가 2 — 장애물 피하기 (가상 거리 센서)

- 장애물: 넘을 수 없는 상자·원기둥(높이 16–20 cm, `duck_sim.OBSTACLES`)
- 센서: 몸통 높이 9 cm 에서 −82.5°~+82.5° 로 7.5° 마다 MuJoCo `mj_ray` 광선 23개 (장애물 그룹만 봄) = 가상 거리 센서
- 판단(`behaviors.Avoider`, 규칙 기반): 방향마다 폭 42 cm 통로 안 가장 가까운 장애물 거리 → "목표 방향에서 벗어난 정도 + 막힌 정도"
  비용이 가장 작은 방향으로 회전, 옆이 30 cm 보다 가까우면 반대쪽으로 밀어내기, 앞이 막히면 감속. 명령은 v8 속도·회전·고개.
  고개는 가까운 장애물 쪽을 흘끗 봐요.
- 말: "장애물 피해(서 가)" = 코스 방향 3 m 앞 목표로 가기. "앞으로 가"로 걷는 중에도 자동으로 피함.
- **결과** (`demo_obstacle_avoid.mp4`, 33초): 상자 → 원기둥 순서로 피하고 약 24초에 "도착!".
  로봇 충돌 메시와 장애물 사이 **최소 거리 0.213 m, 접촉 0회, 넘어짐 0**. 조정할 때 돌린 다른 시행들은 최소 0.14–0.15 m.
- 한계: 시뮬레이터가 광선을 직접 쏘는 것이라 실제 로봇엔 이런 거리 센서가 없어요(카메라 깊이 추정이나 ToF 센서가 필요).
  느려요(평균 0.1–0.15 m/s). 장애물이 지그재그로 촘촘하면(85 cm 간격, 좌우 ±15–20 cm) 제자리에서 맴도는 경우가 있었어요 —
  그래서 영상 코스는 정면 장애물 2개. "앞으로 가" 중 자동 피하기는 이전 시행에서 접촉 0이었지만 첫 상자 근처에서 오래 머뭇거렸어요.

## 추가 3 — Pollen 공개 정책 바꿔 끼우기 (`policies.py`)

`pollen-robotics/microduck-policies` 의 ONNX 를 익명으로 읽어서(계정·업로드 없음) onnxruntime CPU 로 돌려요.
모든 정책이 같은 관측 61개·행동 14개·50 Hz·같은 기본 자세라서, 명령 자리(twist·고개·몸)만 manifest 대로 채우면
v8 과 같은 시뮬레이션에서 바로 바꿔 끼울 수 있어요. 끝나면 v8 velstand 로 돌아와요. 모든 명령에 알아들음 신호 있음.

| 말 | 정책 | 결과 |
|---|---|---|
| 굴러 | `roulade.onnx` (1초) → v8 | **됨.** 실제로 앞으로 한 바퀴 돌고(몸이 완전히 뒤집힘) 약 1초 뒤 v8 로 서 있음 |
| 웅크려 | `roller_crouch.onnx` | **걷는 덕에서는 안 됨 → 대체.** 롤러스케이트(바퀴 발) 덕용이라 걷는 모델에선 넘어져요(시험함). 그래서 (1) 걷는 덕에선 `alpha_sitstand.onnx` 로 쪼그렸다 일어서기로 **대신**하고 화면에 그렇게 써 놨고, (2) 진짜 roller_crouch 는 롤러 덕 모델(`Mjlab-RollerCrouch-Flat`)에서 따로 찍었어요(`clip_roller_crouch.mp4`, 몸 높이 0.125 → 0.069 m → 다시 0.12 m, 안 넘어짐). 롤러 정책은 manifest 대로 행동 ×0.8 이 필요했고, 무작위화(마찰·질량 등)를 켠 시행 하나는 넘어졌어요 |
| 앉아 / 일어서 | `alpha_sitstand.onnx` (vx=1 앉기/0 서기, 2초 경사) | **됨, 다만 얕아요.** 몸이 약 0.116 → 0.06 m 로 내려가지만(직접 측정) 영상에선 무릎 굽힌 '쪼그려 앉기'로 보여요 |
| 바닥 물건 주워 | `alpha_ground_pick.onnx` (위상 4초 주기, 2.8초) | **숙이기는 됨.** 입 끝이 바닥 2 cm 높이까지 내려갔다 올라와요. 이 정책 자체는 집지 않아요(입 관절 없음) → 추가 4 참고 |
| 왼발로 / 오른발로 공 차 | `ball_kick_left/right.onnx` (0.5초) | **됨.** 학습 때 자리(발 앞 9 cm, 옆 ±4.2 cm)에 공을 놓으면 찬 공이 2–3 m 굴러가요. 공 위치는 시뮬레이터가 놓아 줌 |
| 선생님처럼 일어나 | `alpha_stand.onnx` (3초) | **됨.** 엎어 놓으면 바로 일어남. v8 도 원래 혼자 일어나서 큰 차이는 없어요 |
| 옛날처럼 걸어 | `alpha_walking.onnx` | 코드만 있음(선택 사항, 영상 없음) |

## 추가 4 — 부리 (`beak.py`)

- **학습용 XML 은 그대로**고, 시뮬레이션이 불러온 모델 **사본**(MjSpec)에만 주황·노랑 오리 부리를 붙여요:
  윗부리(머리에 고정) + 아랫부리(경첩 관절 `passive_beak`, 0~0.8 rad) + 위치 액추에이터 `beak_motor`.
  관절 이름이 `passive_` 로 시작해서 정책의 관측·행동(14개)에서 자동으로 빠지고, 부리는 시연 코드가 직접 움직여요.
  (부리 때문에 보상 항 일부가 액추에이터 15개를 못 받아서 시연에선 보상·커리큘럼을 껐어요 — 정책 실행엔 영향 없음.)
- "부리로 물어" / "물어와": 2 cm 파란 상자를 입 아래 놓음 → 부리 벌리기 → `alpha_ground_pick.onnx` 로 숙이기 →
  가장 낮을 때 부리 닫기(0.42 rad) → v8 로 서서 물고 있기 → ("물어와"면) 제자리에서 돌아 0.5 m 걸어오기.
- **정직하게: 물건을 무는 건 물리 집기가 아니에요.** 부리를 닫는 순간 물건을 아랫부리 좌표에 **운동학적으로 붙여요**(weld 대신
  매 스텝 위치를 옮기는 방식, 물건은 질량·충돌 없는 보기용). 붙이는 순간 물건이 1.1–1.5 cm 옮겨 가요. 부리는 몸·바닥과도 충돌하지 않아요.

## 실행 (이 박스 기준, 전부 CPU)

```bash
# 시뮬레이션 데모 (약 11분, microduck_rl 의 .venv 사용)
cd /workspace/microduck_rl
.venv/bin/python ../duck_skills_korean/run_demo.py                 # 기본 대본 → videos/
.venv/bin/python ../duck_skills_korean/run_demo.py --script my.txt --out videos/my.mp4 --end 30
.venv/bin/python ../duck_skills_korean/run_demo.py --preset ack       # 알아들음 신호 (약 5분)
.venv/bin/python ../duck_skills_korean/run_demo.py --preset obstacle  # 장애물 (약 8분, --no-video 면 측정만)
.venv/bin/python ../duck_skills_korean/run_demo.py --preset skills    # Pollen 정책들 + 행동별 클립 (약 11분)
.venv/bin/python ../duck_skills_korean/run_demo.py --preset beak      # 부리 가까이 (약 3분)
.venv/bin/python ../duck_skills_korean/roller_crouch_clip.py          # 롤러 덕 웅크리기 (약 1.5분)
#   my.txt 예:  0 앞으로 가 / 5 고개 왼쪽 / 8 춤춰 / 22 멈춰   (한 줄에 "시작초 명령")

python3 korean_commands.py "덕아 공 쫓아가 봐" "오른쪽 봐"           # 파서만 시험
/workspace/voice_venv/bin/python voice_to_behavior.py --selftest    # 음성 → 행동 (TTS로 만든 명령 음성)
/workspace/voice_venv/bin/python voice_to_behavior.py 내목소리.wav
```

v8 체크포인트는 공개 저장소 `pollen-robotics/microduck-training`의 `bundles/velstand/0045i2si_2749/model_2749.pt`
(읽기만 함, 계정 필요 없음)이고, `--ckpt` 로 다른 체크포인트(Kaggle 세션 결과 등)를 줄 수 있어요.
음성은 `../microduck_korean_chat/`의 STT·TTS 코드와 모델을 그대로 씁니다.

## 파일

- `korean_commands.py` 한국어 → 행동 규칙 (+ 음성 인식 오타용 비슷한 말 보정 `parse_fuzzy`)
- `behaviors.py` 행동 → 매 스텝 정책 명령 (걷기, 멈춤, 고개, 일어나기, 춤 안무, 공 따라가기, 장애물 피하기, 알아들음 신호)
- `duck_sim.py` mjlab 시뮬레이션 래퍼: v8 정책 불러오기, 공(지름 7 cm)·장애물·집을 물건·부리 추가, 광선 거리 센서, 넘어뜨리기, 명령 넣기
- `policies.py` Pollen 공개 ONNX 정책 바꿔 끼우기 (굴러·웅크려 대체·앉아/일어서·주워·공 차기·선생님 일어나기·부리로 물어)
- `beak.py` 시연용 모델 사본에 부리(경첩+액추에이터) 붙이기
- `roller_crouch_clip.py` 롤러 덕 모델에서 roller_crouch 영상
- `run_demo.py` 대본 실행 → 자막 MP4 + 구간별 측정값 + 행동별 클립
- `voice_to_behavior.py` 음성 wav → 행동, `voice_selftest.log` 시험 기록
- `NEXT_STEPS.md` 다음 무료 GPU 학습 계획과 실제 로봇에 음성 붙이는 방법

## 정직한 한계

- 모두 **시뮬레이션**이에요. 실제 로봇 명령(robotd)에는 속도·스킬·소리는 있지만, 고개를 따로 돌리는
  명령이 있는지는 아직 확인 못 했어요.
- 공은 대본대로 굴러가는 '움직이는 표적'이에요(덕이 차서 공이 굴러가는 물리는 아님). 덕은 공 위치를
  시뮬레이터에서 바로 받아요.
- 춤은 새로 배운 동작이 아니라 이미 아는 동작을 이어 붙인 안무라서 크기가 작아요.
