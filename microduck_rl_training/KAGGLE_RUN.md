# Kaggle 무료 GPU(T4) 실행 기록

- 커널: https://www.kaggle.com/code/shinmadam/microduck-rl-velstand (비공개)
- 계정: `shinmadam`, 무료 한도만 사용 (GPU 주 30시간, 세션 최대 12시간)
- 실행 방법: Kaggle CLI `kaggle kernels push -p <폴더> --accelerator NvidiaTeslaT4`
  - `kaggle/kernel-metadata.json` 사용 (GPU on, 인터넷 on, `machine_shape: NvidiaTeslaT4`). `code_file`은 push하는 폴더 안의 노트북 이름과 같아야 해요.
  - API 토큰은 `~/.kaggle/`에만 두고 저장소에는 올리지 않아요.

## 환경 (실측)
| 항목 | 값 |
|---|---|
| GPU | Tesla T4 x2 (15 GB, sm_75). 학습은 0번 GPU 1장만 사용 |
| 드라이버 / CUDA | 580.178.04 / CUDA 13.0 (mujoco_warp 요구치 12.4 이상 충족) |
| torch | 2.9.1+cu128, `cuda.is_available()=True` |
| warp / mujoco | warp 1.12.0 (CUDA Toolkit 12.9), mujoco 3.10.0 |
| 설치 | `uv sync` 정상, 클론부터 학습 시작까지 약 6분 |

## 시도 기록 (2026-10-28, 한국 시간)
1. v1 (14:06~14:11): 설치와 GPU 인식은 성공했어요. `--gpu-ids 0` 인자 형식 오류로 학습이 시작되지 않아서 이 인자를 빼고 기본값 `[0]`을 쓰게 했어요.
2. v2 (14:12~14:18): `num_envs=4096`에서 GPU 메모리 부족(`Failed to allocate 471859200 bytes`)이 났어요. 2048로 줄이고, 메모리가 부족하면 1024, 512로 자동 재시도하게 했어요.
3. v3 시험 실행 (14:21~14:39): 성공했어요. 2048 env로 15분 동안 공개 v8 velstand(2749)에서 2822까지 74회 이어 학습했고, `model_2800.pt`와 ONNX를 저장했어요.
4. v4 본 학습 (14:41~01:20): 같은 설정으로 10.5시간 시간 예산, 최대 5000회. 시간 예산에 도달해서 계획대로 스스로 멈췄어요(`time budget reached, stopping`). 결과는 아래 "본 학습 결과"에 있어요.

## 시험 실행 속도
| | CPU (상자, 코어 1개, 64 env) | Kaggle T4 (2048 env) |
|---|---|---|
| Steps per second | 약 177~257 | 중앙값 약 5,333 (4,800~6,100) |
| 반복 1회 | 약 7.3~8.6초 | 중앙값 약 9.2초 |
| 반복당 경험량 | 64 env | 2048 env (32배) |

- 초당 경험량은 CPU보다 약 20~30배 많아요.
- Mean reward는 0.47에서 24.57로 올랐고, Mean episode length는 11.8에서 999로 늘었어요. 이어 학습 직후 curriculum이 다시 시작돼서 처음 값이 낮게 나와요.
- 10.5시간 세션 1번이면 약 4,000회를 학습해요. Pollen v8 velstand 전체(2749회)보다 많은 양이에요. 무료 한도 주 30시간이면 이런 세션을 주 2~3번 돌릴 수 있어요.

## 본 학습 결과 (v4, 2026-10-28 14:41 ~ 10-29 01:20)
- 2749회(공개 v8)에서 6876회까지, 총 4,127회를 이어 학습했어요. 마지막 체크포인트는 `model_6850.pt`예요.
- 속도: 초당 중앙값 약 5,430스텝(4,800~6,100), 반복 1회 중앙값 약 9.1초. 오류나 NaN은 없었어요.
- Mean episode length: 마지막 100회 평균 992 (최대 1000). 쓰러지지 않고 에피소드를 거의 끝까지 버텨요. 시험 실행의 999와 비슷해요.
- Mean reward: 이어 학습 직후 2787회에 최고 31.1을 찍은 뒤 약 10으로 내려와서, 그 뒤로는 9~10.5 사이에서 거의 그대로였어요. 마지막 100회 평균 10.4로, 시험 실행의 24.57보다 낮아요.

| 구간 (반복) | Mean reward 평균 |
|---|---|
| 2749~3248 | 12.4 |
| 3249~3748 | 10.3 |
| 3749~4748 | 약 10.3 |
| 4749~5748 | 약 9.7 |
| 5749~6876 | 약 9.6 |

- 이유로 보이는 것: 지형 난이도(`terrain_levels/mean`)가 2.39에서 거의 0으로 내려갔고, 회전 오차(`error_vel_yaw`)는 2.9에서 3.9로 늘었어요. 이어 학습하면서 넘어뜨리기(topple push), 엎드린 상태에서 시작(45%) 같은 어려운 조건이 처음부터 최종 단계로 켜져 있어서, 정책이 지형 커리큘럼을 따라가지 못한 것으로 보여요. 즉 이번 4,000회는 버티기는 유지했지만 점수가 더 오르지는 않았어요.
- 시뮬레이션 영상(평지, 밀기 없음, 18초): 새 정책(6850)과 공개 v8(2749) 모두 넘어지지 않고 앞으로 걷고 좌우로 돌아요. 앞으로 0.25 m/s를 시켰을 때 실제로는 약 0.1 m/s로 걸어요. 회전은 시킨 것보다 빠르게 돌아요. 눈에 띄는 차이는 아직 없어요. 기록은 `kaggle/full_run_v4/walk_video_telemetry_*.txt`에 있어요.
- 다음에 해 볼 것: `MICRODUCK_WARM_START`로 커리큘럼을 처음부터 다시 올리거나, 지형을 평지로 고정해서 걷기 점수부터 올려 보기. 학습률을 낮춰서 v8 근처에서 미세 조정하기.

### 이어 학습용 체크포인트
- Kaggle 비공개 데이터셋 [shinmadam/microduck-velstand-ckpt](https://www.kaggle.com/datasets/shinmadam/microduck-velstand-ckpt)에 `model_6850.pt`, `agent.yaml`, `env.yaml`을 올렸어요. 저장소에는 `.pt`를 올리지 않아요.
- 다음 세션: `kernel-metadata.json`의 `dataset_sources`에 `shinmadam/microduck-velstand-ckpt`를 넣고 `RESUME_FROM="/kaggle/input/microduck-velstand-ckpt/model_6850.pt"`로 지정해요.

### 파일 (`kaggle/full_run_v4/`)
- `train_summary.log`: 100회마다 핵심 값만 뽑은 요약 로그 (원본 train.log는 23 MB라 제외)
- `metrics.csv`: 반복별 Mean reward, 에피소드 길이, 속도 추종 오차, 지형 단계
- `2026-10-28_05-44-20_free_velstand_resume_model_6850.onnx`, `agent.yaml`
- `microduck_velstand_6850_walk.mp4`: 상자 CPU에서 MuJoCo로 렌더링한 18초 걷기 영상. `render_walk.py`로 만들었어요.

## 세션 2: 걷기 속도 개선 (2026-10-29 11:46 시작, 한국 시간)
- **목표**: 시킨 속도대로 앞으로 걷기. v8도 v4(6850)도 0.4 m/s를 시키면 약 0.17~0.19 m/s, 0.2 m/s를 시키면 0.05~0.09 m/s로만 걸어요.
- **원인 (상자 CPU에서 직접 측정, `kaggle/session2/vx_probe_results.txt`)**: velstand는 서 있는 모든 프레임에서 걷기 선생님(walk anchor `alpha_walking`, 441tzs6d@3750)을 흉내 내도록 배워요(BC 계수 1.0). 그런데 이 선생님 자체가 명령의 0.4~0.5배로만 걸어요(0.4 → 0.20, 1.0 → 0.37 m/s). 그래서 학습을 더 오래 해도 속도는 선생님 수준에 묶여 있었어요. v8에서 회전이 약했던 문제와 같은 구조예요(v8은 선생님에게 더 큰 회전 명령을 넣어 줘서 고쳤어요).
- **바꾼 것** (`kaggle/session2/speed_stage.patch`, microduck_rl `bae39f5` 위에 적용, `MICRODUCK_SPEED_STAGE=1`일 때만 켜져요):
  1. **선생님 속도 보정(anchor lin remap)**: 선생님에게는 학생이 받은 속도보다 큰 명령을 넣어서, 선생님이 실제로 그 속도로 걷는 시범을 보이게 해요. 목표 0.1/0.2/0.3/0.4 m/s → 넣는 값 0.2/0.4/0.7/1.0 (측정에서 1.0까지는 넘어짐 0%, 1.2부터 넘어짐). 학생이 보는 명령은 그대로예요.
  2. **보상**: 선속도 추종 보상 가중치 2 → 3.
  3. **명령 범위**: 앞으로 치우치게 x −0.3~+0.4 m/s(원래 ±0.4), 옆걸음 ±0.2(원래 ±0.3).
  4. **쉬운 것부터**: 평지 과제 `Mjlab-VelStand-Flat-Backlash-MicroDuck` + `MICRODUCK_WARM_START=1`(가중치는 유지, 커리큘럼은 0부터). velstand의 어려운 조건은 2배 느리게 다시 올라와요: 처음 200회는 넘어지면 에피소드 종료(깨끗한 걷기), 넘어뜨리기 밀치기는 ±0.3에서 시작해 800회 ±0.9, 1600회 ±1.2 m/s, 엎드린 상태 시작은 1400회부터 15% → 2800회 45%, 일어나기 보상/벌점은 1200회부터.
- **시작점**: 공개 v8(`0045i2si@2749`). v4의 `model_6850`은 지형 난이도 붕괴(2.39 → 0)와 회전 오차 증가(2.9 → 3.9)가 있었고, 평지 속도 측정에서도 v8보다 나은 점이 없어서(0.2 m/s 명령: 0.05 vs 0.09) 쓰지 않았어요.
- **확인**: 상자 CPU에서 32 env × 3회 시험(`kaggle/session2/cpu_smoke_summary.log`): 패치 적용, warm start(step counter 66048 → 0), 선생님 2명 로드, 보상 가중치 3.0, 커리큘럼 단계(200/300/800/1200/1400/1600/2000/2800회) 모두 확인했어요.
- **Kaggle 설정**: T4, 2048 env(메모리 부족 시 1024/512 자동 재시도), 최대 6000회, 시간 예산 10시간에서 자동 정지, 체크포인트 50회마다, 30분마다 최신 체크포인트를 출력 폴더에 복사(`snapshot_model_N.pt`). 커널 v5. 예상 종료 2026-10-29 22:00~22:20.
- **무료 한도**: 시작 전 GPU 사용 11.05h / 30h(남은 18.95h). 이번 세션 약 10.3h → 끝나면 약 8.6h 남아요. 2026-10-31 09:00에 초기화돼요.
- **노트북**: `kaggle/session2/session2.ipynb`. 측정 스크립트: `kaggle/session2/vx_probe.py`.
- **볼 지표**: `track_linear_velocity` 보상, `error_vel_xy`, 끝난 뒤 평지 속도 측정(0.2/0.4 m/s 명령에서 실제 속도), 일어나기가 무너지지 않았는지.

## 결과 받기
```bash
kaggle kernels status shinmadam/microduck-rl-velstand
kaggle kernels output shinmadam/microduck-rl-velstand -p out   # out/out/에 model_N.pt, *.onnx, train.log, tfevents
```
다음 세션에서 이어 하려면 받은 `model_N.pt`를 Kaggle 데이터셋으로 올려요. 그다음 `kernel-metadata.json`의 `dataset_sources`에 넣고 `RESUME_FROM="/kaggle/input/<데이터셋>/model_N.pt"`로 지정하면 돼요.

## 파일
- `kaggle/kernel-metadata.json`: 커널 설정이에요. 비밀값은 들어 있지 않아요.
- `kaggle/microduck_kaggle_smoke.ipynb`: 15분 시험용 노트북이에요.
- `microduck_kaggle.ipynb`: 본 학습 노트북이에요. 2048 env, OOM이 나면 자동 재시도, 10.5시간 예산이에요.
- `kaggle/smoke_run/`: 시험 실행 요약 로그, agent.yaml, ONNX(`model_2800`)예요. `.pt`는 제외했어요.
