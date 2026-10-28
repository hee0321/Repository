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
4. v4 본 학습 (14:41~): 같은 설정으로 10.5시간 시간 예산, 최대 5000회. 50회마다 체크포인트를 저장해요.

## 시험 실행 속도
| | CPU (상자, 코어 1개, 64 env) | Kaggle T4 (2048 env) |
|---|---|---|
| Steps per second | 약 177~257 | 중앙값 약 5,333 (4,800~6,100) |
| 반복 1회 | 약 7.3~8.6초 | 중앙값 약 9.2초 |
| 반복당 경험량 | 64 env | 2048 env (32배) |

- 초당 경험량은 CPU보다 약 20~30배 많아요.
- Mean reward는 0.47에서 24.57로 올랐고, Mean episode length는 11.8에서 999로 늘었어요. 이어 학습 직후 curriculum이 다시 시작돼서 처음 값이 낮게 나와요.
- 10.5시간 세션 1번이면 약 4,000회를 학습해요. Pollen v8 velstand 전체(2749회)보다 많은 양이에요. 무료 한도 주 30시간이면 이런 세션을 주 2~3번 돌릴 수 있어요.

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
