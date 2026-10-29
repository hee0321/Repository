# Microduck 강화학습

Microduck이 시뮬레이션에서 경험을 쌓으며 배우는 강화학습(PPO) 작업 폴더예요. 원본 학습 코드는 [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)을 쓰고, 여기에는 우리가 돌린 설정, 노트북, 로그, 결과를 모아요.

## 원칙
- 무료로만 진행해요. 유료 GPU나 Hugging Face Jobs는 쓰지 않아요.
- Pollen이 공개한 v8 체크포인트([microduck-training](https://huggingface.co/pollen-robotics/microduck-training))에서 이어 학습하는 것을 우선 시도해요.

## 진행 상황
- 2026-10-28: CPU(코어 1개 사용)에서 학습이 실제로 돌아가는 것을 확인했어요. 평지 걷기는 초당 약 180~250스텝이고, 공개 v8 velstand에서 2751회까지 이어 학습한 뒤 ONNX 변환까지 성공했어요. 다만 본격 학습은 CPU로 2~3주 걸려서 무료 GPU(Kaggle T4)로 하기로 했어요.
- 2026-10-28: Kaggle 무료 T4에서 시험 실행에 성공했어요. 2048 env로 초당 약 5,300스텝이 나와 CPU보다 20~30배 빨라요. 이어서 10.5시간 본 학습을 시작했어요. 자세한 내용은 [KAGGLE_RUN.md](KAGGLE_RUN.md)에 있어요.
- 2026-10-29: Kaggle 본 학습이 10.5시간 만에 계획대로 끝났어요. 2749→6876회(4,127회)를 학습했고 에피소드 길이는 약 992로 잘 버티지만, Mean reward는 약 10에서 더 오르지 않았어요. 체크포인트는 Kaggle 비공개 데이터셋에 올렸고, 걷기 영상(`kaggle/full_run_v4/microduck_velstand_6850_walk.mp4`)도 만들었어요.

## 파일
- `microduck_kaggle.ipynb`: Kaggle/Colab 무료 GPU에서 공개 v8 velstand를 이어 학습하는 노트북이에요. T4에 맞게 2048 env를 쓰고, 메모리가 부족하면 자동으로 줄여요. `MODE="velocity_flat"`로 바꾸면 평지 걷기를 처음부터 학습해요. 10.5시간에 스스로 멈추고 저장해요.
- `train_cpu.py`: GPU에서 저장된 체크포인트를 CPU에서 열 수 있게 고친 실행 스크립트
- `cpu_runs/`: CPU 시험 로그, 설정, ONNX (체크포인트 `.pt`는 용량 때문에 제외)
- `bundles/velstand_bundle.json`: 공개 v8 velstand 번들 정보
- `NOTES.md`: 시험 세부 내용
- `KAGGLE_RUN.md`, `kaggle/`: Kaggle 실행 기록, 커널 설정, 시험 실행 로그
- `kaggle/full_run_v4/`: 10.5시간 본 학습 요약 로그, 지표, ONNX, 걷기 영상
- 신경망 활동 3D 시각화: `../duck_brain_viz/` (https://hee0321.github.io/Repository/)

## Kaggle에서 돌리는 법
1. Kaggle에 `microduck_kaggle.ipynb`를 업로드해요. CLI를 쓰면 `kaggle kernels push -p <폴더> --accelerator NvidiaTeslaT4`로 올려요. 이때 `kaggle/kernel-metadata.json`을 참고하세요.
2. 가속기는 GPU T4 x2, 인터넷은 On으로 바꿔요. P100은 안 돼요.
3. Save Version → Save & Run All을 누르면 브라우저를 닫아도 돌아가요.
4. 다음 세션에서는 저장된 `.pt`를 `RESUME_FROM`에 넣으면 이어서 학습해요. 무료 한도는 주 30시간, 세션당 최대 12시간이에요.
