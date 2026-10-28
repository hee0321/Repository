# Microduck 강화학습

Microduck이 시뮬레이션에서 경험을 쌓으며 배우는 강화학습(PPO) 작업 폴더예요. 원본 학습 코드는 [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)을 쓰고, 여기에는 우리가 돌린 설정, 노트북, 로그, 결과를 모아요.

## 원칙
- 무료로만 진행해요. 유료 GPU나 Hugging Face Jobs는 쓰지 않아요.
- Pollen이 공개한 v8 체크포인트([microduck-training](https://huggingface.co/pollen-robotics/microduck-training))에서 이어 학습하는 것을 우선 시도해요.

## 진행 상황
- 2026-10-28: CPU(코어 1개 사용)에서 학습이 실제로 돌아가는 것을 확인했어요. 평지 걷기는 초당 약 180~250스텝이고, 공개 v8 velstand에서 2751회까지 이어 학습한 뒤 ONNX 변환까지 성공했어요. 다만 본격 학습은 CPU로 2~3주 걸려서 무료 GPU(Kaggle T4)로 하기로 했어요.

## 파일
- `microduck_kaggle.ipynb`: Kaggle/Colab 무료 GPU에서 공개 v8 velstand를 이어 학습하는 노트북이에요. `MODE="velocity_flat"`로 바꾸면 평지 걷기를 처음부터 학습해요. 10.5시간에 스스로 멈추고 저장해요.
- `train_cpu.py`: GPU에서 저장된 체크포인트를 CPU에서 열 수 있게 고친 실행 스크립트
- `cpu_runs/`: CPU 시험 로그, 설정, ONNX (체크포인트 `.pt`는 용량 때문에 제외)
- `bundles/velstand_bundle.json`: 공개 v8 velstand 번들 정보
- `NOTES.md`: 시험 세부 내용

## Kaggle에서 돌리는 법
1. Kaggle에 `microduck_kaggle.ipynb`를 업로드해요.
2. 가속기는 GPU T4 x2, 인터넷은 On으로 바꿔요. P100은 안 돼요.
3. Save Version → Save & Run All을 누르면 브라우저를 닫아도 돌아가요.
4. 다음 세션에서는 저장된 `.pt`를 `RESUME_FROM`에 넣으면 이어서 학습해요. 무료 한도는 주 30시간, 세션당 최대 12시간이에요.
