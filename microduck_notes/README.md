# Microduck 작업 기록

[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)을 무료(CPU)로만 따라가며 남긴 기록이에요. 원본 코드는 Pollen 저장소에 있고, 여기에는 과정과 정리만 담았어요.

## 환경
- 로컬 CUDA GPU가 없어서 학습(MuJoCo Warp + PPO)은 하지 않았어요. HF Jobs 같은 유료 GPU도 쓰지 않았어요.
- Python 3.12 + uv로 `uv sync`, CPU에서 `uv run --with pytest pytest tests/`로 상태를 확인해요.
- 평일 아침마다 develop 브랜치를 받아 테스트하는 무료 점검을 돌려요.

## 무료 점검 기록 (develop)
| 날짜 | 커밋 | pytest | 주요 변경 |
|---|---|---|---|
| 2026-09-15 | 3412ae5 → cb70b79 | 225 passed, 1 skipped | protective fall, distill, backlash, velstand |
| 2026-09-30 | cb70b79 → cfe1c2a | 234 passed, 1 skipped | publish robotics replay 태그, `--base-model`, HF jobs auto-export 수정 |
| 2026-10-02 | cfe1c2a → 8d0db74 | 241 passed, 1 skipped | sitstand fall back 수정, sitstand 학습 계보 문서 |
| 2026-10-05 | 8d0db74 → 1f2e4af | 251 passed, 1 skipped | pickup detector (PR #70) |
| 2026-10-06 | 1f2e4af → 273afe0 | 255 passed, 1 skipped | velstand 몸 높이·기울기 제어, yaw 수정 |
| 2026-10 (10/9~10/12) | 273afe0 → ceca017 | 255 passed, 1 skipped | 라이선스: STL 메시는 CC BY-NC-SA 4.0 |
| 2026-10-13 | ceca017 → b02fdea | 263 passed, 1 skipped | velstand v8 (일어나기 굳음 수정, 회전 개선) |
| 2026-10-28 | b02fdea → bae39f5 | 274 passed, 1 skipped | 훈련 번들(share-training), find_onnx_source.py |

## 공개 훈련 번들 정리 (2026-10-28)
[pollen-robotics/microduck-training](https://huggingface.co/pollen-robotics/microduck-training) (`policies-v8` 태그)에 ONNX 정책 9개의 체크포인트가 공개돼 있어요. 모두 4096개 환경으로 학습했어요.

| 정책 | ONNX | iter | 학습 방식 |
|---|---|---|---|
| velstand v8 | `velstand.onnx` | 2749 | `qz5g3bwi`@1750에서 이어 학습 |
| 걷기 | `alpha_walking.onnx` | 3750 | 처음부터 |
| 일어나기 | `alpha_stand.onnx` | 9750 | `66kx3731`@2500에서 이어 학습 |
| 앉았다 서기 | `alpha_sitstand.onnx` | 2250 | `jngaedcs`@1250에서 이어 학습 |
| 바닥 물건 줍기 | `alpha_ground_pick.onnx` | 2750 | 처음부터 |
| 공 차기 왼발/오른발 | `ball_kick_left/right.onnx` | 2500 / 2000 | 처음부터 |
| 웅크리기 | `roller_crouch.onnx` | 500 | 처음부터 |
| 구르기 | `roulade.onnx` | 750 | 처음부터 |

velstand v8 계보: v7 `j4i6yoq2`@1250 → v8a `qz5g3bwi`@1750 → v8b `0045i2si`@2749.

## 관련 저장소
- 한국어 대화 덕 시제품: [`../microduck_korean_chat`](../microduck_korean_chat)
