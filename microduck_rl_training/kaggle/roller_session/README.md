# 롤러스케이트 학습 세션 (Kaggle 무료 T4) — 준비만, 아직 안 돌림

10/31 09:00 (KST) 무료 GPU 한도(주 30시간)가 다시 채워진 뒤 첫 세션용이에요. 비용 0원, Hugging Face 계정·업로드 없음
(Pollen 공개 파일 `roller.onnx` 를 읽기만 함), `.pt` 는 GitHub 에 올리지 않아요. 지금 돌고 있는 세션 2 커널
(`shinmadam/microduck-rl-velstand`)은 건드리지 않고, **새 비공개 커널 `shinmadam/microduck-rl-roller`** 로 올릴 예정이에요(`kernel-metadata.json`).

| 파일 | 내용 |
|---|---|
| `roller_session.ipynb` | Kaggle 노트북 (설정 → 설치 → warm start 준비 → 학습(30분 스냅샷, 10시간 자동 정지) → 요약 → ONNX + 속도 측정) |
| `onnx_to_ckpt.py` | 공개 `roller.onnx` → rsl_rl 체크포인트 변환 (노트북 안에도 같은 내용) |
| `roller_probe.py` | 새 ONNX 와 공개 roller.onnx 의 속도·방향·자세를 같은 조건으로 비교 |
| `kernel-metadata.json` | 새 커널 설정 (T4, 비공개, 인터넷 켬) |

## 무엇을 배우나
- 작업 `Mjlab-Velocity-Flat-MicroDuck-Rollers` (`robot_groundcontact_rollers.xml`: 관절 14개 + 발 밑 수동 바퀴 4개, 평지).
  cmd_x: 0 = 서기/미끄러지기, + = 밀기, − = 브레이크. 보상의 중심은 `wheel_speed`(바퀴를 실제로 굴려야 점수), 직진 유지는 `heading_hold`.
- 공개 roller.onnx 시뮬 측정(`roller_probe.py`, 시작 방향 고정, 무작위화 끔, 마지막 5초 평균):
  cmd 0.2 → 0.00 m/s(그냥 섬), 0.4 → 0.38 m/s, 0.6 → 0.40 m/s(방향이 최대 40° 돌아감). 0.2 이하 명령을 무시하고 0.4 위로는 거의 안 빨라지는 걸
  고치는 게 목표예요(명령에 맞는 속도 + 직진).

## warm start (기본, `MODE = "roller_warm"`)
- `roller.onnx` 는 `Sub(mean) → Div(std) → MLP 61→512→256→128→14 (ELU)` 라 이 저장소 actor 와 모양이 같아요.
  가중치와 관측 정규화를 그대로 복사하고, manifest 의 `action_scale 0.8`(환경은 1.0)은 **마지막 층에 0.8 을 곱해 접어 넣어요**.
  → 이렇게 만든 체크포인트/ONNX 는 **action_scale 1.0** 으로 써요.
- 박스 CPU 시험 (전부 통과):
  - 변환 정확도: 무작위 관측 256개에서 `|torch actor − 0.8 × onnx| ≤ 1.9e-6`.
  - 변환 → `scripts/export.py` → 다시 측정: cmd 0.4 → 0.376 m/s (원본 0.385), 0.6 → 0.417 (원본 0.399) — 같은 정책.
  - `--init-checkpoint` + `MICRODUCK_WARM_START=1` 로 64 env × 4회 학습: 평균 보상 +0.09 → +1.31, wheel_speed 0 → 0.066.
    같은 조건 처음부터는 −12.6 → −18.3, wheel_speed 0.004. (CPU 는 1회 약 16초라 아주 짧은 시험)
- critic(가치 함수)은 새로 배워서 처음 수백 회는 actor 가 조금 흔들릴 수 있어요. 행동 잡음 std 는 0.3 으로 작게 시작(`WARM_STD`).
- 처음부터 하려면 `MODE = "roller_scratch"` (std 1.0).

## 예상 시간 (T4, 2048 env)
- 걷기 velstand(Rough·백래시)가 T4 에서 1회 평균 9.0초였어요. 롤러 모델은 평지·접촉이 적어서 약 6–9초/회로 예상 → **10시간 ≈ 4,000–6,000회**.
- warm start: 이미 타는 정책에서 시작하니 **2–4시간(1,000–2,000회)** 이면 속도·직진 변화가 보일 것으로 예상. 바퀴 베어링 마찰 커리큘럼이
  2,000·3,500·5,000회에 올라가므로 그 지점마다 스냅샷을 비교해요.
- 처음부터: 쓸 만한 타기까지 3,000–5,000회 = **약 8–12시간** (세션 1–2회).
- 주 30시간 중 춤(안 A, 약 4–5시간)과 나눠 쓰려면 `TIME_BUDGET_H = 5.0` 정도로 줄여도 돼요.

## 돌리는 법 (아직 안 함)
```bash
export PATH=$HOME/.local/bin:$PATH; export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)   # 토큰은 출력·커밋 금지
cd microduck_rl_training/kaggle/roller_session && kaggle kernels push -p .
kaggle kernels status shinmadam/microduck-rl-roller
kaggle kernels output shinmadam/microduck-rl-roller -p ./out     # 끝난 뒤
```
결과 확인: `out/roller_probe.json` (새 정책 vs 공개 roller, cmd 0.2/0.4/0.6), `train.log` 의 `wheel_speed`.
그다음 `duck_skills_korean/roller_demo.py` 에서 `roller.onnx` 대신 새 ONNX(scale 1.0)로 같은 영상을 다시 찍어 비교해요.
