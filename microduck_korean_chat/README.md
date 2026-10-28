# 마이크로덕 한국어 대화 프로토타입 (CPU 전용, 무료)

덕의 보행/동작 RL 정책(관절 관측 + 속도 명령 → 관절 목표)은 언어를 다루지 않습니다. 그래서 한국어 대화는
**정책을 다시 학습하지 않고**, 미리 학습된 무료 공개 모델을 엮은 별도 계층으로 만들었습니다.

```
마이크(wav) → STT(faster-whisper small, int8) → 의도 규칙(한국어 → robotd 명령)
                                               ↘ 대화면 LLM(Qwen2.5-1.5B-Instruct Q4, llama.cpp)
                                                 → TTS(Supertonic 2, 한국어) → 대답 wav
                                               → robotd JSON-RPC 명령 (지금은 화면 출력만, dry-run)
```

GPU, 유료 API, HF Jobs 전부 안 씁니다. 모델을 내려받은 뒤에는 인터넷 없이 돌아갑니다.

## 실행

```bash
cd /workspace/microduck_korean_chat
uv venv -p 3.12 .venv && . .venv/bin/activate
uv pip install -r requirements.txt     # llama-cpp-python 은 CPU 휠:
#   uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --only-binary llama-cpp-python
python setup_models.py                 # 약 1.8 GB (처음 한 번)

python chat_text.py                    # 텍스트로 대화 (Ctrl+D 로 끝)
python chat_text.py --tts "안녕!" "공 차 봐"   # 문장 넣고 대답 wav 도 out/ 에 저장
python voice_turn.py 내목소리.wav       # 음성 한 턴: wav → 대답 out/reply.wav (+ 단계별 지연 시간)
python voice_turn.py --make-input "덕아, 앞으로 가 줄래?"   # TTS로 시험 음성부터 만들어서 한 턴
python demo_voice.py                   # 8턴 데모 → out/demo_transcript.md, out/demo_results.json
python -m duckchat.intent "왼쪽으로 돌아" "3초 동안 뒤로 가"  # 의도 규칙만 시험
```

## 알아듣는 말 → 실제 명령

명령 이름은 `pollen-robotics/microduck`(commit `5def2f3`)의 `duck-ipc-proto/src/lib.rs`와
cheatsheet 에서 확인한 것만 씁니다. 소켓은 `/run/robotd.sock`, 줄 단위 JSON-RPC 2.0
(`microduck_rl/scripts/pickup_e2e_robotd.py` 가 쓰는 방식과 같음).

| 한국어 예 | 명령 |
|---|---|
| 앞으로 가 / 걸어 / 이리 와 | `robot.move {vx: 0.2}` 를 N초(기본 2초, "3초 동안" 인식) 반복 후 `robot.stop` |
| 뒤로 가 | `robot.move {vx: -0.15}` |
| 왼쪽으로 돌아 / 돌아 | `robot.move {vyaw: +0.7}` (양수 = 왼쪽) |
| 오른쪽으로 돌아 | `robot.move {vyaw: -0.7}` |
| 왼쪽/오른쪽으로 가 | `robot.move {vy: ±0.15}` |
| 멈춰 / 그만 | `robot.stop` (속도 0, 힘 빼기 아님) |
| 앉아 / 일어나 | `robot.do {skill: "sit_toggle"}` (토글이라 실제론 `robot.state` 확인 후 보내야 함) |
| 공 차 / 오른발로 공 차 | `robot.do {skill: "kick_left"}` / `"kick_right"` |
| 주워 / 집어 | `robot.do {skill: "ground_pick"}` |
| 굴러 | `robot.do {skill: "roulade"}` |
| 꽥 해 / 소리 내 | `robot.sound {tag: "chirp"}` |

속도는 microduck_rl 학습 범위(vx ±0.4 m/s, vy ±0.3, vyaw ±1.0 rad/s) 안에서 보수적으로 잡았습니다.
"넘어졌을 때 일어나기"는 명령이 아니라 velstand 정책이 알아서 합니다.

명령이 있는 턴은 LLM 대신 **정해진 대답**(`duckchat/brain.py` 의 `ACKS`)을 씁니다. 1.5B 모델이 가끔
"일어나기 싫어"처럼 동작과 반대로 말해서입니다. `reply(..., llm_for_commands=True)` 로 바꿀 수 있어요.

## 파일

- `duckchat/intent.py` 한국어 → 명령 규칙, `duckchat/brain.py` LLM + 덕 페르소나,
  `duckchat/speech.py` STT/TTS, `duckchat/robot.py` dry-run 출력
- `chat_text.py` 텍스트 대화, `voice_turn.py` 음성 한 턴, `demo_voice.py` 다턴 데모
- `out/` 결과 wav 와 대화 기록

## 모델과 라이선스

| 용도 | 모델 | 크기 | 라이선스 | 상업적 사용 |
|---|---|---|---|---|
| STT | `Systran/faster-whisper-small` (OpenAI Whisper small) | 464 MB | MIT | 가능 |
| LLM | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` q4_k_m | 1.1 GB | Apache-2.0 | 가능 (주의: Qwen2.5 **3B**·72B 는 별도 라이선스) |
| TTS | `Supertone/supertonic-2` (ONNX, 66M) | 256 MB | 모델 OpenRAIL-M, 코드 MIT | 가능하나 사용 제한 조항(기만, 동의 없는 음성 복제 등)을 지키고 배포 시 같은 제한을 넘겨야 함 |
| 런타임 | llama.cpp / llama-cpp-python, CTranslate2, faster-whisper, onnxruntime | | MIT | 가능 |

## 한계

- 시험 입력은 사람 목소리가 아니라 TTS 합성 음성입니다. 실제 마이크·소음·아이 목소리에서는 정확도가 다를 수 있어요.
- Whisper 가 "덕아"를 "더 가"로 받아적는 일이 잦아서 문장 앞 호출어는 지웁니다(`strip_wake`).
  짧고 깨끗한 클립에서 VAD 를 켜면 첫 단어가 잘려서 기본값은 VAD 끔.
- 1.5B LLM 이라 존댓말/반말이 섞이고 가끔 엉뚱한 말을 합니다. 지식 질문은 약해요.
- 덕 본체(RK3566, RAM 1 GB)에서는 그대로 못 돌립니다. 이 박스에서 한 턴 최대 RSS 약 3 GB.
