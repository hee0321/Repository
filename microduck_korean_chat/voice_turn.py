"""음성 한 턴: 입력 wav → STT → 의도/LLM → TTS 대답 wav. 단계별 지연 시간 출력.

  python voice_turn.py out/test_input.wav            # 이미 있는 wav
  python voice_turn.py --make-input "덕아, 앞으로 가 줄래?"   # TTS로 시험 입력부터 만들기
"""
import argparse, json, time, wave
from pathlib import Path
from duckchat.intent import parse
from duckchat.robot import dispatch

ap = argparse.ArgumentParser()
ap.add_argument("wav", nargs="?")
ap.add_argument("--make-input", metavar="TEXT")
ap.add_argument("--out", default="out")
a = ap.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)
T = {}

def tic(k):
    T[k] = time.perf_counter()

def toc(k):
    T[k] = time.perf_counter() - T[k]

tic("load_tts"); from duckchat.speech import TTS, STT; tts = TTS(); toc("load_tts")
tic("load_stt"); stt = STT(); toc("load_stt")
tic("load_llm"); from duckchat.brain import Brain; brain = Brain(); toc("load_llm")

wav = a.wav
if a.make_input:
    wav = str(out / "test_input.wav")
    tts_in = TTS(voice="M1")  # 사람 역할은 다른 목소리로
    tts_in.say(a.make_input, wav, speed=1.0)
with wave.open(wav) as w:
    audio_s = w.getnframes() / w.getframerate()

tic("stt"); heard = stt.transcribe(wav); toc("stt")
tic("intent"); cmd = parse(heard); toc("intent")
tic("llm"); reply = brain.reply(heard, cmd) if heard else "꽥? 잘 못 들었어. 다시 말해 줄래?"; toc("llm")
reply_wav = str(out / "reply.wav")
tic("tts"); reply_s = tts.say(reply, reply_wav); toc("tts")

print(f"입력 오디오: {wav} ({audio_s:.1f}s)")
if a.make_input: print(f"원문      : {a.make_input}")
print(f"STT       : {heard}")
print(f"명령      : {json.dumps(cmd.as_dict(), ensure_ascii=False) if cmd else '없음 (대화만)'}")
for l in dispatch(cmd): print("  " + l)
print(f"덕 대답   : {reply}")
print(f"대답 wav  : {reply_wav} ({reply_s:.1f}s)")
turn = T["stt"] + T["intent"] + T["llm"] + T["tts"]
print("지연(초)  : " + ", ".join(f"{k}={v:.2f}" for k, v in T.items()) + f" | 한 턴 합계={turn:.2f}")
