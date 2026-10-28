"""텍스트 모드 대화: 키보드로 한국어 입력 → 덕 대답 + 명령(dry-run). --tts 면 대답 wav 도 저장."""
import argparse, time
from pathlib import Path
from duckchat.intent import parse
from duckchat.brain import Brain
from duckchat.robot import dispatch

ap = argparse.ArgumentParser()
ap.add_argument("--tts", action="store_true", help="대답을 out/text_reply_N.wav 로 저장")
ap.add_argument("lines", nargs="*", help="주면 대화형 대신 이 문장들을 차례로 처리")
a = ap.parse_args()

brain = Brain()
tts = None
if a.tts:
    from duckchat.speech import TTS
    tts = TTS()
Path("out").mkdir(exist_ok=True)

def handle(i, text):
    cmd = parse(text)
    t = time.perf_counter(); reply = brain.reply(text, cmd); dt = time.perf_counter() - t
    print(f"🧑 {text}\n🦆 {reply}   ({dt:.1f}s)")
    for l in dispatch(cmd):
        print("   " + l)
    if tts:
        p = f"out/text_reply_{i}.wav"; tts.say(reply, p); print(f"   🔊 {p}")

if a.lines:
    for i, s in enumerate(a.lines): handle(i, s)
else:
    print("마이크로덕과 대화하기 (끝내려면 Ctrl+D)")
    i = 0
    while True:
        try: s = input("나> ").strip()
        except EOFError: break
        if s: handle(i, s); i += 1
