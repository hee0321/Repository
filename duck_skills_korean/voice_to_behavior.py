"""음성(wav) → 글자(faster-whisper small, CPU) → 행동. 무료·오프라인.

STT/TTS 모델과 코드는 ../microduck_korean_chat 것을 그대로 씁니다 (먼저 그 폴더의 setup_models.py).
  python voice_to_behavior.py 내목소리.wav ...      # 녹음 파일로
  python voice_to_behavior.py --selftest            # TTS로 명령 음성을 만들어 끝까지 시험
"""
from __future__ import annotations
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = os.environ.get("DUCKCHAT_DIR", os.path.join(HERE, "..", "microduck_korean_chat"))
sys.path.insert(0, HERE); sys.path.insert(0, CHAT)
from korean_commands import parse, parse_fuzzy  # noqa: E402

PHRASES = ["덕아, 앞으로 가", "멈춰", "왼쪽으로 돌아", "오른쪽으로 돌아", "고개 왼쪽", "고개 오른쪽",
           "고개 들어", "아래 봐", "일어나", "춤춰", "춤 춰 봐", "공 따라가"]


# 덕 명령 어휘를 Whisper 에 미리 알려 주면(initial prompt) 짧은 명령을 훨씬 잘 알아듣습니다.
VOCAB_PROMPT = "덕에게 하는 짧은 명령: 앞으로 가. 멈춰. 왼쪽으로 돌아. 오른쪽으로 돌아. 고개 왼쪽. 아래 봐. 일어나. 춤춰. 공 따라가."


def main():
    import duckchat.speech as speech
    speech.PROMPT = VOCAB_PROMPT
    stt = STT = speech.STT()
    if sys.argv[1:] == ["--selftest"]:
        from duckchat.speech import TTS
        tts = TTS()
        os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
        files = []
        for i, p in enumerate(PHRASES):
            f = os.path.join(HERE, "out", f"cmd_{i:02d}.wav")
            tts.say(p, f)
            files.append((p, f))
    else:
        files = [(None, f) for f in sys.argv[1:]]
    ok = 0
    for want, f in files:
        t = time.time(); text = stt.transcribe(f); dt = time.time() - t
        b = parse(text) or parse_fuzzy(text)
        good = want is None or (b is not None and parse(want) is not None and b.label == parse(want).label)
        ok += good
        print(f"{(want or '-'):14} | 들은 말: {text:16} | {b.label if b else '(모르는 말)'} | {dt:.1f}s {'OK' if good else 'X'}")
    if files and files[0][0]:
        print(f"맞음 {ok}/{len(files)}")


if __name__ == "__main__":
    main()
