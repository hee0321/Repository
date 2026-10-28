"""여러 턴 음성 데모: 각 문장을 TTS(M1 목소리)로 '사람 음성'을 만들고 → STT → 의도/LLM → TTS(F1) 대답.
대답 wav 를 다시 STT 로 돌려 발음이 알아들을 만한지도 확인. 결과: out/demo_transcript.md"""
import json, time, wave
import numpy as np
np.random.seed(0)  # Supertonic 은 잡음 샘플링을 하므로 재현용 시드
from pathlib import Path
from duckchat.speech import TTS, STT
from duckchat.brain import Brain
from duckchat.intent import parse
from duckchat.robot import dispatch

LINES = ["안녕 덕아, 넌 이름이 뭐야?", "덕아, 앞으로 걸어가 줄래?", "이제 왼쪽으로 돌아.",
         "잠깐 앉아 볼래?", "다시 일어나!", "오른발로 공 차 봐.", "오늘 기분 어때?", "멈춰!"]
out = Path("out"); out.mkdir(exist_ok=True)
human, duck, stt, brain = TTS("M1"), TTS("F1"), STT(), Brain()
rows, md = [], ["# 마이크로덕 한국어 음성 데모 (CPU)", ""]
for i, said in enumerate(LINES):
    inp = out / f"demo_in_{i}.wav"; human.say(said, str(inp), speed=1.0)
    t0 = time.perf_counter(); heard = stt.transcribe(str(inp)); t1 = time.perf_counter()
    cmd = parse(heard); t2 = time.perf_counter()
    reply = brain.reply(heard, cmd) if heard else "꽥? 잘 못 들었어. 다시 말해 줄래?"; t3 = time.perf_counter()
    rw = out / f"demo_reply_{i}.wav"; dur = duck.say(reply, str(rw)); t4 = time.perf_counter()
    back = stt.transcribe(str(rw))
    r = dict(said=said, heard=heard, cmd=cmd.as_dict() if cmd else None, reply=reply, reply_wav=str(rw),
             reply_audio_s=round(dur, 2), reply_heard_back=back,
             stt_s=round(t1 - t0, 2), intent_ms=round((t2 - t1) * 1000, 2), llm_s=round(t3 - t2, 2),
             tts_s=round(t4 - t3, 2), turn_s=round(t4 - t0, 2))
    rows.append(r); print(json.dumps(r, ensure_ascii=False))
    md += [f"**사람(원문)**: {said}  ", f"**STT**: {heard}  ",
           f"**명령**: `{cmd.method} {json.dumps(cmd.params, ensure_ascii=False)}` ({cmd.action})  " if cmd else "**명령**: 없음  ",
           f"**덕**: {reply}  ", f"*(대답 wav 재인식: {back})*  ",
           f"지연: STT {r['stt_s']}s · LLM {r['llm_s']}s · TTS {r['tts_s']}s · 합계 {r['turn_s']}s", ""]
import statistics as st
for k in ("stt_s", "llm_s", "tts_s", "turn_s"):
    md.append(f"- {k} 중앙값 {st.median(x[k] for x in rows):.2f}s, 최대 {max(x[k] for x in rows):.2f}s")
(out / "demo_transcript.md").write_text("\n".join(md)); (out / "demo_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
print("\n".join(md[-4:]))
