"""STT(faster-whisper small, int8)와 TTS(Supertonic 2, 한국어) — 둘 다 CPU, 오프라인."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SUPERTONIC_CACHE_DIR", str(ROOT / "models/supertonic"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 모델 받은 뒤로는 인터넷 안 씀


PROMPT = "마이크로덕과의 대화."


class STT:
    def __init__(self, model_dir: str | None = None, threads: int | None = None, vad: bool = False):
        # vad=False: 짧고 깨끗한 클립에선 Silero VAD가 첫 단어를 잘라먹음(시험에서 확인). 실제 마이크
        # 상시 청취에선 앞단에서 발화 구간을 자른 뒤 넘기거나 vad=True 로.
        self.vad = vad
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_dir or str(ROOT / "models/whisper-small"), device="cpu",
                                  compute_type="int8", cpu_threads=threads or os.cpu_count())

    def transcribe(self, wav_path: str) -> str:
        import numpy as np, soundfile as sf, soxr
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != 16000:
            audio = soxr.resample(audio, sr, 16000)
        pad = np.zeros(4000, np.float32)  # 앞뒤 0.25초 무음: 첫 음절 잘림 방지
        audio = np.concatenate([pad, audio.astype(np.float32), pad])
        segs, _ = self.model.transcribe(audio, language="ko", beam_size=1, vad_filter=self.vad,
                                        initial_prompt=PROMPT, temperature=0.0,
                                        condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segs).strip()
        return "" if text.strip(" .") == PROMPT.strip(" .") else text  # 무음일 때 프롬프트를 그대로 뱉는 환각 제거


class TTS:
    def __init__(self, voice: str = "F1"):
        from supertonic import TTS as _TTS
        self.tts = _TTS(model="supertonic-2")
        self.style = self.tts.get_voice_style(voice)
        self.sample_rate = self.tts.sample_rate

    def say(self, text: str, out_path: str, speed: float = 1.1) -> float:
        wav, dur = self.tts.synthesize(text, self.style, lang="ko", speed=speed, total_steps=8)
        self.tts.save_audio(wav, out_path)
        return float(dur[0])
