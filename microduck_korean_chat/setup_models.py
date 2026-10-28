"""모델 내려받기 (무료, 로그인 불필요). 총 약 1.8 GB."""
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
R = Path(__file__).resolve().parent / "models"
hf_hub_download("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf", local_dir=R / "llm")
snapshot_download("Systran/faster-whisper-small", local_dir=R / "whisper-small")
os.environ["SUPERTONIC_CACHE_DIR"] = str(R / "supertonic")
from supertonic import TTS
TTS(model="supertonic-2")  # 첫 실행 시 내려받음
print("완료:", R)
