"""로컬 LLM(Qwen2.5-1.5B-Instruct GGUF, llama.cpp CPU)으로 덕의 한국어 대답 만들기."""
from __future__ import annotations

import os
import random
import re
from pathlib import Path

from llama_cpp import Llama

from .intent import Command, strip_wake

ROOT = Path(__file__).resolve().parents[1]
LLM_PATH = Path(os.environ.get("DUCK_LLM", ROOT / "models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"))

SYSTEM = (
    "너는 '마이크로덕'이라는 키 25cm짜리 작은 오리 로봇이야. 걷기, 돌기, 앉기, 일어서기, 공 차기, "
    "부리로 바닥 물건 줍기, 앞으로 구르기를 할 수 있어. 성격은 밝고 호기심 많고 장난스러워. "
    "규칙: 친구한테 말하듯 반말로만 말해. 질문에 맞는 내용을 한두 문장으로 답해. "
    "이모지, 영어, 목록은 쓰지 마. "
    "사용자 말 끝에 [동작: ...]이 붙어 있으면 지금 그 동작을 하는 중이니, 그걸 한다고 신나게 말해."
)


# 반말·짧은 대답을 잡아 주는 예시 대화 (1.5B 모델은 예시가 없으면 존댓말로 흐름)
FEWSHOT = [
    {"role": "user", "content": "안녕!"},
    {"role": "assistant", "content": "안녕! 나는 마이크로덕이야. 같이 놀자, 꽥!"},
    {"role": "user", "content": "뒤로 가 봐 [동작: 뒤로 걷기]"},
    {"role": "assistant", "content": "알겠어, 뒤로 뒤뚱뒤뚱 걸어갈게!"},
    {"role": "user", "content": "너 지금 기분 어때?"},
    {"role": "assistant", "content": "완전 좋아! 오늘은 다리에 힘이 넘쳐서 막 걷고 싶어."},
    {"role": "user", "content": "공 차 [동작: 왼발 공 차기]"},
    {"role": "assistant", "content": "간다, 왼발 슛! 골인이면 좋겠다, 꽥!"},
    {"role": "user", "content": "넌 뭘 좋아해?"},
    {"role": "assistant", "content": "나는 공 차는 거랑 바닥에 있는 걸 부리로 줍는 게 제일 좋아!"},
]


# 명령이 있는 턴은 고정 대답(빠르고 틀리지 않음). 1.5B 모델은 가끔 '싫어'처럼 동작과 반대로 말해서.
ACKS = {
    "앞으로 걷기": ["알았어, 앞으로 뒤뚱뒤뚱 간다!", "출발! 앞으로 걸어갈게, 꽥!"],
    "뒤로 걷기": ["뒤로 살금살금 갈게!", "알았어, 뒤로 물러날게."],
    "왼쪽으로 돌기": ["왼쪽으로 빙글 돌게!", "좋아, 왼쪽으로 돈다!"],
    "오른쪽으로 돌기": ["오른쪽으로 빙글 돌게!", "좋아, 오른쪽으로 돈다!"],
    "제자리 돌기(왼쪽)": ["빙글빙글 돌아 볼게!", "한 바퀴 돌아 볼게, 꽥!"],
    "왼쪽 옆걸음": ["왼쪽으로 옆걸음!"], "오른쪽 옆걸음": ["오른쪽으로 옆걸음!"],
    "앉기": ["알았어, 털썩 앉을게.", "잠깐 앉아서 쉴게!"],
    "일어서기": ["영차, 일어날게!", "알았어, 다시 일어선다!"],
    "왼발 공 차기": ["간다, 왼발 슛!", "왼발로 뻥 찰게, 꽥!"],
    "오른발 공 차기": ["간다, 오른발 슛!", "오른발로 뻥 찰게, 꽥!"],
    "바닥 물건 줍기": ["부리로 콕 집어 볼게!"], "구르기": ["데굴데굴 구른다!"],
    "멈추기": ["멈췄어!", "알았어, 딱 멈출게."], "꽥 소리": ["꽥꽥!"],
}


class Brain:
    def __init__(self, n_threads: int | None = None, max_history: int = 6):
        self.llm = Llama(model_path=str(LLM_PATH), n_ctx=2048, n_threads=n_threads or os.cpu_count(),
                         verbose=False)
        self.history: list[dict] = []
        self.max_history = max_history

    def reply(self, user_text: str, cmd: Command | None = None, max_tokens: int = 80,
              llm_for_commands: bool = False) -> str:
        if cmd is not None and not llm_for_commands and cmd.action in ACKS:
            text = random.choice(ACKS[cmd.action])
            self.history += [{"role": "user", "content": strip_wake(user_text) + f" [동작: {cmd.action}]"},
                             {"role": "assistant", "content": text}]
            return text
        content = strip_wake(user_text) + (f" [동작: {cmd.action}]" if cmd else "")
        msgs = [{"role": "system", "content": SYSTEM}, *FEWSHOT, *self.history[-self.max_history:],
                {"role": "user", "content": content}]
        out = self.llm.create_chat_completion(messages=msgs, max_tokens=max_tokens, temperature=0.6,
                                              top_p=0.9, repeat_penalty=1.1)
        text = clean(out["choices"][0]["message"]["content"])
        if len(text.replace("꽥", "").strip(" !?.~")) < 2:  # '꽥!'만 하는 퇴화 응답이면 한 번 더
            out = self.llm.create_chat_completion(messages=msgs, max_tokens=max_tokens, temperature=0.9,
                                                  top_p=0.95, repeat_penalty=1.1)
            text = clean(out["choices"][0]["message"]["content"])
        self.history += [{"role": "user", "content": content}, {"role": "assistant", "content": text}]
        return text


def clean(text: str) -> str:
    """TTS가 읽기 좋게: 이모지·괄호·한자/영문 잡음 제거, 2문장으로 자르기."""
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", "", text)
    text = re.sub(r"[^\w\s가-힣.,!?~'\"-]", "", text)
    text = re.sub(r"[\u4e00-\u9fff\u3040-\u30ff]+", "", text)
    text = re.sub(r"[A-Za-z]+[,.!]?\s*", "", text)  # 섞여 나온 영단어 (TTS가 한국어 모드에서 못 읽음)  # 소형 모델이 가끔 섞는 한자/가나
    sents = re.split(r"(?<=[.!?~])\s+", text.strip())
    return " ".join(sents[:2]).strip() or "꽥?"
