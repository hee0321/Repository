"""안전 실드(행동 단위 규칙 마스킹): 학습 모델의 행동을 그대로 쓰되, 한국 교통 규칙·안전상 꼭 지켜야 하는
순간에만 기준 제어기(규칙 기반)의 행동으로 '덮어쓴다'.

- 가속(세로): 기준 제어기가 규칙 때문에 속도를 줄이는 중(빨간불 정지, 빨간불 우회전 일시정지, 보호구역·제한속도 감속,
  앞차 따라가기, 장애물 앞 정지, 보행자 옆 서행, 마주 오는 차 양보, 시야 거리 안 정지 가능 속도, 자갈길 서행)이거나
  지금 속도가 규칙상 목표 속도보다 빠르면 → 가속 = min(모델, 기준 제어기). 모델이 더 '조심스러운' 건 그대로 둔다.
- 조향(가로): 주변에 차·보행자·장애물이 가까이 있거나(앞지르기·비켜 가기·옆길 진입 차), 길 가장자리에 너무 붙으면
  → 조향 = 기준 제어기 (기준 제어기는 앞지르기 허용 구간·시야·마주 오는 차 조건일 때만 앞지른다).

점수표는 꼭 둘로 나눠 본다: '모델 단독(실드 없음)' = 학습 모델 실력, '모델+실드' = 실제 자율 모드에서 쓰는 조합.
실드가 바꾼 스텝 비율(override_share)도 같이 기록한다 (높을수록 규칙 기반이 많이 운전한 것).
"""
from __future__ import annotations

import numpy as np

from .controllers import PurePursuitController

FREE_NOTE = "규정 속도 유지"


class SafetyShield:
    def __init__(self, env, near_m: float = 45.0, edge_m: float = 0.3):
        self.env = env
        self.exp = PurePursuitController(env)
        self.near_m, self.edge_m = near_m, edge_m
        self.steps = self.long_over = self.lat_over = 0
        self.last = {}

    def obstacle_context(self):
        env = self.env
        for k in range(len(env.ob_s)):
            g = env._gap(env.ob_s[k])
            if -15.0 < g < self.near_m and env.visible(k) or -15.0 < g < 0.0:
                return True
        return False

    def filter(self, obs, a_model):
        env = self.env
        a_model = np.clip(np.asarray(a_model, dtype=np.float32).reshape(-1)[:2], -1, 1)
        a_exp = self.exp.act(obs)
        a = a_model.copy()
        note = self.exp.rule_note
        v_ref = getattr(self.exp, "last_v_ref", np.inf)
        long_o = lat_o = False
        if (note != FREE_NOTE or env.car.v > v_ref + 0.3) and a_exp[1] < a[1]:
            a[1] = a_exp[1]; long_o = True
        if self.obstacle_context() or env.edge_clearance() < self.edge_m or env.is_passing():
            if abs(a[0] - a_exp[0]) > 1e-6:
                a[0] = a_exp[0]; lat_o = True
        self.steps += 1
        self.long_over += long_o
        self.lat_over += lat_o
        self.last = {"rule": note, "long": long_o, "lat": lat_o, "a_model": a_model, "a_exp": a_exp}
        return a

    def stats(self):
        n = max(self.steps, 1)
        return {"steps": self.steps, "long_override_share": round(self.long_over / n, 3),
                "lat_override_share": round(self.lat_over / n, 3)}


def shielded_policy(model):
    """(env, obs) -> action. env 마다 실드 하나 (bench.make_policy 와 같은 모양)."""
    cache = {}

    def act(env, obs):
        sh = cache.get(id(env))
        if sh is None or sh.env is not env:
            sh = cache[id(env)] = SafetyShield(env)
        return sh.filter(obs, model.predict(obs, deterministic=True)[0])
    act.cache = cache
    return act
