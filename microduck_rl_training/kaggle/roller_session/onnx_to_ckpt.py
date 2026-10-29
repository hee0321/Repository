"""Pollen 공개 roller.onnx → rsl_rl 체크포인트(.pt) 변환 (warm start 용, 무료·CPU 몇 초).

roller.onnx = Sub(mean) → Div(std) → MLP 61→512→256→128→14 (ELU). 이 저장소 롤러 작업의 actor 와 같은 모양.
  - actor 가중치·관측 정규화(mean/std)를 ONNX 에서 복사
  - manifest: roller 정책은 action_scale 0.8, 학습 환경(Mjlab-Velocity-Flat-MicroDuck-Rollers)은 1.0
    → 마지막 층(mlp.6) 가중치·편향 × 0.8 로 접어 넣어 '환경 행동 = 0.8 × ONNX 출력' 을 그대로 재현
  - 행동 잡음 std = --std (기본 0.3; 처음부터 학습은 1.0) — 이미 타는 정책을 흔들지 않게 작게
  - critic(가치 함수)은 템플릿(1회 학습한 새 모델) 그대로 = 처음부터 배움. optimizer 상태도 템플릿 것.
  - 정규화 count 를 크게(1e6) 해서 학습 중 mean/std 가 천천히만 움직이게
템플릿 = 같은 작업으로 `--agent.max-iterations 1` 돌려 나온 model_0.pt (구조·optimizer 모양을 맞추려고).

  python onnx_to_ckpt.py roller.onnx template_model_0.pt out_roller_init.pt [--check]
"""
import argparse
import numpy as np
import onnx
from onnx import numpy_helper
import torch

ap = argparse.ArgumentParser()
ap.add_argument("onnx_file"); ap.add_argument("template"); ap.add_argument("out")
ap.add_argument("--action-scale", type=float, default=0.8)
ap.add_argument("--std", type=float, default=0.3)
ap.add_argument("--check", action="store_true", help="onnxruntime 출력 × 0.8 과 torch actor 출력 비교")
a = ap.parse_args()

m = onnx.load(a.onnx_file)
W = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
div = [n for n in m.graph.node if n.op_type == "Div"][0]
std_name = [x for x in div.input if x in W][0]
mean, std = W["obs_normalizer._mean"].astype(np.float32), W[std_name].astype(np.float32)

ck = torch.load(a.template, map_location="cpu", weights_only=False)
sd = ck["actor_state_dict"]
for k in ("mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias", "mlp.4.weight", "mlp.4.bias", "mlp.6.weight", "mlp.6.bias"):
    v = torch.from_numpy(W[k].copy())
    assert sd[k].shape == v.shape, (k, sd[k].shape, v.shape)
    sd[k] = v * a.action_scale if k.startswith("mlp.6") else v
sd["obs_normalizer._mean"] = torch.from_numpy(mean).reshape(sd["obs_normalizer._mean"].shape)
sd["obs_normalizer._std"] = torch.from_numpy(std).reshape(sd["obs_normalizer._std"].shape)
sd["obs_normalizer._var"] = sd["obs_normalizer._std"] ** 2
sd["obs_normalizer.count"] = torch.tensor(1e6, dtype=sd["obs_normalizer.count"].dtype)
sd["distribution.std_param"] = torch.full_like(sd["distribution.std_param"], a.std)
ck["actor_state_dict"] = sd
ck["iter"] = 0
ck.setdefault("infos", {})["converted_from"] = f"{a.onnx_file} (action_scale {a.action_scale} folded into mlp.6)"
torch.save(ck, a.out)
print("saved", a.out)

if a.check:
    import onnxruntime as ort
    s = ort.InferenceSession(a.onnx_file, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).normal(size=(256, 61)).astype(np.float32) * std + mean
    y_onnx = np.concatenate([s.run(None, {s.get_inputs()[0].name: x[i:i + 1]})[0] for i in range(len(x))]) * a.action_scale
    h = (torch.from_numpy(x) - sd["obs_normalizer._mean"]) / sd["obs_normalizer._std"]
    for i in (0, 2, 4):
        h = torch.nn.functional.elu(h @ sd[f"mlp.{i}.weight"].T + sd[f"mlp.{i}.bias"])
    y = (h @ sd["mlp.6.weight"].T + sd["mlp.6.bias"]).numpy()
    print("max |torch - 0.8*onnx| =", float(np.abs(y - y_onnx).max()), "(action range", float(np.abs(y_onnx).max()), ")")
