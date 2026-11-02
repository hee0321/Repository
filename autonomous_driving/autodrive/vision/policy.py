"""비전 정책: 작은 CNN (PyTorch, CPU). 입력 = 카메라 영상(지금 RGB + 0.2초 전 흑백) + 내비/계기판 벡터.
출력 = [조향 -1~1, 가속 -1~1, 방향지시등 -1(오른쪽)~+1(왼쪽)]. 앞 둘은 환경 행동, 셋째는 운전자가 레버를 켜는 것."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .nav import NAV_DIM

IMG_C, IMG_H, IMG_W = 4, 72, 128
N_OUT = 3


class VisionPolicy(nn.Module):
    def __init__(self, nav_dim=NAV_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(IMG_C, 24, 5, stride=2, padding=2), nn.ReLU(),     # 36×64
            nn.Conv2d(24, 36, 3, stride=2, padding=1), nn.ReLU(),        # 18×32
            nn.Conv2d(36, 48, 3, stride=2, padding=1), nn.ReLU(),        # 9×16
            nn.Conv2d(48, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 8)), nn.Flatten())                   # 64×3×8 = 1536
        self.nav = nn.Sequential(nn.Linear(nav_dim, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(1536 + 64, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, N_OUT))

    def forward(self, img_u8, nav):
        x = img_u8.float() / 255.0 - 0.5
        z = torch.cat([self.conv(x), self.nav(nav)], dim=1)
        return torch.tanh(self.head(z))


def stack_frames(cur_rgb: np.ndarray, prev_rgb: np.ndarray | None) -> np.ndarray:
    """(H,W,3) uint8 두 장 → (4,H,W) uint8: 지금 RGB + 이전 흑백(움직임 단서)."""
    prev = cur_rgb if prev_rgb is None else prev_rgb
    gray = (0.299 * prev[..., 0] + 0.587 * prev[..., 1] + 0.114 * prev[..., 2]).astype(np.uint8)
    return np.concatenate([cur_rgb.transpose(2, 0, 1), gray[None]], axis=0)


def load_policy(path):
    m = VisionPolicy()
    sd = torch.load(path, map_location="cpu")
    w = sd.get("nav.0.weight")
    if w is not None and w.shape[1] < NAV_DIM:   # 예전(28개) 가중치: 새 입력 칸은 0 으로 이어 붙임 (처음엔 무시)
        sd["nav.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], NAV_DIM - w.shape[1])], dim=1)
    hw, hb = sd.get("head.4.weight"), sd.get("head.4.bias")
    if hw is not None and hw.shape[0] < N_OUT:     # 예전(출력 2개) 가중치: 지시등 출력은 0 에서 시작
        sd["head.4.weight"] = torch.cat([hw, torch.zeros(N_OUT - hw.shape[0], hw.shape[1])], dim=0)
        sd["head.4.bias"] = torch.cat([hb, torch.zeros(N_OUT - hb.shape[0])])
    m.load_state_dict(sd)
    m.eval()
    return m
