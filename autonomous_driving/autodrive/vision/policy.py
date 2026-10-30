"""비전 정책: 작은 CNN (PyTorch, CPU). 입력 = 카메라 영상(지금 RGB + 0.2초 전 흑백) + 내비/계기판 벡터.
출력 = [조향 -1~1, 가속 -1~1] (환경 행동과 같음)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .nav import NAV_DIM

IMG_C, IMG_H, IMG_W = 4, 72, 128


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
        self.head = nn.Sequential(nn.Linear(1536 + 64, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2))

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
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m
