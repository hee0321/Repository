"""픽셀 분할 CNN (작은 U-Net, CPU). 클래스 = camera.LABELS (도로, 차선, 차, 사람, 물체, 신호등, 빨강/노랑/초록 등, 건물).
합성 장면 시제품: 시뮬레이터 렌더러가 그린 영상으로만 학습해서, 실제 영상에서 된다는 뜻이 아니에요."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vision.camera import LABELS

N_CLS = len(LABELS)
PERC_W, PERC_H = 256, 144


def _cb(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SegNet(nn.Module):
    def __init__(self, n_cls=N_CLS):
        super().__init__()
        self.e1 = nn.Sequential(_cb(3, 16), _cb(16, 16))          # 144×256
        self.e2 = nn.Sequential(_cb(16, 32, 2), _cb(32, 32))      # 72×128
        self.e3 = nn.Sequential(_cb(32, 48, 2), _cb(48, 48))      # 36×64
        self.e4 = nn.Sequential(_cb(48, 64, 2), _cb(64, 64))      # 18×32
        self.d3 = _cb(64 + 48, 48)
        self.d2 = _cb(48 + 32, 32)
        self.d1 = _cb(32 + 16, 16)
        self.out = nn.Conv2d(16, n_cls, 1)

    def forward(self, x_u8):
        x = x_u8.float() / 255.0 - 0.5
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        d = self.d3(torch.cat([F.interpolate(e4, size=e3.shape[2:], mode="nearest"), e3], 1))
        d = self.d2(torch.cat([F.interpolate(d, size=e2.shape[2:], mode="nearest"), e2], 1))
        d = self.d1(torch.cat([F.interpolate(d, size=e1.shape[2:], mode="nearest"), e1], 1))
        return self.out(d)


def load_seg(path):
    m = SegNet()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


@torch.no_grad()
def segment(model, img):
    """(H,W,3) uint8 → (클래스 (H,W) uint8, 확신도 (H,W) float)."""
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None]
    p = torch.softmax(model(x), 1)[0]
    conf, cls = p.max(0)
    return cls.numpy().astype(np.uint8), conf.numpy()
