"""머리 카메라 RGB에서 빨간 공을 찾는 규칙. 학습된 시각 모델이 아니다.

numpy HSV 임계값과 가장 큰 연결 영역만 쓴다. 시뮬레이터의 공 좌표, 세그멘테이션 id,
거리·방향은 받지 않는다. 주황(발·턱판, 색상 약 20–40°)은 빨간 색상 창(약 ±12°) 밖으로 둔다.

방향은 픽셀과 카메라 자체 자세(설치 방향)로만 환산한다. 공의 시뮬레이터 좌표는 쓰지 않는다.
"""
from __future__ import annotations

import math

import numpy as np

# 색상은 0–1 바퀴. 빨강은 0 근처. 12/360 ≈ 0.033.
HUE_HALF = 12.0 / 360.0
SAT_MIN = 0.45
VAL_MIN = 0.22
# 연결 영역이 이보다 작으면 잡음. 320×180 에서 1.4 m 거리의 지름 7 cm 공은 대략 이 이상.
MIN_PIXELS = 18
MIN_FILL = 0.35


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """rgb uint8 HWC → float HSV, H in [0, 1)."""
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = np.max(x, axis=-1)
    mn = np.min(x, axis=-1)
    df = mx - mn
    h = np.zeros_like(mx)
    ok = df > 1e-6
    rmax = ok & (mx == r)
    gmax = ok & (mx == g)
    bmax = ok & (mx == b)
    h[rmax] = np.mod((g[rmax] - b[rmax]) / df[rmax], 6.0) / 6.0
    h[gmax] = ((b[gmax] - r[gmax]) / df[gmax] + 2.0) / 6.0
    h[bmax] = ((r[bmax] - g[bmax]) / df[bmax] + 4.0) / 6.0
    s = np.zeros_like(mx)
    nz = mx > 1e-6
    s[nz] = df[nz] / mx[nz]
    return np.stack([h, s, mx], axis=-1)


def red_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = rgb_to_hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_red = (h <= HUE_HALF) | (h >= 1.0 - HUE_HALF)
    return hue_red & (s >= SAT_MIN) & (v >= VAL_MIN)


def _components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-연결 라벨. scipy 가 있으면 그걸 쓰고, 없으면 작은 마스크용 스캔."""
    try:
        from scipy import ndimage
        labels, n = ndimage.label(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        return labels, int(n)
    except Exception:
        h, w = mask.shape
        labels = np.zeros((h, w), np.int32)
        n = 0
        parent = [0]

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for y in range(h):
            for x in range(w):
                if not mask[y, x]:
                    continue
                left = labels[y, x - 1] if x else 0
                up = labels[y - 1, x] if y else 0
                if left and up:
                    a, b = find(left), find(up)
                    labels[y, x] = a
                    if a != b:
                        parent[b] = a
                elif left or up:
                    labels[y, x] = left or up
                else:
                    n += 1
                    parent.append(n)
                    labels[y, x] = n
        if n == 0:
            return labels, 0
        remap = {}
        out = np.zeros_like(labels)
        k = 0
        flat = labels.ravel()
        dest = out.ravel()
        for i, v in enumerate(flat):
            if v == 0:
                continue
            r = find(int(v))
            if r not in remap:
                k += 1
                remap[r] = k
            dest[i] = remap[r]
        return out, k


def detect_red(rgb: np.ndarray, min_pixels: int = MIN_PIXELS) -> dict | None:
    """가장 큰 빨간 덩어리. 없으면 None.

    반환은 이미지 좌표뿐: cx, cy (픽셀), ex/ey 는 화면 중심 기준 −1..1
    (오른쪽·아래가 양수), area, bbox, height, fill.
    """
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"RGB HWC expected, got {rgb.shape}")
    mask = red_mask(rgb)
    labels, n = _components(mask)
    if n == 0:
        return None
    counts = np.bincount(labels.ravel(), minlength=n + 1)
    counts[0] = 0
    lab = int(np.argmax(counts))
    area = int(counts[lab])
    if area < min_pixels:
        return None
    ys, xs = np.nonzero(labels == lab)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    fill = area / float(bw * bh)
    if fill < MIN_FILL:
        return None
    h, w = mask.shape
    cx = float(xs.mean())
    cy = float(ys.mean())
    return {
        "cx": cx,
        "cy": cy,
        "ex": (cx - (w - 1) / 2.0) / (w / 2.0),
        "ey": (cy - (h - 1) / 2.0) / (h / 2.0),
        "area": area,
        "bbox": (x0, y0, x1, y1),
        "height": bh,
        "width": bw,
        "fill": fill,
        "mask_fraction": area / float(h * w),
        "diam": 2.0 * math.sqrt(area / math.pi),
    }


def pixel_bearing(cx: float, cy: float, cam_xmat, fovy_deg: float, width: int, height: int, yaw: float) -> float:
    """빨간 픽셀이 몸통 기준 어느 쪽인지(rad, 왼쪽이 양수).

    핀홀: 픽셀 → 카메라 좌표계 광선. 카메라 행렬은 머리 관절로 정해지는 카메라 자신의 방향이다.
    R @ ray 가 2026-11-07 측정에서 정면 0.2°, 오른쪽 90°에서 4° 안으로 맞았다.
    """
    fovy = math.radians(float(fovy_deg))
    fy = (height / 2.0) / math.tan(fovy / 2.0)
    fx = fy
    x = (cx - (width - 1) / 2.0) / fx
    y = -((cy - (height - 1) / 2.0) / fy)
    ray_c = np.array([x, y, -1.0], dtype=float)
    ray_c /= np.linalg.norm(ray_c)
    rw = np.asarray(cam_xmat, dtype=float).reshape(3, 3) @ ray_c
    c, s = math.cos(yaw), math.sin(yaw)
    bx = c * rw[0] + s * rw[1]
    by = -s * rw[0] + c * rw[1]
    return math.atan2(by, bx)
