"""인식·추적·예측을 카메라 영상과 옆 평면도에 그리는 도구. 예측 = 색 상자·점선, 정답 = 흰 점선(구별)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..vision.camera import LABELS

PAL = np.array([[0, 0, 0], [90, 90, 90], [255, 255, 255], [0, 120, 255], [255, 0, 200], [255, 140, 0],
                [120, 120, 0], [255, 40, 30], [255, 200, 0], [40, 255, 120], [100, 60, 30]], np.uint8)
COL = dict(car=(0, 160, 255), ped=(255, 0, 200), object=(255, 140, 0), signal=(255, 220, 0))


def overlay(img, cls_map, dets, lamps, alpha=0.35):
    base = Image.fromarray(img).convert("RGBA")
    m = Image.fromarray(PAL[cls_map]).convert("RGBA")
    m.putalpha(int(255 * alpha))
    out = Image.alpha_composite(base, m).convert("RGB")
    dr = ImageDraw.Draw(out)
    for d in dets:
        col = COL.get(d.cls, (255, 255, 0))
        u0, v0, u1, v1 = d.box
        dr.rectangle((u0, v0, u1, v1), outline=col, width=2)
        dr.text((u0, max(0, v0 - 10)), f"{d.cls} {d.dist:.0f}±{d.sigma:.0f}m", fill=col)
    for l in lamps:
        col = {"red": (255, 40, 30), "yellow": (255, 200, 0), "green": (40, 255, 120)}[l.extra["state"]]
        u0, v0, u1, v1 = l.box
        dr.ellipse((u0 - 2, v0 - 2, u1 + 2, v1 + 2), outline=col, width=2)
        dr.text((u0, v0 - 10), f"{l.extra['state']} {l.fwd:.0f}m", fill=col)
    return out


def birdseye(frame, scale=6.0, ahead=50.0, side=12.0):
    """평면도 한 장 (예측 = 색, 정답 = 흰 점선). scale px/m."""
    H = int(ahead * scale) + 40
    W = int(2 * side * scale) + 40
    im = Image.new("RGB", (W, H), (30, 32, 36))
    dr = ImageDraw.Draw(im)
    ox, oy = W // 2, H - 20

    def xy(fwd, left):
        return ox - left * scale, oy - fwd * scale

    # 격자
    for d in range(0, int(ahead) + 1, 10):
        y = oy - d * scale
        dr.line((0, y, W, y), fill=(50, 55, 60))
        dr.text((2, y - 8), f"{d}m", fill=(120, 120, 120))
    # 내 차
    dr.rectangle((ox - 0.9 * scale, oy - 3.25 * scale, ox + 0.9 * scale, oy + 0.75 * scale), outline=(220, 220, 220), width=2)
    # 차선·도로 경계
    marks, lb, rb = frame["lanes"]
    for pts, col in ((marks, (200, 200, 200)), (lb, (180, 180, 80)), (rb, (180, 180, 80))):
        for p in pts:
            if 0 < p[0] < ahead and abs(p[1]) < side:
                u, v = xy(p[0], p[1])
                dr.ellipse((u - 1, v - 1, u + 1, v + 1), fill=col)
    # 예측 경로
    for p in frame["path"]:
        u, v = xy(p[1], p[2])
        if 0 < p[1] < ahead:
            dr.ellipse((u - 2, v - 2, u + 2, v + 2), fill=(80, 220, 80))
    # 추적 물체 + 예측 (색)
    for tid, cls, pe, ve, sig, conf, miss, traj in frame["tracks"]:
        if pe[0] < -2 or pe[0] > ahead or abs(pe[1]) > side:
            continue
        col = COL.get(cls, (255, 255, 0))
        u, v = xy(pe[0], pe[1])
        r = (2.0 if cls == "car" else 0.6) * scale
        if conf:
            dr.ellipse((u - r, v - r, u + r, v + r), outline=col, width=2)
        else:
            dr.ellipse((u - r, v - r, u + r, v + r), outline=col)
        # 2σ 불확실성
        rr = (sig * 2 + 0.5) * scale
        dr.ellipse((u - rr, v - rr, u + rr, v + rr), outline=(*col, ), width=1)
        dr.text((u + 3, v - 8), f"#{tid}", fill=col)
        for t, p, s in traj:
            uu, vv = xy(p[0], p[1])
            if 0 < p[0] < ahead:
                dr.ellipse((uu - 1, vv - 1, uu + 1, vv + 1), fill=col)
                if abs(t - 2.0) < 0.2 or abs(t - 3.0) < 0.2:
                    rs = (s * 2 + 0.5) * scale
                    dr.ellipse((uu - rs, vv - rs, uu + rs, vv + rs), outline=col)
    # 정답 (흰 점선 상자 — 예측과 구별)
    for k, cls, fx, ly, vel, npx in frame["gt"]:
        if 0 < fx < ahead and abs(ly) < side:
            u, v = xy(fx, ly)
            r = (2.0 if cls == "car" else 0.6) * scale
            for d in range(-int(r), int(r) + 1, 3):
                dr.point((u + d, v - r), fill=(255, 255, 255))
                dr.point((u + d, v + r), fill=(255, 255, 255))
                dr.point((u - r, v + d), fill=(255, 255, 255))
                dr.point((u + r, v + d), fill=(255, 255, 255))
    # 신호
    for l in frame["lamps"]:
        col = {"red": (255, 40, 30), "yellow": (255, 200, 0), "green": (40, 255, 120)}[l.extra["state"]]
        stop = l.fwd + 2.45 - 22.0
        if 0 < stop < ahead:
            y = oy - stop * scale
            dr.line((ox - 4 * scale, y, ox + 4 * scale, y), fill=col, width=2)
            dr.text((ox + 4 * scale, y - 8), f"정지선({l.extra['state']})", fill=col)
    note = frame.get("reason") or "정상"
    dr.text((4, 4), f"v={frame['v']*3.6:.0f}km/h  빈공간={frame.get('free',0):.0f}m  {note}", fill=(230, 230, 230))
    dr.text((4, H - 14), "색=예측(인식) · 흰 점선=시뮬 정답(측정용)", fill=(160, 160, 160))
    return im


def compose(frame):
    left = overlay(frame["img"], frame["cls"], frame["dets"], frame["lamps"])
    right = birdseye(frame)
    H = max(left.height, right.height)
    canvas = Image.new("RGB", (left.width + right.width, H), (20, 20, 20))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas
