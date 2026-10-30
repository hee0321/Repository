"""비전 정책 시연 영상: 왼쪽 큰 화면 = 정책이 실제로 보는 앞 카메라(128×72 를 확대), 오른쪽 = 위성 미니맵(경로·차·
주변 차) + 카메라 장착 위치 그림 + 상태. 모든 프레임에 출처 표시(브이월드 공공누리 제1유형, OSM ODbL).

사용: python render_vision_video.py --scenario gangnam_traffic --t0 0 --dur 30 --out /workspace/vision_gangnam.mp4
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from autodrive.vision.ground import ATTRIBUTION

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
W, H = 1280, 720
CAM_W, CAM_H = 960, 540


def font(sz):
    try:
        return ImageFont.truetype(FONT, sz, index=1)
    except OSError:
        return ImageFont.load_default()


F14, F18, F24, F40 = font(14), font(18), font(24), font(40)
SIG_COL = {"red": (255, 60, 50), "yellow": (255, 200, 0), "green": (60, 230, 120)}


def minimap(env, ground, fr, others, size=320, span=120.0):
    """차 중심 위성 미니맵 (북쪽 위). span m 폭."""
    res = span / size
    xs = fr["x"] + (np.arange(size) - size / 2 + 0.5) * res
    ys = fr["y"] - (np.arange(size) - size / 2 + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    if ground is not None:
        rgb, ok = ground.sample(X.ravel(), Y.ravel())
        rgb = np.where(ok[:, None], rgb, 60.0).reshape(size, size, 3)
    else:
        rgb = np.full((size, size, 3), (70, 100, 60), np.float32)
    im = Image.fromarray(rgb.astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")

    def px(P):
        return [((x - fr["x"]) / res + size / 2, (fr["y"] - y) / res + size / 2) for x, y in P]
    tr = env.track
    i0 = tr._idx(fr["s"])
    idx = np.arange(max(0, i0 - 120), min(tr.m, i0 + 200)) if not tr.closed else np.arange(i0 - 120, i0 + 200) % tr.m
    d.line(px(tr.center[idx]), fill=(0, 220, 255, 200), width=4)
    for (ox, oy, kind) in others:
        c = (255, 150, 40, 255) if kind == "ped" else (255, 255, 255, 255)
        p = px([(ox, oy)])[0]
        d.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), fill=c, outline=(0, 0, 0, 255))
    # 내 차 (화살표) + 카메라 시야 부채꼴
    c, s = np.cos(fr["yaw"]), np.sin(fr["yaw"])
    cx, cy = fr["x"] + 2.45 * c, fr["y"] + 2.45 * s
    half = np.radians(27.5)
    fov = [(cx, cy)] + [(cx + 45 * np.cos(fr["yaw"] + a), cy + 45 * np.sin(fr["yaw"] + a)) for a in np.linspace(-half, half, 9)]
    d.polygon(px(fov), fill=(255, 255, 0, 50))
    k = 2.0  # 보이게 2배 크게 그림
    body = [(fr["x"] + k * (dx * c - dy * s), fr["y"] + k * (dx * s + dy * c)) for dx, dy in ((-0.75, -0.9), (3.25, -0.9), (3.25, 0.9), (-0.75, 0.9))]
    d.polygon(px(body), fill=(230, 40, 40, 255), outline=(255, 255, 255, 255))
    return im


def mount_icon(size=(320, 150)):
    """카메라 장착 위치 그림: 위에서 본 차 + 옆에서 본 차, 빨간 점 = 카메라."""
    im = Image.new("RGB", size, (25, 27, 32))
    d = ImageDraw.Draw(im)
    d.text((8, 4), "카메라 장착 (테슬라식)", font=F14, fill=(230, 230, 230))
    # 위에서 (길이 4.0 m → 120 px, 폭 1.8 m → 54 px)
    x0, y0, k = 18, 34, 30
    d.rounded_rectangle((x0, y0, x0 + 4.0 * k, y0 + 1.8 * k), 10, outline=(200, 200, 200), width=2)
    d.line((x0 + 2.35 * k, y0 + 4, x0 + 2.35 * k, y0 + 1.8 * k - 4), fill=(120, 170, 220), width=3)  # 앞 유리
    cam = (x0 + (0.75 + 2.45) * k, y0 + 0.9 * k)
    d.ellipse((cam[0] - 5, cam[1] - 5, cam[0] + 5, cam[1] + 5), fill=(255, 40, 40))
    d.text((x0, y0 + 1.8 * k + 4), "위: 중심선(좌우 0 m)", font=F14, fill=(200, 200, 200))
    # 옆에서
    x1, yb = 170, 110
    d.polygon([(x1, yb), (x1 + 4.0 * 34, yb), (x1 + 4.0 * 34, yb - 0.75 * 34), (x1 + 3.1 * 34, yb - 0.85 * 34),
               (x1 + 2.3 * 34, yb - 1.42 * 34), (x1 + 1.0 * 34, yb - 1.45 * 34), (x1 + 0.3 * 34, yb - 0.9 * 34), (x1, yb - 0.8 * 34)],
              outline=(200, 200, 200))
    camx = x1 + (4.0 - 0.8) * 34 - 0.55 * 34
    camy = yb - 1.35 * 34
    d.ellipse((camx - 5, camy - 5, camx + 5, camy + 5), fill=(255, 40, 40))
    d.line((camx, camy, camx + 40, camy + 40 * np.tan(np.radians(3))), fill=(255, 220, 0), width=1)
    d.text((x1 - 4, yb + 4), "옆: 높이 1.35 m", font=font(12), fill=(200, 200, 200))
    d.text((8, size[1] - 20), "앞 범퍼 0.8 m 뒤 · 3° 숙임 · 수평 화각 55°", font=font(12), fill=(200, 200, 200))
    return im


def compose(env, ground, fr, title, mount, others, stats):
    canvas = Image.new("RGB", (W, H), (15, 16, 20))
    cam = Image.fromarray(fr["img"]).resize((CAM_W, CAM_H), Image.BILINEAR)
    canvas.paste(cam, (0, 40))
    d = ImageDraw.Draw(canvas)
    d.text((10, 6), title, font=F24, fill=(240, 240, 240))
    d.text((12, 46), "정책이 보는 화면: 앞 카메라 1대 (128×72, 확대)", font=F18, fill=(255, 255, 255))
    # HUD: 속도·제한속도 (계기판/내비 정보, 정책 입력과 같음)
    kmh = fr["v"] * 3.6
    d.rounded_rectangle((16, 440, 200, 568), 14, fill=(0, 0, 0))
    d.text((30, 450), f"{kmh:4.0f}", font=F40, fill=(255, 255, 255))
    d.text((140, 470), "km/h", font=F18, fill=(200, 200, 200))
    lim = fr["limit"] * 3.6
    d.ellipse((120, 505, 175, 560), fill=(255, 255, 255), outline=(220, 20, 20), width=6)
    d.text((133 if lim < 100 else 126, 518), f"{lim:.0f}", font=F18, fill=(0, 0, 0))
    st, ac = fr["a"]
    d.text((30, 520), f"핸들 {st:+.2f}", font=F14, fill=(200, 200, 200))
    d.text((30, 540), f"{'가속' if ac >= 0 else '감속'} {abs(ac):.2f}", font=F14, fill=(120, 230, 120) if ac >= 0 else (255, 120, 100))
    # 내비 안내 화살표 (정책 입력 중 경로 방향)
    turn = float(fr["nav"][6])  # 25 m 앞 경로 방향(÷0.5 rad)
    ax, ay = 860, 470
    ang = -turn * 0.5
    d.rounded_rectangle((790, 430, 950, 568), 14, fill=(0, 0, 0))
    tip = (ax + 40 * np.sin(-ang), ay - 40 * np.cos(-ang) + 20)
    d.line((ax, ay + 50, ax, ay + 20, tip[0], tip[1]), fill=(80, 200, 255), width=8)
    d.text((800, 436), "내비 경로", font=F14, fill=(200, 200, 200))
    sg = fr["sig"]
    if sg is not None and sg[0] < 120:
        d.text((800, 540), f"신호 교차로 {sg[0]:.0f} m", font=F14, fill=(220, 220, 220))
    # 오른쪽 패널
    mm = minimap(env, ground, fr, others)
    canvas.paste(mm, (W - 320, 40))
    d.text((W - 314, 44), "위성 미니맵 (북쪽 위, 120 m)", font=F14, fill=(255, 255, 255))
    canvas.paste(mount, (W - 320, 364))
    y = 594
    for line in stats:
        d.text((16, y), line, font=F18, fill=(220, 220, 220))
        y += 26
    if sg is not None and sg[0] < 120:
        d.ellipse((W - 312, 530, W - 294, 548), fill=SIG_COL[sg[1]])
        d.text((W - 288, 530), f"(참고용 정답) 다음 신호 {sg[1]}", font=F14, fill=(180, 180, 180))
        d.text((W - 312, 552), "정책은 신호 색을 카메라로만 봄", font=F14, fill=(180, 180, 180))
    d.rectangle((0, H - 36, W, H), fill=(0, 0, 0))
    d.text((10, H - 30), ATTRIBUTION + "  |  시뮬레이션 (실제 차량 아님)", font=F14, fill=(230, 230, 230))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="gangnam_traffic")
    ap.add_argument("--policy", default="weights/vision_policy.pt")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=30.0)
    ap.add_argument("--out", default="/workspace/vision_gangnam.mp4")
    ap.add_argument("--still", default=None)
    ap.add_argument("--still-t", type=float, default=None)
    a = ap.parse_args()
    import imageio_ffmpeg
    import torch
    torch.set_num_threads(4)
    from autodrive.bench import ALL_BENCH, REF_SPEED, scenario_score
    from autodrive.scenarios import make_env
    from autodrive.vision.camera import data_key_for
    from autodrive.vision.drive import drive_episode
    from autodrive.vision.ground import ground_for
    from autodrive.vision.policy import load_policy
    env = make_env(ALL_BENCH[a.scenario], seed=0)
    # 기록: 매 결정(10 Hz) 마다 주변 물체 위치도 같이 저장
    others_log = []
    orig_step = env.step

    def step(act):
        out = orig_step(act)
        tr = env.track
        others_log.append([(*(tr.point_at(env.ob_s[k]) + tr.normal_at(env.ob_s[k]) * env.ob_d[k]), env.ob_kind[k])
                           for k in range(len(env.ob_s)) if -20 < env._gap(env.ob_s[k]) < 80])
        return out
    env.step = step
    res, _, frames = drive_episode(env, load_policy(a.policy), seed=0, record=True)
    res["score"] = scenario_score(res, REF_SPEED.get(a.scenario, 30))
    print("episode", {k: res[k] for k in ("score", "crashed", "completion", "red_runs", "right_turn_no_stop", "collisions") if k in res})
    ground = ground_for(data_key_for(env), offline=True) if data_key_for(env) else None
    mount = mount_icon()
    title = f"카메라 1대 + 위성지도로 운전 (모방학습 CNN, 실드 없음) — {a.scenario}"
    sel = [i for i, f in enumerate(frames) if a.t0 <= f["t"] < a.t0 + a.dur]
    w = imageio_ffmpeg.write_frames(a.out, (W, H), fps=10, codec="libx264", quality=7, macro_block_size=8,
                                    output_params=["-pix_fmt", "yuv420p"])
    w.send(None)
    for i in sel:
        f = frames[i]
        v = f["vio"]
        stats = [f"시간 {f['t']:5.1f} s   진행 {f['s']:6.0f} m",
                 f"충돌 {'예' if f['crashed'] else '없음'}  신호위반 {v.get('red_runs', 0) + v.get('right_turn_no_stop', 0)}  과속 {v.get('speeding_s', 0):.1f}s"]
        img = compose(env, ground, f, title, mount, others_log[min(2 * i, len(others_log) - 1)], stats)
        w.send(np.ascontiguousarray(np.asarray(img)))
        if a.still and a.still_t is not None and abs(f["t"] - a.still_t) < 0.06:
            img.save(a.still)
    w.close()
    print("wrote", a.out, len(sel), "frames")


if __name__ == "__main__":
    main()
