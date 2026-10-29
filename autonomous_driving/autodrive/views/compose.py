"""시점 조합(한 화면 / 나란히 / 2×2) + MP4·GIF 저장."""
from __future__ import annotations

import os

from PIL import Image

from .driver import DriverView
from .engine import EngineView
from .satellite_panel import SatelliteView
from .state import interpolate
from .topdown import TopDownView

# 보기 이름 → 패널 배치(행 목록)
VIEWS = {
    "driver": [["driver"]],
    "top": [["top"]],
    "engine": [["engine"]],
    "satellite": [["satellite"]],
    "both": [["driver", "top"]],
    "all": [["driver", "top"], ["engine", "satellite"]],
}
PW, PH = 640, 360


def make_panels(env, names, kinds):
    out = {}
    for n in names:
        if n == "driver":
            out[n] = DriverView(env, PW, PH, kinds=kinds)
        elif n == "top":
            out[n] = TopDownView(env, PW, PH, kinds=kinds)
        elif n == "engine":
            out[n] = EngineView(env, PW, PH)
        elif n == "satellite":
            out[n] = SatelliteView(env, PW, PH, kinds=kinds)
    return out


def compose(panels, layout, f):
    rows = len(layout)
    cols = max(len(r) for r in layout)
    img = Image.new("RGB", (cols * PW, rows * PH), (0, 0, 0))
    for i, r in enumerate(layout):
        for j, n in enumerate(r):
            img.paste(panels[n].render(f), (j * PW, i * PH))
    return img


def _save_gif(frames, path, fps, max_mb):
    scale = 1.0
    for _ in range(6):
        fr = frames if scale == 1.0 else [im.resize((int(im.width * scale) // 2 * 2, int(im.height * scale) // 2 * 2), Image.LANCZOS)
                                          for im in frames]
        pal = [im.convert("P", palette=Image.ADAPTIVE, colors=96) for im in fr]
        pal[0].save(path, save_all=True, append_images=pal[1:], duration=int(1000 / fps), loop=0, optimize=True, disposal=1)
        if os.path.getsize(path) <= max_mb * 1e6:
            return path
        scale *= 0.8
    return path


def render_views(env, rows, kinds, view, out_base, fps=25.0, t0=None, t1=None, mp4=True, gif=True,
                 gif_fps=12.5, gif_width=None, gif_max_mb=8.0, still_at=None, progress=True):
    """rows: StateRecorder.rows → fps로 보간 → 시점 배치대로 그려서 out_base + '.mp4' / '.gif' 저장.
    still_at: 이 시각(초) 프레임을 out_base + '_still.png'로도 저장. 반환: 저장한 파일 목록."""
    layout = VIEWS[view]
    names = [n for r in layout for n in r]
    panels = make_panels(env, names, kinds)
    frames = interpolate(rows, fps, t0, t1)
    outs = []
    writer = None
    if mp4:
        import imageio_ffmpeg
        cols, nrows = max(len(r) for r in layout), len(layout)
        size = (cols * PW, nrows * PH)
        p_mp4 = out_base + ".mp4"
        writer = imageio_ffmpeg.write_frames(p_mp4, size, fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                             output_params=["-crf", "23", "-preset", "medium"], macro_block_size=1)
        writer.send(None)
    gif_frames = []
    gif_every = max(1, int(round(fps / gif_fps)))
    if gif_width is None:
        gif_width = 640 if len(layout[0]) == 1 else 800
    still = None
    for k, f in enumerate(frames):
        img = compose(panels, layout, f)
        if writer is not None:
            writer.send(img.tobytes())
        if gif and k % gif_every == 0:
            h = int(img.height * gif_width / img.width) // 2 * 2
            gif_frames.append(img.resize((gif_width, h), Image.LANCZOS) if img.width != gif_width else img.copy())
        if still_at is not None and (still is None or abs(f["t"] - still_at) < abs(still[0] - still_at)):
            still = (f["t"], img)
        if progress and k % 100 == 0:
            print(f"  [{view}] 프레임 {k}/{len(frames)}", flush=True)
    if writer is not None:
        writer.close()
        outs.append(out_base + ".mp4")
    if gif and gif_frames:
        outs.append(_save_gif(gif_frames, out_base + ".gif", fps / gif_every, gif_max_mb))
    if still is not None:
        still[1].save(out_base + "_still.png")
        outs.append(out_base + "_still.png")
    return outs
