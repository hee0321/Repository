"""자율 주행 재생: 운전 화면 + 판단 근거 패널 → GIF/MP4, 그리고 브라우저용 HTML 재생기."""
from __future__ import annotations

import json
import os

import numpy as np

MODE_COLOR = {"AUTONOMOUS": "#2a9d8f", "TAKEOVER": "#e76f51", "SAFE_STOP": "#d62828", "BASELINE": "#457b9d"}
MODE_KO = {"AUTONOMOUS": "자율 주행(모델)", "TAKEOVER": "개입 요청 → 기준 제어기", "SAFE_STOP": "안전 정지",
           "BASELINE": "기준 제어기"}
KIND_COLOR = {"car": "#6c757d", "tractor": "#bc6c25", "oncoming": "#9d0208", "ped": "#7209b7", "object": "#343a40"}
LIGHT_KO = {"green": "초록", "yellow": "노랑", "red": "빨강", "-": "없음"}


def _car_poly(x, y, yaw, L=4.0, W=1.8):
    pts = np.array([[-0.8, -W / 2], [L - 0.8, -W / 2], [L - 0.8, W / 2], [-0.8, W / 2]])
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return pts @ R.T + [x, y]


def panel_lines(f):
    lines = [f"모드: {MODE_KO.get(f['mode'], f['mode'])}",
             f"속도 {f['v']:.0f} / 제한 {f['limit']} km/h" + ("  [어린이보호구역]" if f["school"] else ""),
             f"신호: {LIGHT_KO.get(f['light'], f['light'])}" + (f" ({f['light_d']:.0f} m)" if f["light_d"] is not None else ""),
             f"가장 가까운 장애물 TTC: {'-' if f['ttc'] is None else str(f['ttc']) + ' s'}",
             f"시야 거리: {f['sight']:.0f} m",
             f"적용 규칙: {f['rule'] or '-'}",
             f"선택 행동: 조향 {f['action'][0]:+.2f}, 가속 {f['action'][1]:+.2f}"]
    if f["a_model"] is not None:
        lines.append(f"모델 평균 {f['a_model'][0]:+.2f},{f['a_model'][1]:+.2f} ± {f['std'][0]:.2f}")
        lines.append(f"확신도 {f['conf']:.2f}  가치 {f['value']:.1f}")
    lines.append(f"개입 요청 누적: {f['n_events']}회")
    if f["risks"]:
        lines.append("⚠ " + ", ".join(f["risks"])[:60])
    return lines


def render_autopilot(env, frames, name, out_dir, gif=True, mp4=False, html=True, view=45.0, max_frames=120):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    plt.rcParams["font.family"] = ["NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"]
    tr = env.track
    outs = []
    fig = plt.figure(figsize=(8.4, 4.2), dpi=72)
    ax = fig.add_axes([0.0, 0.0, 0.5, 1.0])
    tx = fig.add_axes([0.5, 0.0, 0.5, 1.0]); tx.axis("off")
    ax.set_facecolor("#1d3557" if env.rules.light < 0.5 else "#a7c957")  # 밤/낮 바탕
    for b in (tr.left, tr.right):
        ax.plot(b[:, 0], b[:, 1], color="white", lw=2)
    ax.fill(np.r_[tr.left[:, 0], tr.right[::-1, 0]], np.r_[tr.left[:, 1], tr.right[::-1, 1]],
            color="#b7b7a4" if env.rules.rural else "#6c757d", zorder=0)
    ax.plot(tr.center[:, 0], tr.center[:, 1], "--", color="#ffd166" if not env.rules.rural else "#ddd", lw=1)
    for sg in env.rules.signals:
        ax.plot(*tr.point_at(sg.s), "s", color="white", ms=6, zorder=3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    car = plt.Polygon(_car_poly(0, 0, 0), color="#00b4d8", zorder=6); ax.add_patch(car)
    rays = [ax.plot([], [], color="#ff595e", lw=0.5, alpha=0.6, zorder=4)[0] for _ in range(len(frames[0]["rays"]))]
    obs_sc = ax.scatter([], [], s=[], zorder=5)
    sig_dot, = ax.plot([], [], "o", ms=9, zorder=7)
    texts = [tx.text(0.03, 0.87 - i * 0.072, "", fontsize=10, transform=tx.transAxes, family=plt.rcParams["font.family"])
             for i in range(12)]
    head = tx.text(0.03, 0.985, name, fontsize=11, weight="bold", transform=tx.transAxes, va="top")
    angles = env.lidar.angles

    def upd(i):
        f = frames[i]
        car.set_xy(_car_poly(f["x"], f["y"], f["yaw"]))
        car.set_color(MODE_COLOR.get(f["mode"], "#00b4d8"))
        for ln, a, r in zip(rays, angles, f["rays"]):
            ln.set_data([f["x"], f["x"] + r * np.cos(f["yaw"] + a)], [f["y"], f["y"] + r * np.sin(f["yaw"] + a)])
        if f["obs"]:
            o = np.array([[q[0], q[1]] for q in f["obs"]])
            obs_sc.set_offsets(o)
            obs_sc.set_sizes([max(20, (q[2] * 18) ** 2) for q in f["obs"]])
            obs_sc.set_color([KIND_COLOR.get(q[3], "#333") for q in f["obs"]])
        else:
            obs_sc.set_offsets(np.zeros((0, 2)))
        if f["light"] in ("green", "yellow", "red"):
            nx = env.rules.next_signal(f["s"], 150.0)
            if nx is not None:
                p = tr.point_at(nx[1].s)
                sig_dot.set_data([p[0]], [p[1]]); sig_dot.set_color({"green": "#2dc653", "yellow": "#ffd60a", "red": "#e5383b"}[f["light"]])
        else:
            sig_dot.set_data([], [])
        ax.set_xlim(f["x"] - view, f["x"] + view); ax.set_ylim(f["y"] - view, f["y"] + view)
        head.set_text(f"{name}   t={f['t']:.1f}s")
        lines = panel_lines(f)
        for k, t in enumerate(texts):
            t.set_text(lines[k] if k < len(lines) else "")
            t.set_color(MODE_COLOR.get(f["mode"]) if k == 0 else ("#d62828" if lines[k:k + 1] and lines[k].startswith("⚠") else "black"))
        return [car, obs_sc, sig_dot, head, *rays, *texts]

    # GIF/MP4는 용량을 줄이려고 프레임을 고르게 솎되, 개입 요청이 생긴 프레임은 남긴다
    idx = np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).astype(int).tolist()
    ev = [k for k in range(1, len(frames)) if frames[k]["n_events"] != frames[k - 1]["n_events"]][:15]
    idx = sorted(set(idx) | set(ev) | {min(e + j, len(frames) - 1) for e in ev for j in range(1, 6)})
    anim = FuncAnimation(fig, lambda k: upd(idx[k]), frames=len(idx), blit=False)
    if gif:
        p = os.path.join(out_dir, f"autopilot_{name}.gif")
        anim.save(p, writer=PillowWriter(fps=10)); outs.append(p)
    if mp4:
        p = os.path.join(out_dir, f"autopilot_{name}.mp4")
        anim.save(p, writer=FFMpegWriter(fps=10, bitrate=900)); outs.append(p)
    plt.close(fig)
    if html:
        p = os.path.join(out_dir, f"autopilot_{name}.html")
        write_html(env, frames, name, p); outs.append(p)
    return outs


def write_html(env, frames, name, path):
    tr = env.track
    step = max(1, len(tr.left) // 1500)
    road = {"left": np.round(tr.left[::step], 1).tolist(), "right": np.round(tr.right[::step], 1).tolist(),
            "night": env.rules.light < 0.5, "rural": env.rules.rural,
            "signals": [np.round(tr.point_at(sg.s), 1).tolist() for sg in env.rules.signals]}
    for f in frames:
        f["panel"] = panel_lines(f)
        f.pop("rays", None)
    data = json.dumps({"name": name, "road": road, "frames": frames}, ensure_ascii=False, default=float)
    html = HTML_TMPL.replace("__DATA__", data).replace("__NAME__", name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)


HTML_TMPL = """<!doctype html><html lang="ko"><meta charset="utf-8"><title>자율 주행 재생 - __NAME__</title>
<style>body{font-family:sans-serif;margin:12px;background:#f8f9fa}#wrap{display:flex;gap:14px}
canvas{background:#a7c957;border-radius:8px}#panel{white-space:pre-line;font-size:15px;line-height:1.6;min-width:360px}
.mode{font-weight:bold;font-size:18px}.warn{color:#d62828}</style>
<h2>자율 주행 재생: __NAME__</h2>
<div><button id="play">▶ 재생</button> <input id="sl" type="range" min="0" value="0" style="width:520px">
<span id="tt"></span> 속도 <select id="sp"><option>1</option><option selected>2</option><option>4</option></select>x</div>
<div id="wrap"><canvas id="cv" width="560" height="560"></canvas><div id="panel"></div></div>
<p style="font-size:12px;color:#666">파랑=자율, 주황=개입 요청(기준 제어기), 빨강=안전 정지. 지도 데이터 © OpenStreetMap contributors (ODbL).</p>
<script>
const D=__DATA__;const F=D.frames,R=D.road;const cv=document.getElementById('cv'),c=cv.getContext('2d');
const sl=document.getElementById('sl');sl.max=F.length-1;let i=0,playing=false;
const MC={AUTONOMOUS:'#2a9d8f',TAKEOVER:'#e76f51',SAFE_STOP:'#d62828',BASELINE:'#457b9d'};
const KC={car:'#6c757d',tractor:'#bc6c25',oncoming:'#9d0208',ped:'#7209b7',object:'#343a40'};
function draw(){const f=F[i],S=6,W=cv.width;cv.style.background=R.night?'#1d3557':'#a7c957';
c.clearRect(0,0,W,W);const tx=x=>W/2+(x-f.x)*S,ty=y=>W/2-(y-f.y)*S;
c.fillStyle=R.rural?'#b7b7a4':'#6c757d';c.beginPath();R.left.forEach((p,k)=>k?c.lineTo(tx(p[0]),ty(p[1])):c.moveTo(tx(p[0]),ty(p[1])));
for(let k=R.right.length-1;k>=0;k--)c.lineTo(tx(R.right[k][0]),ty(R.right[k][1]));c.fill();
c.strokeStyle='white';c.lineWidth=2;[R.left,R.right].forEach(b=>{c.beginPath();b.forEach((p,k)=>k?c.lineTo(tx(p[0]),ty(p[1])):c.moveTo(tx(p[0]),ty(p[1])));c.stroke()});
R.signals.forEach(p=>{c.fillStyle=f.light=='red'?'#e5383b':f.light=='yellow'?'#ffd60a':'#2dc653';c.fillRect(tx(p[0])-5,ty(p[1])-5,10,10)});
f.obs.forEach(o=>{c.fillStyle=KC[o[3]]||'#333';c.beginPath();c.arc(tx(o[0]),ty(o[1]),Math.max(3,o[2]*S),0,7);c.fill()});
c.save();c.translate(tx(f.x),ty(f.y));c.rotate(-f.yaw);c.fillStyle=MC[f.mode]||'#00b4d8';c.fillRect(-0.8*S,-0.9*S,4*S,1.8*S);c.restore();
const P=document.getElementById('panel');P.innerHTML='<div class="mode" style="color:'+MC[f.mode]+'">'+f.panel[0]+'</div>'+
f.panel.slice(1).map(l=>l.startsWith('⚠')?'<div class="warn">'+l+'</div>':'<div>'+l+'</div>').join('');
document.getElementById('tt').textContent='t='+f.t.toFixed(1)+'s';sl.value=i;}
sl.oninput=()=>{i=+sl.value;draw()};document.getElementById('play').onclick=()=>{playing=!playing;
document.getElementById('play').textContent=playing?'⏸ 멈춤':'▶ 재생'};
setInterval(()=>{if(playing){i=Math.min(F.length-1,i+(+document.getElementById('sp').value));if(i>=F.length-1)playing=false;draw()}},100);draw();
</script></html>"""
