"""매 시뮬레이션 스텝 상태 기록 → 원하는 fps로 보간(부드러운 영상용) → 구간 고르기."""
from __future__ import annotations

import numpy as np

# 보간할 연속 값 / 가장 가까운 스텝 값을 쓸 이산 값
CONT = ("t", "x", "y", "v", "steer", "steer_cmd", "accel_cmd", "s", "d", "limit", "mu", "sight", "lat_acc",
        "acc", "soc", "torque", "motor_rpm", "p_motor_kw", "p_bat_kw", "regen_kw", "fric_kw", "fric_frac",
        "e_used_wh", "e_regen_wh")
EV_KEYS = ("soc", "torque", "motor_rpm", "p_motor_kw", "p_bat_kw", "regen_kw", "fric_kw", "fric_frac", "e_used_wh", "e_regen_wh")
DISC = ("mode", "risks", "school", "gravel", "slipping", "n_events", "ttc")


class StateRecorder:
    """env.step 뒤마다 record(env, action, why)를 부른다. 장애물은 '보이는 것'만이 아니라 전부 같은 순서로 기록해서
    프레임 사이 보간이 가능하다(밤에 안 보이는 건 그리는 쪽에서 조명으로 처리)."""

    def __init__(self, env):
        self.env = env
        self.rows = []
        self.kinds = None

    def record(self, env, action=(0.0, 0.0), why: dict | None = None):
        why = why or {}
        c = env.car
        n = len(env.ob_s)
        if n:
            P = np.array([env.track.point_at(env.ob_s[k]) + env.track.normal_at(env.ob_s[k]) * env.ob_d[k] for k in range(n)])
            H = np.array([env.track.heading_at(env.ob_s[k]) for k in range(n)])
            ob = np.column_stack([P, env.ob_r, env.ob_l, H, env.ob_v, env.ob_s])
        else:
            ob = np.zeros((0, 7))
        if self.kinds is None:
            self.kinds = list(env.ob_kind)
        ttc = why.get("ttc", env.ttc())
        nx = env.rules.next_signal(env._s, 150.0)
        self.rows.append({
            "t": env.t, "x": c.x, "y": c.y, "yaw": c.yaw, "v": c.v, "steer": c.steer,
            "steer_cmd": float(action[0]), "accel_cmd": float(action[1]), "s": env._s, "d": env._d,
            "limit": env.rules.limit_at(env._s), "mu": env.params.mu, "sight": env.sight, "lat_acc": c.lat_acc,
            "mode": why.get("mode", "BASELINE"), "risks": list(why.get("risks", [])),
            "school": env.rules.school_at(env._s), "gravel": env.rules.gravel_at(env._s), "slipping": c.slipping,
            "n_events": why.get("n_events", 0), "ttc": None if ttc is None or not np.isfinite(ttc) else float(ttc),
            "sig_d": None if nx is None else float(nx[0]), "ob": ob,
            "acc": float(getattr(c, "accel", 0.0)), **ev_snapshot(c)})


def ev_snapshot(car):
    """전기차 구동계 값(없는 옛 차 객체면 0)."""
    ev = getattr(car, "ev", None)
    if ev is None:
        return {k: (0.8 if k == "soc" else 0.0) for k in EV_KEYS}
    return {k: float(v) for k, v in ev.snapshot().items()}


def _lerp_angle(a, b, w):
    return a + np.angle(np.exp(1j * (b - a))) * w


def interpolate(rows, fps=25.0, t0=None, t1=None):
    """스텝 기록(보통 20 Hz)을 fps로 다시 뽑는다. 연속 값은 선형 보간, 각도는 감아서 보간."""
    if not rows:
        return []
    ts = np.array([r["t"] for r in rows])
    t0 = ts[0] if t0 is None else max(t0, ts[0])
    t1 = ts[-1] if t1 is None else min(t1, ts[-1])
    out = []
    for t in np.arange(t0, t1 + 1e-9, 1.0 / fps):
        i = int(np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(rows) - 1))
        j = min(i + 1, len(rows) - 1)
        w = 0.0 if j == i or ts[j] == ts[i] else float((t - ts[i]) / (ts[j] - ts[i]))
        a, b = rows[i], rows[j]
        f = {k: a[k] + (b[k] - a[k]) * w for k in CONT}
        f["t"] = float(t)
        f["yaw"] = float(_lerp_angle(a["yaw"], b["yaw"], w))
        near = a if w < 0.5 else b
        for k in DISC:
            f[k] = near[k]
        if a["ob"].shape == b["ob"].shape and len(a["ob"]):
            ob = a["ob"] + (b["ob"] - a["ob"]) * w
            ob[:, 4] = _lerp_angle(a["ob"][:, 4], b["ob"][:, 4], w)
            f["ob"] = ob
        else:
            f["ob"] = near["ob"]
        out.append(f)
    return out


def pick_window(rows, seconds=18.0, start=None, kinds=None):
    """영상 구간 고르기. start를 주면 그대로, 아니면 '볼거리'(가까운 차·보행자, 자갈길, 개입 요청, 굽이)가
    가장 많은 구간을 고른다. → (t0, t1)"""
    ts = np.array([r["t"] for r in rows])
    if seconds <= 0 or ts[-1] - ts[0] <= seconds:
        return float(ts[0]), float(ts[-1])
    if start is not None:
        return float(start), float(start + seconds)
    score = np.zeros(len(rows))
    for k, r in enumerate(rows):
        ob = r["ob"]
        if len(ob):
            gap = ob[:, 6] - r["s"]
            score[k] += 2.0 * np.sum((gap > 0) & (gap < 40))
        score[k] += 0.6 * r["gravel"] + 3.0 * bool(r["risks"]) + 0.5 * min(abs(r["steer"]) * 6, 1.0)
        score[k] += 0.2 * min(r["v"] / 8.0, 1.0)
        if r.get("sig_d") is not None and r["sig_d"] < 60:
            score[k] += 1.5  # 신호등 교차로  # 서 있는 구간보다 달리는 구간
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.05
    n = max(1, int(seconds / dt))
    cs = np.convolve(score, np.ones(n), mode="valid")
    k0 = int(np.argmax(cs))
    return float(ts[k0]), float(ts[k0] + seconds)
