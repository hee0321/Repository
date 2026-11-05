"""카메라 인식 + 추적 + 예측 + 안전 계획기를 붙인 비전 운전 루프.

- 조향·기본 가속: 기존 비전 정책(128×72 영상 + 내비) 그대로.
- 인식: 같은 카메라(같은 장착 위치)의 256×144 영상 → 분할 CNN → 물체·신호·차선.
- 안전 계획기가 필요한 감속을 정하면 정책 가속보다 더 세게 줄여요(가속은 절대 더하지 않음).
- 시뮬레이터 정답은 measure=True 일 때 오차 측정용으로만 계산해 기록(제어에 안 씀).
"""
from __future__ import annotations

import numpy as np

from ..vision.camera import CameraMount, LBL, camera_for
from ..vision.drive import CONTROL_EVERY
from ..vision.nav import Nav
from ..vision.policy import stack_frames
from .detect import Geometry, detect, lane_points
from .planner import A_MAX, SafetyPlanner
from .seg import PERC_H, PERC_W, segment
from .track import Odometry, Tracker


def gt_objects(env, cam, inst):
    """정답 물체(측정용): 영상에 4 px 이상 보이는 것만. 반환 [(번호, 종류, 앞, 왼, 속도)] (뒷축 기준)."""
    out = []
    ids, cnt = np.unique(inst, return_counts=True)
    vis = {int(i): int(c) for i, c in zip(ids, cnt) if 0 < i < 1000 and c >= 4}
    c, s = np.cos(env.car.yaw), np.sin(env.car.yaw)
    for k1, npx in vis.items():
        k = k1 - 1
        p = env.track.point_at(env.ob_s[k]) + env.track.normal_at(env.ob_s[k]) * env.ob_d[k]
        d = p - (env.car.x, env.car.y)
        kind = env.ob_kind[k]
        cls = "ped" if kind == "ped" else ("object" if (kind == "object" or abs(env.ob_v[k]) < 1e-6 and kind not in ("car", "entering")) else "car")
        out.append((k, cls, float(d[0] * c + d[1] * s), float(-d[0] * s + d[1] * c), float(env.ob_v[k]), npx))
    return out


def gt_signal(env, inst):
    ids = [int(i) - 1000 for i in np.unique(inst) if i >= 1000]
    if not ids:
        return None
    sg = env.rules.signals[ids[0]]
    return sg.state(env.t)


def drive_perception(env, policy, seg_model, seed=0, use_planner=True, measure=False, record=False, rng=None):
    import torch
    rng = rng or np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    cam = camera_for(env, offline=True, rng=rng)
    pm = CameraMount(width=PERC_W, height_px=PERC_H)
    pcam = camera_for(env, offline=True, rng=rng, mount=pm)
    geo = Geometry(pm)
    nav = Nav(env, rng=np.random.default_rng(seed + 1000))
    odo, tracker = Odometry(np.random.default_rng(seed + 7)), Tracker()
    planner = SafetyPlanner(geo, odo)
    hist, frames, metrics = [], [], dict(det=[], sig=[])
    a_exec = np.zeros(2, np.float32)
    a_prev = np.zeros(2, np.float32)
    devs, speeds, dsteer, dacc = [], [], [], []
    done, step = False, 0
    yaw_prev = env.car.yaw
    brake_steps = 0
    dtc = env.dt * CONTROL_EVERY
    while not done:
        if step % CONTROL_EVERY == 0:
            frame = cam.render()
            hist.append(frame)
            prev_frame = hist[-3] if len(hist) >= 3 else hist[0]
            if len(hist) > 4:
                hist.pop(0)
            x_nav = nav.features(float(a_prev[1]))
            with torch.no_grad():
                out = policy(torch.from_numpy(stack_frames(frame, prev_frame)[None]), torch.from_numpy(x_nav[None])).numpy()[0]
            a = out[:2].copy()
            if len(out) > 2:
                ind = out[2]
                if ind > 0.5 and env.indicator != 1:
                    env.set_indicator(1)
                elif ind < -0.5 and env.indicator != -1:
                    env.set_indicator(-1)
                elif abs(ind) < 0.2 and env.indicator != 0:
                    env.set_indicator(0)
            # --- 인식 (카메라 영상만) ---
            if measure or record:
                pimg, gt_cls, gt_inst = pcam.render_with_labels()
            else:
                pimg = pcam.render()
            cls_map, conf = segment(seg_model, pimg)
            dyaw = (env.car.yaw - yaw_prev + np.pi) % (2 * np.pi) - np.pi
            yaw_prev = env.car.yaw
            odo.step(env.car.v, dyaw / dtc, dtc)          # 내 차 계기(속도·요레이트)
            dets, lamps = detect(cls_map, conf, geo)
            tracks = tracker.step(dets, odo, dtc)
            a_req, path = planner.plan(env.car.v, env.car.steer, tracks, lamps, cls_map, conf, dtc)
            if use_planner and a_req > 0.05:
                a[1] = min(a[1], -min(a_req, A_MAX) / A_MAX)
                brake_steps += 1
            a_exec = np.clip(a, -1, 1).astype(np.float32)
            if measure or record:
                gto = gt_objects(env, pcam, gt_inst)
                gsig = gt_signal(env, gt_inst)
                metrics["det"].append(_match(dets, gto))
                det_sig = None
                if lamps:
                    det_sig = min(lamps, key=lambda l: l.fwd).extra["state"]
                metrics["sig"].append((gsig, det_sig))
            if record:
                frames.append(dict(img=pimg, cls=cls_map, conf=conf, dets=dets, lamps=lamps,
                                   tracks=[(t.tid, t.cls, odo.to_ego(t.x[:2]).copy(), odo.rot().T @ t.x[2:],
                                            np.sqrt(max(np.linalg.eigvalsh(t.P[:2, :2]).max(), 0)), t.confirmed, t.misses,
                                            [(tt, odo.to_ego(p).copy(), np.sqrt(max(np.linalg.eigvalsh(Pc).max(), 0)))
                                             for tt, p, Pc in t.predict_traj()]) for t in tracks],
                                   gt=gto, gsig=gsig, lanes=lane_points(cls_map, geo), path=path, free=planner.free,
                                   reason=planner.reason, a_req=a_req, v=env.car.v, t=env.t, a=a_exec.copy(),
                                   crashed=env.crashed))
        obs, r, term, trunc, info = env.step(a_exec)
        devs.append(abs(info["lane_dev"]))
        speeds.append(info["speed"])
        dsteer.append(abs(a_exec[0] - a_prev[0]))
        dacc.append(abs(a_exec[1] - a_prev[1]))
        a_prev = a_exec.copy()
        step += 1
        done = term or trunc
    v = info["violations"]
    res = {
        "completion": round(info["completion"], 3), "finished": bool(info["finished"] or info["completion"] >= 0.999),
        "crashed": bool(info["crashed"]), "time_s": round(info["t"], 1),
        "mean_speed_kmh": round(float(np.mean(speeds)) * 3.6, 1),
        "mean_lane_dev_m": round(float(np.mean(devs)), 3), "max_lane_dev_m": round(float(np.max(devs)), 3),
        "smooth_steer": round(float(np.mean(dsteer)), 4), "smooth_accel": round(float(np.mean(dacc)), 4), **v,
        "planner_brake_share": round(brake_steps / max(step / CONTROL_EVERY, 1), 3),
    }
    return res, metrics, frames


def _match(dets, gto, max_d=60.0):
    """같은 종류끼리 가장 가까운 짝 (정답 기준 3 m 또는 거리 15 % 안). 반환: 측정용 목록."""
    rows = []
    used = set()
    for k, cls, fx, ly, vel, npx in gto:
        d_gt = float(np.hypot(fx, ly))
        if d_gt > max_d:
            continue
        best = None
        for j, dt in enumerate(dets):
            if j in used or dt.cls != cls:
                continue
            e = float(np.hypot(dt.fwd + 2.45 - fx, dt.left - ly))
            if e < max(3.0, 0.15 * d_gt) and (best is None or e < best[0]):
                best = (e, j)
        if best:
            used.add(best[1])
            d = dets[best[1]]
            rows.append(dict(cls=cls, gt_d=d_gt, hit=True, err=float(np.hypot(d.fwd + 2.45, d.left) - d_gt), sigma=d.sigma,
                             occl=d.occluded))
        else:
            rows.append(dict(cls=cls, gt_d=d_gt, hit=False))
    fp = sum(1 for j, d in enumerate(dets) if j not in used and d.dist < max_d)
    return dict(rows=rows, fp=fp, n_det=len([d for d in dets if d.dist < max_d]))
