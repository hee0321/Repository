"""카메라+내비만 보고 운전하는 루프 (평가·데이터 모으기·영상이 모두 이 함수를 씀).

제어 주기 10 Hz (시뮬레이터 20 Hz 두 스텝마다 한 번 보고 결정, 사람 반응처럼 0.1 s 유지).
expert(규칙 기반 기준 제어기)는 '선생님'으로만 매 스텝 계산 — 정책 입력에는 들어가지 않는다.
"""
from __future__ import annotations

import numpy as np

from ..controllers import PurePursuitController
from .camera import camera_for
from .nav import Nav
from .policy import stack_frames

CONTROL_EVERY = 2


def drive_episode(env, policy=None, seed=0, beta=0.0, dart=0.0, collect=False, record=False, rng=None,
                  cam_noise=0.0, offline=True, smooth=0.0):
    """policy: torch 모델(VisionPolicy) 또는 None(=expert 가 운전).
    beta: 이 확률로 그 결정 구간은 expert 가 운전(DAgger 섞기). dart: 실행 행동에 더할 잡음(σ).
    collect=True 면 (영상, 내비, expert 행동, 가중치) 를 모아 돌려준다. record=True 면 영상용 프레임 기록.
    반환: (결과 dict, 모은 데이터 또는 None, 프레임 목록 또는 None)"""
    import torch
    rng = rng or np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    expert = PurePursuitController(env)
    cam = camera_for(env, offline=offline, rng=rng)
    nav = Nav(env, rng=np.random.default_rng(seed + 1000))
    prev_frame = None
    hist = []
    imgs, navs, acts, wts = [], [], [], []
    frames = []
    a_exec = np.zeros(2, np.float32)
    a_prev = np.zeros(2, np.float32)
    devs, speeds, dsteer, dacc = [], [], [], []
    done = False
    step = 0
    use_expert = policy is None
    while not done:
        a_exp = expert.act(obs)
        if step % CONTROL_EVERY == 0:
            frame = cam.render(noise=cam_noise)
            hist.append(frame)
            prev_frame = hist[-3] if len(hist) >= 3 else hist[0]   # 0.2 s 전
            if len(hist) > 4:
                hist.pop(0)
            x_img = stack_frames(frame, prev_frame)
            x_nav = nav.features(float(a_prev[1]))
            if collect:
                imgs.append(x_img)
                navs.append(x_nav)
                acts.append(a_exp.copy())
                # 안전 중요 순간 가중: 세게 감속, 정지선·장애물 가까이, 큰 조향
                w = 1.0 + 2.0 * float(a_exp[1] < -0.3) + 1.0 * float(abs(a_exp[0]) > 0.3)
                ttc = env.ttc()
                if ttc < 4 or env.stop_distance() < 30:
                    w += 2.0
                wts.append(w)
            if policy is not None:
                use_expert = rng.random() < beta
            if use_expert:
                a = a_exp.copy()
            else:
                with torch.no_grad():
                    a = policy(torch.from_numpy(x_img[None]), torch.from_numpy(x_nav[None])).numpy()[0]
                if smooth > 0:
                    a = (1 - smooth) * a + smooth * a_exec
            if dart > 0:
                a = a + rng.normal(0, dart, 2).astype(np.float32)
            a_exec = np.clip(a, -1, 1).astype(np.float32)
            if record:
                frames.append(dict(img=frame, x=env.car.x, y=env.car.y, yaw=env.car.yaw, v=env.car.v, t=env.t,
                                   s=env._s, a=a_exec.copy(), a_exp=a_exp.copy(), limit=env.rules.limit_at(env._s),
                                   nav=x_nav.copy(), sig=_sig_state(env), crashed=env.crashed,
                                   vio=dict(env.vlog.summary())))
        if use_expert and policy is not None:
            a_cmd = a_exp if dart == 0 else a_exec
        else:
            a_cmd = a_exec if policy is not None or dart > 0 else a_exp
        obs, r, term, trunc, info = env.step(a_cmd)
        devs.append(abs(info["lane_dev"]))
        speeds.append(info["speed"])
        dsteer.append(abs(a_cmd[0] - a_prev[0]))
        dacc.append(abs(a_cmd[1] - a_prev[1]))
        a_prev = np.asarray(a_cmd, np.float32)
        step += 1
        done = term or trunc
    v = info["violations"]
    res = {
        "completion": round(info["completion"], 3), "finished": bool(info["finished"] or info["completion"] >= 0.999),
        "crashed": bool(info["crashed"]), "time_s": round(info["t"], 1),
        "mean_speed_kmh": round(float(np.mean(speeds)) * 3.6, 1),
        "mean_lane_dev_m": round(float(np.mean(devs)), 3), "max_lane_dev_m": round(float(np.max(devs)), 3),
        "smooth_steer": round(float(np.mean(dsteer)), 4), "smooth_accel": round(float(np.mean(dacc)), 4), **v,
    }
    data = None
    if collect:
        data = dict(img=np.stack(imgs), nav=np.stack(navs), act=np.stack(acts).astype(np.float32),
                    w=np.array(wts, np.float32))
    return res, data, (frames if record else None)


def _sig_state(env):
    nx = env.rules.next_signal(env._s, 150.0)
    return None if nx is None else (round(nx[0], 1), nx[1].state(env.t), bool(nx[1].right_turn))
