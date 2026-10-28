import json
cells = []
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n")})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n")})

md("""
# Microduck RL — 무료 GPU에서 강화학습 이어 하기 (Kaggle / Colab)

- **Kaggle**: Settings → Accelerator **GPU T4 x2** (P100은 안 됨: 이 레포가 고정한 torch 2.9.1+cu128은 sm_70 미만 GPU를 지원하지 않음), **Internet on**. 계정 전화번호 인증이 있어야 GPU/인터넷을 켤 수 있음.
- **Colab**: 런타임 유형 → **T4 GPU**. 같은 노트북이 그대로 동작 (출력 폴더만 `/content/out`).
- wandb 사용 안 함 (`WANDB_MODE=disabled`, tensorboard 로거). 비용 0원.
- 기본: Pollen 공개 v8 번들 `velstand/0045i2si_2749`에서 이어 학습. `MODE="velocity_flat"`로 바꾸면 걷기(평지)를 처음부터 학습.
- 세션 한도(Kaggle/Colab 모두 최대 12시간) 전에 끝나도록 `TIME_BUDGET_H`로 학습을 끊고, 체크포인트 + ONNX를 출력 폴더에 저장. 다음 세션에서는 `RESUME_FROM`에 이전 `.pt`를 넣어 이어 가면 됨.
""")
code("""
# ==== 설정 ====
MODE = "velstand_resume"        # "velstand_resume" | "velocity_flat"
NUM_ENVS = 4096                 # T4 메모리가 부족하면 2048 / 1024로
MAX_ITERATIONS = 1000           # 추가로 돌릴 반복 수 (시간 예산에서 먼저 끊길 수 있음)
SAVE_INTERVAL = 50
TIME_BUDGET_H = 10.5            # 12시간 세션 한도보다 여유 있게
RESUME_FROM = None              # 예: "/kaggle/input/<내 데이터셋>/model_2800.pt" (이전 세션 결과로 이어 가기)
REPO = "https://github.com/pollen-robotics/microduck_rl.git"
BRANCH = "develop"
COMMIT = "bae39f5"              # CPU에서 v8 번들 이어 학습이 검증된 커밋. None이면 최신 develop
""")
code("""
import os, pathlib, subprocess, sys
OUT = pathlib.Path("/kaggle/working/out") if os.path.isdir("/kaggle/working") else pathlib.Path("/content/out")
OUT.mkdir(parents=True, exist_ok=True)
WORK = pathlib.Path("/kaggle/temp" if os.path.isdir("/kaggle") else "/content") / "work"
WORK.mkdir(parents=True, exist_ok=True)
print("output dir:", OUT, "| work dir:", WORK)
!nvidia-smi
""")
md("GPU 이름이 **Tesla T4**인지, Driver/CUDA Version이 12.4 이상인지 확인하세요 (mujoco_warp는 CUDA 12.4 미만 드라이버에서 `Minimum supported CUDA version: 12.4` 오류를 냄).")
code("""
!pip -q install uv
!cd {WORK} && (test -d microduck_rl || git clone -b {BRANCH} {REPO} microduck_rl)
if COMMIT:
    !cd {WORK}/microduck_rl && git checkout -q {COMMIT}
!cd {WORK}/microduck_rl && git log --oneline -1 && uv sync 2>&1 | tail -3
""")
code("""
# GPU 인식 확인 (torch + warp)
!cd {WORK}/microduck_rl && uv run --no-sync python -c "import torch,warp as wp; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); wp.init(); print('warp devices', wp.get_devices())"
""")
code("""
import shlex, time
env = dict(os.environ, WANDB_MODE="disabled", CUDA_VISIBLE_DEVICES="0")
if MODE == "velstand_resume":
    task = "Mjlab-VelStand-Rough-Backlash-MicroDuck"
    init = RESUME_FROM or "hf://pollen-robotics/microduck-training/bundles/velstand/0045i2si_2749"
    extra = ["--init-checkpoint", init]
elif MODE == "velocity_flat":
    task = "Mjlab-Velocity-Flat-MicroDuck"
    extra = ["--init-checkpoint", RESUME_FROM] if RESUME_FROM else []
else:
    raise ValueError(MODE)
cmd = ["uv", "run", "--no-sync", "train", task,
       "--gpu-ids", "0",
       "--agent.logger", "tensorboard",
       "--env.scene.num-envs", str(NUM_ENVS),
       "--agent.max-iterations", str(MAX_ITERATIONS),
       "--agent.save-interval", str(SAVE_INTERVAL),
       "--agent.run-name", f"free_{MODE}"] + extra
print(" ".join(shlex.quote(c) for c in cmd))
log = open(OUT / "train.log", "w")
t0 = time.time()
p = subprocess.Popen(cmd, cwd=WORK / "microduck_rl", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
try:
    for line in p.stdout:
        log.write(line); log.flush()
        if any(k in line for k in ("Learning iteration", "Steps per second", "Mean reward", "Iteration time", "Error", "Traceback")):
            print(line.rstrip())
        if time.time() - t0 > TIME_BUDGET_H * 3600:
            print("time budget reached, stopping"); p.terminate(); break
finally:
    p.wait(); log.close()
print("exit", p.returncode, "elapsed h", round((time.time() - t0) / 3600, 2))
""")
code("""
# 최신 체크포인트를 ONNX로 내보내고 출력 폴더에 복사
import glob, re, shutil
runs = sorted(glob.glob(str(WORK / "microduck_rl/logs/rsl_rl/*/*")), key=os.path.getmtime)
run = runs[-1]; print("run dir:", run)
pts = sorted(glob.glob(run + "/model_*.pt"), key=lambda f: int(re.findall(r"model_(\\d+)", f)[0]))
last = pts[-1]; print("last checkpoint:", last)
onnx = OUT / (pathlib.Path(run).name + "_" + pathlib.Path(last).stem + ".onnx")
subprocess.run(["uv", "run", "--no-sync", "python", "scripts/export.py", task,
                "--checkpoint-file", last, "--onnx-file", str(onnx)],
               cwd=WORK / "microduck_rl", env=env, check=True)
shutil.copy(last, OUT)
for f in glob.glob(run + "/events.out.tfevents*") + glob.glob(run + "/params/*"):
    shutil.copy(f, OUT)
!ls -la {OUT}
""")
md("""
결과는 출력 폴더(Kaggle: Output 탭 `/kaggle/working/out`, Colab: 왼쪽 파일 패널 `/content/out`)에 남아요. Kaggle은 **Save Version → Save & Run All (Commit)** 으로 돌리면 브라우저를 닫아도 끝까지 돌고 출력이 보존돼요. 다음 세션에서는 이 출력(`model_N.pt`)을 데이터셋으로 붙이고 `RESUME_FROM`에 경로를 넣으면 이어서 학습돼요.
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"gpuType": "T4"},
      "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("microduck_kaggle.ipynb", "w"), ensure_ascii=False, indent=1)
print("ok")
