# Microduck RL — free-only experiments (2026-10-28, box CPU, repo develop @ bae39f5)

## CPU results (8-core Xeon, no CUDA; warp CPU kernels run on ~1 core)
| run | envs | iters | steps/s | iter time | log |
|---|---|---|---|---|---|
| Velocity-Flat from scratch | 64 | 5 | 177-209 | 7.3-8.7 s | cpu_runs/velocity_cpu64.log |
| Velocity-Flat from scratch | 256 | 3 | 235-257 | 23.8-26.1 s | cpu_runs/velocity_cpu256.log |
| Velocity-Flat from scratch | 1024 | 2 | 244-248 | 98.8-100.7 s | cpu_runs/velocity_cpu1024.log |
| VelStand-Rough-Backlash resume from hf v8 (0045i2si_2749) | 64 | 3 (2749->2751) | 57-58 | 26.1-26.9 s | cpu_runs/velstand_resume_v8_cpu64.log |

- mjlab picks device=cpu automatically when CUDA_VISIBLE_DEVICES is empty (`--gpu-ids None`).
- Resume on CPU fails with stock `train` (mjlab runner torch.load map_location=None on a CUDA-saved checkpoint);
  `train_cpu.py` here patches map_location="cpu" without touching the repo.
- ONNX export on CPU works: cpu_runs/velstand_cpu_resume_2751.onnx.
- Estimates at measured throughput: Pollen-scale walk (4096 envs x 24 steps x 3750 iters = 368.6M steps) at ~245 steps/s ≈ 17 days;
  velstand +1000 iters (98.3M steps) at ~57 steps/s ≈ 20 days.

## Free GPU
microduck_kaggle.ipynb — Kaggle (T4 x2, Internet on) or Colab (T4).
