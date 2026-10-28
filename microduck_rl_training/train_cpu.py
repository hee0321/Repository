"""CPU wrapper around `uv run train`: maps CUDA-saved checkpoints to CPU.

mjlab's MjlabOnPolicyRunner.load() calls torch.load(map_location=None), which fails on a
CPU-only box for checkpoints saved on GPU (all Pollen Hub bundles). This patches the default
map_location to "cpu" when CUDA is unavailable, without touching the repo.
Usage: uv run --project /workspace/microduck_rl python /workspace/microduck_rl_free/train_cpu.py <TASK> [train args]
"""
import sys

sys.argv[0] = "train"  # train_hook only consumes --init-checkpoint when argv[0] is `train`

import torch
from mjlab.rl.runner import MjlabOnPolicyRunner

if not torch.cuda.is_available():
    _orig = MjlabOnPolicyRunner.load

    def _load_cpu(self, path, load_cfg=None, strict=True, map_location=None):
        return _orig(self, path, load_cfg=load_cfg, strict=strict, map_location=map_location or "cpu")

    MjlabOnPolicyRunner.load = _load_cpu

import mjlab_microduck.tasks.mdp  # noqa: F401  (installs warm-start patch on top of ours)
from mjlab_microduck.train_cli import main

sys.exit(main())
