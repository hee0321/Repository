import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import onnx
import zipfile
import io
import time

# --- CONFIGURATION ---
HIDDEN_DIM = 4
MAX_EPOCHS = 10
LEARNING_RATE = 0.01
TARGET_ZIP = "neurogolf-2026.zip"

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from neurogolf_utils import load_task, grid_to_tensor

class NeuroGolfNetV1(nn.Module):
    def __init__(self, hidden_dim=4):
        super(NeuroGolfNetV1, self).__init__()
        self.conv1 = nn.Conv2d(10, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, 10, kernel_size=3, padding=1)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return torch.sigmoid(self.conv3(x))

def process_single_task(tid):
    task_file = f"task{tid:03d}.json"
    onnx_path = f"task{tid:03d}.onnx"
    if not os.path.exists(task_file): return False
    try:
        task_data = load_task(task_file)
        inputs = torch.cat([grid_to_tensor(p['input']) for p in task_data['train']], dim=0)
        targets = torch.cat([grid_to_tensor(p['output']) for p in task_data['train']], dim=0)
        model = NeuroGolfNetV1(hidden_dim=HIDDEN_DIM)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        for epoch in range(MAX_EPOCHS):
            model.train()
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
        model.eval()
        dummy = torch.randn(1, 10, 30, 30)
        torch.onnx.export(model, dummy, onnx_path, opset_version=10)
        return True
    except Exception as e:
        print(f"Error {tid}: {e}")
        return False

if __name__ == "__main__":
    print("Starting fast test...")
    for tid in range(1, 11):
        success = process_single_task(tid)
        print(f"Task {tid}: {'SUCCESS' if success else 'FAILED'}")
    print("Done.")
