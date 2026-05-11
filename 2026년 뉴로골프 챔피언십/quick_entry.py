import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import onnx
import zipfile
import io
import time

# --- HYPER-FAST CONFIG ---
HIDDEN_DIM = 8
MAX_EPOCHS = 10 
TARGET_ZIP = "neurogolf-2026.zip"

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from neurogolf_utils import load_task, grid_to_tensor

class NeuroGolfNetV1(nn.Module):
    def __init__(self, hidden_dim=8):
        super().__init__()
        self.conv1 = nn.Conv2d(10, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, 10, kernel_size=3, padding=1)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return torch.sigmoid(self.conv3(x))

def quick_process(tid):
    task_file = f"task{tid:03d}.json"
    onnx_path = f"task{tid:03d}.onnx"
    if not os.path.exists(task_file): return False
    try:
        task_data = load_task(task_file)
        inputs = torch.cat([grid_to_tensor(p['input']) for p in task_data['train']], dim=0)
        targets = torch.cat([grid_to_tensor(p['output']) for p in task_data['train']], dim=0)
        model = NeuroGolfNetV1(hidden_dim=HIDDEN_DIM)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        for _ in range(MAX_EPOCHS):
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
        model.eval()
        dummy = torch.randn(1, 10, 30, 30)
        torch.onnx.export(model, dummy, onnx_path, opset_version=11, do_constant_folding=True)
        return True
    except: return False

if __name__ == "__main__":
    print("=== GENERATING QUICK VALID SUBMISSION ===")
    start_time = time.time()
    for tid in range(1, 401):
        quick_process(tid)
        if tid % 50 == 0:
            print(f"Progress: {tid}/400 | {time.time()-start_time:.1f}s")
            
    onnx_files = sorted([f for f in os.listdir(".") if f.startswith("task") and f.endswith(".onnx")])
    with zipfile.ZipFile(TARGET_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in onnx_files: zf.write(f, f)
    print(f"DONE! {TARGET_ZIP} ({os.path.getsize(TARGET_ZIP)/1024:.1f} KB)")
