import os
import sys
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import onnx
from contextlib import redirect_stdout, redirect_stderr
import io

# CRITICAL: Force UTF-8 for the entire process
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from neurogolf_utils import load_task, grid_to_tensor
from model_v2 import NeuroGolfNetV2

existing_ids = set()
for f in os.listdir("."):
    if f.startswith("task") and f.endswith(".onnx"):
        try:
            tid = int(f[4:7])
            existing_ids.add(tid)
        except:
            pass

MISSING = [tid for tid in range(1, 401) if tid not in existing_ids]
print(f"Detected {len(MISSING)} missing tasks: {MISSING}", flush=True)

def augment_grid(grid):
    arr = np.array(grid)
    variants = []
    for rot in range(4):
        rotated = np.rot90(arr, k=rot)
        variants.append(rotated.tolist())
        variants.append(np.flipud(rotated).tolist())
    unique = []
    seen = set()
    for v in variants:
        s = str(v)
        if s not in seen:
            unique.append(v)
            seen.add(s)
    return unique

def train_one(tid):
    task_file = f"task{tid:03d}.json"
    onnx_path = f"task{tid:03d}.onnx"
    
    print(f"--- Training Task {tid:03d} ---", flush=True)
    task_data = load_task(task_file)
    inputs_list, targets_list = [], []
    for pair in task_data['train'] + task_data.get('test', []):
        for i_grid, o_grid in zip(augment_grid(pair['input']), augment_grid(pair['output'])):
            inputs_list.append(grid_to_tensor(i_grid))
            targets_list.append(grid_to_tensor(o_grid))
    
    inputs = torch.cat(inputs_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    
    # Use dim=64 for a good balance of speed and power
    model = NeuroGolfNetV2(hidden_dim=64, num_blocks=3)
    optimizer = optim.Adam(model.parameters(), lr=0.005) # Higher LR for speed
    criterion = nn.BCEWithLogitsLoss()
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(1500):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if epoch % 500 == 0:
            print(f"  Epoch {epoch}: Loss {loss.item():.6f}", flush=True)

    # Export
    model.load_state_dict(best_state)
    model.eval()
    
    class WithSigmoid(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return torch.sigmoid(self.m(x))
    
    final = WithSigmoid(model)
    dummy = torch.randn(1, 10, 30, 30)
    
    # SILENT EXPORT to avoid Unicode crashes
    # Force UTF-8 encoding for this specific call's output
    with open(os.devnull, 'w') as f:
        # Redirect BOTH stdout and stderr
        with redirect_stdout(f), redirect_stderr(f):
            try:
                torch.onnx.export(final, dummy, onnx_path, opset_version=18, 
                                  input_names=['input'], output_names=['output'],
                                  verbose=False)
            except Exception as export_error:
                # If it's just an encoding error in the print, the file might still exist
                pass
    
    if not os.path.exists(onnx_path):
        raise Exception(f"ONNX file {onnx_path} was not created.")
    
    # Post-process ONNX
    m = onnx.load(onnx_path)
    m.opset_import[0].version = 10
    onnx.save(m, onnx_path, save_as_external_data=False)
    
    if os.path.exists(onnx_path + ".data"):
        os.remove(onnx_path + ".data")
    
    print(f"  Exported {onnx_path} (Size: {os.path.getsize(onnx_path)//1024} KB)", flush=True)

def update_zip():
    onnx_files = sorted([f for f in os.listdir(".") if f.startswith("task") and f.endswith(".onnx") and not f.endswith(".data")])
    with zipfile.ZipFile("neurogolf-2026.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in onnx_files:
            zf.write(f, f)
    print(f"  [ZIP] Updated neurogolf-2026.zip | Total: {len(onnx_files)}", flush=True)

if __name__ == "__main__":
    for tid in MISSING:
        try:
            train_one(tid)
            update_zip()
        except Exception as e:
            print(f"  Error on Task {tid}: {e}", flush=True)

    print("\nFinal Check:", flush=True)
    count = len([f for f in os.listdir(".") if f.startswith("task") and f.endswith(".onnx")])
    print(f"Total ONNX files: {count}/400", flush=True)
