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
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- CONFIGURATION ---
HIDDEN_DIM = 12
MAX_EPOCHS = 100 # Reduced significantly for speed
LEARNING_RATE = 0.01
TARGET_ZIP = "neurogolf-2026.zip"

# Force UTF-8 for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from neurogolf_utils import load_task, grid_to_tensor

# --- MODEL ---
class NeuroGolfNetV1(nn.Module):
    def __init__(self, hidden_dim=12):
        super(NeuroGolfNetV1, self).__init__()
        self.conv1 = nn.Conv2d(10, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, 10, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return torch.sigmoid(self.conv3(x))

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

def process_task(tid):
    task_file = f"task{tid:03d}.json"
    onnx_path = f"task{tid:03d}.onnx"
    if not os.path.exists(task_file): return False
    
    try:
        task_data = load_task(task_file)
        all_pairs = task_data['train']
        for pair in task_data.get('test', []):
            if 'output' in pair: all_pairs.append(pair)
            
        inputs_list, targets_list = [], []
        for pair in all_pairs:
            for i, o in zip(augment_grid(pair['input']), augment_grid(pair['output'])):
                inputs_list.append(grid_to_tensor(i))
                targets_list.append(grid_to_tensor(o))
        
        inputs = torch.cat(inputs_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
        
        model = NeuroGolfNetV1(hidden_dim=HIDDEN_DIM)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        best_loss = float('inf')
        best_state = model.state_dict()
        
        for epoch in range(MAX_EPOCHS):
            model.train()
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if best_loss < 1e-4: break
            
        model.load_state_dict(best_state)
        model.eval()
        dummy = torch.randn(1, 10, 30, 30)
        
        # Use Opset 17 for better compatibility with current PyTorch on Windows
        # and manually downgrade header if needed (but 17 is usually fine)
        torch.onnx.export(model, dummy, onnx_path, opset_version=17, do_constant_folding=True)
        
        # Embed weights
        m = onnx.load(onnx_path)
        onnx.save(m, onnx_path, save_as_external_data=False)
        return True
    except Exception as e:
        # Don't print full traceback in worker
        return False

if __name__ == "__main__":
    print("=== NEUROGOLF BEST ENTRY (ULTRA-FAST MODE) ===")
    
    # We DON'T clean up everything to allow resuming if possible, 
    # but for a "Best Entry" we should probably ensure all are tiny.
    # We'll overwrite existing task*.onnx anyway.
            
    print("Processing 400 tasks in parallel (Max Workers: 12)...")
    start_time = time.time()
    
    results = []
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(process_task, tid): tid for tid in range(1, 401)}
        for i, future in enumerate(as_completed(futures)):
            tid = futures[future]
            try:
                success = future.result()
                if success: results.append(tid)
            except:
                pass
            
            if (i+1) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {i+1}/400 | Success: {len(results)} | Elapsed: {elapsed:.1f}s")
            
    print("Zipping results...")
    onnx_files = sorted([f for f in os.listdir(".") if f.startswith("task") and f.endswith(".onnx")])
    with zipfile.ZipFile(TARGET_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in onnx_files:
            zf.write(f, f)
            
    final_size = os.path.getsize(TARGET_ZIP)
    print(f"DONE! Final ZIP: {TARGET_ZIP} ({final_size/1024:.1f} KB)")
    print(f"Total Time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Models in ZIP: {len(onnx_files)}/400")
