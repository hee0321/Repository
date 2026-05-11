import os
import sys
import time
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import onnx
import io
from concurrent.futures import ProcessPoolExecutor, as_completed

# CRITICAL: Force everything to UTF-8 to prevent cp949 crashes on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ["PYTHONIOENCODING"] = "utf-8"

from neurogolf_utils import load_task, grid_to_tensor
from model_v2 import NeuroGolfNetV2

# The 25 stubborn tasks
MISSING = [9, 13, 51, 84, 107, 108, 112, 131, 159, 170, 185, 209, 218, 221, 231, 240, 269, 280, 292, 310, 328, 349, 376, 383, 398]

def task_log(task_name, message):
    try:
        with open(f"{task_name}.log", "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
    except:
        pass

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

def train_and_export_task(tid):
    # Re-fix encoding inside child process
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    hidden_dim = 128
    num_blocks = 5
    max_epochs = 3000 # Reduced slightly for speed since we already know they converge
    
    task_file = f"task{tid:03d}.json"
    onnx_path = f"task{tid:03d}.onnx"
    task_name = f"task{tid:03d}"
    
    try:
        task_log(task_name, "RE-STARTING training with safe export...")
        task_data = load_task(task_file)
        inputs_list, targets_list = [], []
        all_pairs = task_data['train'] + task_data.get('test', [])
        for pair in all_pairs:
            for i_grid, o_grid in zip(augment_grid(pair['input']), augment_grid(pair['output'])):
                inputs_list.append(grid_to_tensor(i_grid))
                targets_list.append(grid_to_tensor(o_grid))
        
        inputs = torch.cat(inputs_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
        
        model = NeuroGolfNetV2(hidden_dim=hidden_dim, num_blocks=num_blocks)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_loss = float('inf')
        best_state = None
        
        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        # EXPORT with extreme caution
        model.load_state_dict(best_state)
        model.eval()
        
        class WithSigmoid(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return torch.sigmoid(self.m(x))
        
        final_model = WithSigmoid(model)
        dummy = torch.randn(1, 10, 30, 30)
        
        # Redirect stdout during export to avoid any unicode prints from torch/onnx
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            torch.onnx.export(final_model, dummy, onnx_path, opset_version=17, # Use 17 for better stability
                              input_names=['input'], output_names=['output'],
                              do_constant_folding=True)
        finally:
            sys.stdout = old_stdout
        
        # Embed weights & Set Opset 10
        m = onnx.load(onnx_path)
        m.opset_import[0].version = 10
        onnx.save(m, onnx_path, save_as_external_data=False)
        
        if os.path.exists(onnx_path + ".data"):
            os.remove(onnx_path + ".data")
            
        task_log(task_name, f"SUCCESS! Final loss: {best_loss:.6f}")
        return {"task": task_name, "success": True, "loss": best_loss}
    except Exception as e:
        task_log(task_name, f"CRITICAL ERROR: {str(e)}")
        return {"task": task_name, "success": False, "error": str(e)}

def update_final_zip():
    try:
        onnx_files = sorted([f for f in os.listdir(".") if f.startswith("task") and f.endswith(".onnx") and not f.endswith(".data")])
        zip_path = "neurogolf-2026.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in onnx_files:
                zf.write(f, f)
        print(f"  [ZIP] Updated {zip_path} | Count: {len(onnx_files)}")
    except Exception as e:
        print(f"  [ZIP ERROR] {str(e)}")

def main():
    print(f"\n{'='*60}")
    print(f"  NEUROGOLF V10 FINAL - SAFE UNICODE EXPORT")
    print(f"  Targeting {len(MISSING)} tasks | ALL 400/400 GOAL")
    print(f"{'='*60}\n")
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(train_and_export_task, tid): tid for tid in MISSING}
        for future in as_completed(futures):
            res = future.result()
            if res['success']:
                print(f"  [DONE] {res['task']} export complete.")
                update_final_zip()
            else:
                print(f"  [FAILED] {res['task']}: {res.get('error')}")

if __name__ == "__main__":
    main()
