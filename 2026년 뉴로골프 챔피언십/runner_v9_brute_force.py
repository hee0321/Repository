import os
import time
import zipfile
import onnx
from train_v2 import train_on_task_v2
from export_onnx_v2 import export_to_onnx_v2
from concurrent.futures import ProcessPoolExecutor, as_completed

MISSING_TASKS = [9, 13, 51, 84, 107, 108, 112, 131, 159, 170, 185, 209, 218, 221, 231, 240, 269, 280, 292, 310, 328, 349, 376, 383, 398]

def update_zip(output_zip="neurogolf-2026.zip"):
    """Rebuild the zip with all existing ONNX files (weights embedded)."""
    onnx_files = sorted([f for f in os.listdir(".") if f.startswith("task") and f.endswith(".onnx") and not f.endswith(".data")])
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in onnx_files:
            # We need to ensure weights are embedded for each file we add
            # But they should already be embedded if they were exported correctly.
            # To be safe, we don't re-embed here (slow), just trust the export.
            zf.write(f, f)
    print(f"  [ZIP] Updated {output_zip} | Models: {len(onnx_files)}")

def brute_force_task(task_id, data_dir="."):
    task_name = f"task{task_id:03d}.json"
    task_file = os.path.join(data_dir, task_name)
    onnx_path = os.path.join(data_dir, f"task{task_id:03d}.onnx")
    
    if not os.path.exists(task_file):
        return {"task": task_name, "success": False, "error": "File not found"}

    print(f"  [BRUTE] Starting {task_name} | dim=128, blocks=5, epochs=5000")
    
    # Aggressive training config
    res = train_on_task_v2(task_file, hidden_dim=128, num_blocks=5, max_epochs=5000, lr=0.001)
    
    if res['success']:
        pth_path = f"best_model_task{task_id:03d}.pth"
        try:
            export_to_onnx_v2(pth_path, onnx_path, hidden_dim=128)
            # Re-save with embedded weights for the zip
            model = onnx.load(onnx_path)
            onnx.save(model, onnx_path, save_as_external_data=False)
            
            if os.path.exists(pth_path):
                os.remove(pth_path)
            print(f"  [SUCCESS] {task_name} REFINED!")
            return {"task": task_name, "success": True}
        except Exception as e:
            return {"task": task_name, "success": False, "error": f"Export failed: {str(e)}"}
    else:
        print(f"  [STILL FAILED] {task_name} | Best Loss: {res['loss']:.6f}")
        return {"task": task_name, "success": False, "loss": res['loss']}

def run_brute_force(max_workers=4):
    print(f"\n{'#'*60}")
    print(f"  NEUROGOLF V9 - BRUTE FORCE 100% COMPLETION")
    print(f"  Target: {len(MISSING_TASKS)} stubborn tasks")
    print(f"  Config: Dim=128, Blocks=5, Epochs=5000")
    print(f"{'#'*60}\n")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(brute_force_task, tid): tid for tid in MISSING_TASKS}
        
        for future in as_completed(futures):
            tid = futures[future]
            res = future.result()
            if res['success']:
                update_zip()

if __name__ == "__main__":
    # Use fewer workers because dim=128 + 5 blocks is memory/CPU intensive
    run_brute_force(max_workers=4)
