import os
import glob
import pandas as pd
import torch
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

from train_v2 import train_on_task_v2
from export_onnx_v2 import export_to_onnx_v2

def process_task_lite(task_file, data_dir, hidden_dim=64, max_epochs=5000, num_blocks=3, kernel_size=3):
    """Retrain and export for 1.44MB compliance."""
    torch.set_num_threads(4) # More workers, fewer threads
    
    task_name = os.path.basename(task_file)
    onnx_path = os.path.join(data_dir, task_name.replace('.json', '.onnx'))
    
    print(f"  [LITE START] {task_name} | Dim: {hidden_dim} | Blocks: {num_blocks}", flush=True)
    
    try:
        # We ALWAYS retrain to ensure we have the correct architecture and embedded weights
        result = train_on_task_v2(task_file, hidden_dim=hidden_dim, max_epochs=max_epochs, num_blocks=num_blocks, kernel_size=kernel_size)
        
        pth_path = f"best_model_{task_name.replace('.json', '')}.pth"
        
        # If success, export normally
        # If failed, we export anyway (it might still get partial credit if loss is low)
        export_to_onnx_v2(pth_path, onnx_path, hidden_dim=hidden_dim)
        
        result['task'] = task_name
        result['onnx_path'] = onnx_path
        
        if os.path.exists(pth_path):
            os.remove(pth_path)
            
        status = "SUCCESS" if result['success'] else "PARTIAL (Low Loss)"
        print(f"  [LITE {status}] {task_name}", flush=True)
        return result
        
    except Exception as e:
        print(f"  [LITE ERROR] {task_name}: {str(e)}", flush=True)
        return {"task": task_name, "success": False, "error": str(e)}

def run_v4_lite(data_dir=".", max_workers=4):
    """Run Phase 8: Lite-Titan (1.44MB Compliant) for ALL tasks."""
    # Find ALL task json files
    task_files = sorted(glob.glob(os.path.join(data_dir, "task*.json")))
    
    if not task_files:
        print("No task files found.")
        return
    
    print(f"\n{'#'*60}")
    print(f"  NeuroGolf V8 LITE-TITAN MODE (1.44MB COMPLIANT)")
    print(f"  Processing {len(task_files)} tasks")
    print(f"  Workers: {max_workers} | Epochs: 5000 | Dim: 64 | Blocks: 3 | K: 3")
    print(f"{'#'*60}\n", flush=True)
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_task_lite, tf, data_dir): tf for tf in task_files}
        
        for i, future in enumerate(as_completed(futures)):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"  [CRASH] {futures[future]}: {str(e)}")
                continue
            
            if (i + 1) % 10 == 0:
                print(f"\n>>> LITE PROGRESS: {i+1}/{len(task_files)} processed <<<\n", flush=True)
    
    # Save report
    df = pd.DataFrame(results)
    df.to_csv("competition_report_v4_lite.csv", index=False)
    
    print(f"\n{'#'*60}")
    print(f"  LITE FINISHED: {len(results)}/400 models exported")
    print(f"{'#'*60}")

if __name__ == "__main__":
    run_v4_lite(max_workers=8) # Use more workers for smaller models
