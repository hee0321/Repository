import os
import glob
import pandas as pd
import torch
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

if sys.platform == "win32":
    # Ensure stdout/stderr use UTF-8 on Windows to avoid CP949 encoding errors with Unicode symbols
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass
from train_v2 import train_on_task_v2
from export_onnx_v2 import export_to_onnx_v2

def process_task_refine(task_file, data_dir, dim_list=[160], max_epochs=8000, num_blocks=6, kernel_size=5):
    """Refine training for failed tasks with Hyper-Titan configuration."""
    torch.set_num_threads(8) # 8 threads per worker for memory safety
    
    task_name = os.path.basename(task_file)
    onnx_path = os.path.join(data_dir, task_name.replace('.json', '.onnx'))
    
    # Double check if it was solved while waiting
    if os.path.exists(onnx_path):
        return {"task": task_name, "success": True, "onnname": onnx_path, "skipped": True}

    print(f"  [HYPER START] {task_name} | Dims: {dim_list} | Epochs: {max_epochs} | Blocks: {num_blocks} | K: {kernel_size}", flush=True)
    
    try:
        for dim in dim_list:
            result = train_on_task_v2(task_file, hidden_dim=dim, max_epochs=max_epochs, num_blocks=num_blocks, kernel_size=kernel_size)
            if result['success']:
                pth_path = f"best_model_{task_name.replace('.json', '')}.pth"
                export_to_onnx_v2(pth_path, onnx_path, hidden_dim=dim)
                result['task'] = task_name
                result['onnx_path'] = onnx_path
                if os.path.exists(pth_path):
                    os.remove(pth_path)
                print(f"  [REFINE SUCCESS] {task_name}", flush=True)
                return result
        
        print(f"  [REFINE FAILED] {task_name}", flush=True)
        return {"task": task_name, "success": False, "error": "Deep refinement failed"}
    except Exception as e:
        print(f"  [REFINE ERROR] {task_name}: {str(e)}", flush=True)
        return {"task": task_name, "success": False, "error": str(e)}

def run_v3_refine(data_dir=".", max_workers=2):
    """Run Phase 7: Hyper-Titan Refinement for failed tasks."""
    failed_list_path = os.path.join(data_dir, "failed_tasks.txt")
    if not os.path.exists(failed_list_path):
        print("failed_tasks.txt not found.")
        return
        
    # Use utf-8-sig to handle possible BOM
    task_files = []
    if os.path.exists(failed_list_path):
        with open(failed_list_path, 'r', encoding='utf-8-sig') as f:
            task_files = [line.strip() for line in f if line.strip()]
    
    if not task_files:
        print("No failed tasks to refine.")
        return
    
    print(f"\n{'#'*60}")
    print(f"  NeuroGolf V7 HYPER-TITAN MODE")
    print(f"  Focusing on {len(task_files)} failed tasks")
    print(f"  Workers: {max_workers} | Epochs: 8000 | Dim: 160 | Blocks: 6 | K: 5")
    print(f"{'#'*60}\n", flush=True)
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_task_refine, os.path.join(data_dir, tf), data_dir): tf for tf in task_files}
        
        for i, future in enumerate(as_completed(futures)):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                import traceback
                error_msg = f"WORKER CRASH on task {futures[future]}: {str(e)}\n{traceback.format_exc()}"
                print(f"  [CRASH] {futures[future]}")
                with open("runner_refine_error.log", "a") as log:
                    log.write(error_msg + "\n")
                continue
            
            # Progress check
            current_onnx = len(glob.glob(os.path.join(data_dir, "task*.onnx")))
            if (i + 1) % 5 == 0:
                print(f"\n>>> REFINEMENT PROGRESS: {i+1}/{len(task_files)} checked | Total ONNX: {current_onnx}/400 <<<\n", flush=True)
    
    # Save report
    df = pd.DataFrame(results)
    df.to_csv("competition_report_v3_refine.csv", index=False)
    
    final_count = len(glob.glob(os.path.join(data_dir, "task*.onnx")))
    print(f"\n{'#'*60}")
    print(f"  REFINEMENT FINISHED: {final_count}/400 ONNX models ready")
    print(f"{'#'*60}")

if __name__ == "__main__":
    try:
        run_v3_refine()
    except Exception as e:
        import traceback
        with open("runner_refine_error.log", "a") as f:
            f.write(f"REFINEMENT RUNNER FATAL ERROR: {str(e)}\n")
            f.write(traceback.format_exc() + "\n")
        print("Refinement runner fatal error! Check runner_refine_error.log")
