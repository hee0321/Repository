import os
import glob
import csv
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from train_v2 import train_on_task_v2
from export_onnx_v2 import export_to_onnx_v2

def process_task_v2(task_file, data_dir, dim_list=[48], max_epochs=1500):
    """Process a single task with V2 model, optimized for speed."""
    # One thread per worker to leave 8 cores free for user
    torch.set_num_threads(1)
    
    task_name = os.path.basename(task_file)
    onnx_path = os.path.join(data_dir, task_name.replace('.json', '.onnx'))
    
    # Skip if already has ONNX
    if os.path.exists(onnx_path):
        return {"task": task_name, "success": True, "onnx_path": onnx_path, "skipped": True}

    print(f"  [START] {task_name}", flush=True)
    
    try:
        for dim in dim_list:
            result = train_on_task_v2(task_file, hidden_dim=dim, max_epochs=max_epochs)
            if result['success']:
                pth_path = f"best_model_{task_name.replace('.json', '')}.pth"
                export_to_onnx_v2(pth_path, onnx_path, hidden_dim=dim)
                result['task'] = task_name
                result['onnx_path'] = onnx_path
                if os.path.exists(pth_path):
                    os.remove(pth_path)
                print(f"  [SUCCESS] {task_name}", flush=True)
                return result
        
        print(f"  [FAILED] {task_name}", flush=True)
        return {"task": task_name, "success": False, "error": "Attempt failed"}
    except Exception as e:
        print(f"  [ERROR] {task_name}: {str(e)}", flush=True)
        return {"task": task_name, "success": False, "error": str(e)}

def run_v2_high_speed(data_dir=".", max_workers=8):
    """Run V2 training in HIGH SPEED MODE."""
    # Prioritize unattempted tasks (160-400) first to boost count quickly
    all_task_files = sorted(glob.glob(os.path.join(data_dir, "task*.json")))
    task_files = [f for f in all_task_files if int(os.path.basename(f).replace('task','').replace('.json','')) >= 160]
    task_files += [f for f in all_task_files if int(os.path.basename(f).replace('task','').replace('.json','')) < 160]
    
    if not task_files:
        print("No task files found.")
        return
    
    existing_onnx_count = len(glob.glob(os.path.join(data_dir, "task*.onnx")))
    
    print(f"\n{'#'*60}", flush=True)
    print(f"  NeuroGolf V2 HIGH SPEED RUNNER")
    print(f"  STRATEGY: Prioritizing Task 160-400")
    print(f"  Workers: {max_workers} | Already done: {existing_onnx_count} / {len(task_files)}")
    print(f"{'#'*60}\n", flush=True)
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_task_v2, tf, data_dir): tf for tf in task_files}
        
        for i, future in enumerate(as_completed(futures)):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                import traceback
                error_msg = f"WORKER CRASH on task {futures[future]}: {str(e)}\n{traceback.format_exc()}"
                print(f"  [CRASH] {futures[future]}")
                with open("runner_error.log", "a") as f:
                    f.write(error_msg + "\n")
                continue
            
            # Count current ONNX files
            current_onnx = len(glob.glob(os.path.join(data_dir, "task*.onnx")))
            
            # Milestone report
            if (i + 1) % 5 == 0:
                print(f"\n>>> MILESTONE: {i+1} tasks checked | {current_onnx}/400 ONNX ready <<<\n", flush=True)
    
    # Save report
    df = pd.DataFrame(results)
    df.to_csv("competition_report_v2_fast.csv", index=False)
    
    final_count = len(glob.glob(os.path.join(data_dir, "task*.onnx")))
    print(f"\n{'#'*60}")
    print(f"  FINAL RESULT: {final_count}/400 ONNX models ready")
    print(f"{'#'*60}")

if __name__ == "__main__":
    try:
        run_v2_high_speed()
    except Exception as e:
        import traceback
        with open("runner_error.log", "a") as f:
            f.write(f"MAIN RUNNER FATAL ERROR: {str(e)}\n")
            f.write(traceback.format_exc() + "\n")
        print("Main runner fatal error! Check runner_error.log")
