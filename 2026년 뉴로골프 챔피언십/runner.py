import os
import glob
import pandas as pd
from train import train_on_task
from concurrent.futures import ProcessPoolExecutor, as_completed
from export_onnx import export_to_onnx

def process_task(task_file, data_dir, hidden_dim_list=[8, 16]):
    task_name = os.path.basename(task_file)
    onnx_path = os.path.join(data_dir, task_name.replace('.json', '.onnx'))
    
    pth_path = f"best_model_{task_name.replace('.json', '')}.pth"
    if os.path.exists(onnx_path) or os.path.exists(pth_path):
        return {"task": task_name, "success": True, "onnx_path": onnx_path, "skipped": True}

    print(f"Starting {task_name}...")
    
    try:
        result = {"success": False}
        for dim in hidden_dim_list:
            # Try with reduced epochs for speed in first pass if possible? 
            # No, let's stick to a reasonable default.
            result = train_on_task(task_file, hidden_dim=dim, max_epochs=1500)
            if result['success']:
                break
            print(f"Task {task_name} failed with dim {dim}, retrying...")

        result['task'] = task_name
        
        if result['success']:
            success_dim = result.get('hidden_dim', 8)
            export_to_onnx(f"best_model_{task_name.replace('.json', '')}.pth", onnx_path, hidden_dim=success_dim)
            result['onnx_path'] = onnx_path
            # Cleanup .pth to save space
            pth_path = f"best_model_{task_name.replace('.json', '')}.pth"
            if os.path.exists(pth_path):
                os.remove(pth_path)
                
        return result
    except Exception as e:
        return {"task": task_name, "success": False, "error": str(e)}

def run_all_tasks_parallel(data_dir=".", max_workers=8):
    task_files = glob.glob(os.path.join(data_dir, "task*.json"))
    task_files.sort()
    
    if not task_files:
        print(f"No task files found in {data_dir}.")
        return
    
    results = []
    print(f"Processing {len(task_files)} tasks sequentially...")
    
    for tf in task_files:
        res = process_task(tf, data_dir)
        results.append(res)
        status = "SUCCESS" if res['success'] else "FAILED"
        print(f"Finished {res['task']}: {status}")
            
    # Save Report
    df = pd.DataFrame(results)
    df.to_csv("competition_report.csv", index=False)
    
    success_count = df['success'].sum()
    print(f"\nSummary: {success_count}/{len(df)} tasks solved.")

if __name__ == "__main__":
    # Use half of the cores to keep system responsive
    run_all_tasks_parallel(max_workers=8)
