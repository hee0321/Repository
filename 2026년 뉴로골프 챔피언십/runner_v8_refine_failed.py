import os
import glob
import pandas as pd
from train_v2 import train_on_task_v2
from export_onnx_v2 import export_to_onnx_v2
from concurrent.futures import ProcessPoolExecutor, as_completed

def refine_failed_tasks(failed_file="failed_tasks.txt", data_dir=".", max_workers=8):
    if not os.path.exists(failed_file):
        print(f"Error: {failed_file} not found.")
        return

    with open(failed_file, "r") as f:
        failed_tasks = [line.strip() for line in f if line.strip()]

    print(f"\n{'='*60}")
    print(f"  NeuroGolf V8 - REFINEMENT RUN")
    print(f"  Targeting {len(failed_tasks)} failed tasks")
    print(f"  Strategy: dim=64, epochs=2000, lr=0.003")
    print(f"{'='*60}\n")

    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for task_name in failed_tasks:
            task_file = os.path.join(data_dir, task_name)
            if not os.path.exists(task_file):
                print(f"  [SKIP] {task_name} (File not found)")
                continue
            
            # Use more powerful config for refinement
            futures[executor.submit(train_on_task_v2, task_file, hidden_dim=64, max_epochs=2000, lr=0.003)] = task_name

        for i, future in enumerate(as_completed(futures)):
            task_name = futures[future]
            try:
                res = future.result()
                res['task'] = task_name
                
                if res['success']:
                    onnx_path = os.path.join(data_dir, task_name.replace('.json', '.onnx'))
                    pth_path = f"best_model_{task_name.replace('.json', '')}.pth"
                    export_to_onnx_v2(pth_path, onnx_path, hidden_dim=64)
                    if os.path.exists(pth_path):
                        os.remove(pth_path)
                    print(f"  [REFINED] {task_name} - SUCCESS")
                else:
                    print(f"  [STILL FAILED] {task_name} - Best Loss: {res['loss']:.6f}")
                
                results.append(res)
            except Exception as e:
                print(f"  [CRASH] {task_name}: {str(e)}")
                results.append({"task": task_name, "success": False, "error": str(e)})

    # Append to existing report if possible
    report_file = "competition_report_v8_refine.csv"
    pd.DataFrame(results).to_csv(report_file, index=False)
    print(f"\nRefinement complete. Results saved to {report_file}")

if __name__ == "__main__":
    refine_failed_tasks()
