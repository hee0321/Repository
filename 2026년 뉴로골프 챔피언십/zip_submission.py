import zipfile
import os
import glob

def create_submission_zip(output_zip="submission.zip", model_dir="."):
    # 1. Find all task ONNX files and their external data
    all_files = glob.glob(os.path.join(model_dir, "task*.*"))
    # Filter for .onnx and .onnx.data
    valid_files = [f for f in all_files if f.endswith('.onnx') or f.endswith('.onnx.data')]
    valid_files.sort()
    
    if not valid_files:
        print("No taskXXX files found to zip.")
        return
    
    # 2. Create the ZIP file
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in valid_files:
            arcname = os.path.basename(file)
            zipf.write(file, arcname)
            if arcname.endswith('.onnx'):
                print(f"Added model: {arcname}")
            
    print(f"\nSuccessfully created {output_zip} with {len([f for f in valid_files if f.endswith('.onnx')])} models.")
    
    # Check for missing tasks (total 400)
    task_ids = set()
    for f in valid_files:
        if not f.endswith('.onnx'): continue
        try:
            name = os.path.basename(f)
            tid = int(name.replace('task', '').replace('.onnx', ''))
            task_ids.add(tid)
        except:
            continue
            
    missing = [i for i in range(1, 401) if i not in task_ids]
    if missing:
        print(f"Warning: {len(missing)} tasks are missing from the zip (e.g., {missing[:5]}...)")
        print("Kaggle may require all 400 files for a full score.")

if __name__ == "__main__":
    create_submission_zip()
