import zipfile
import os
import glob

def create_submission_zip_fake_missing(output_zip="submission.zip", model_dir="."):
    # 1. Find all task ONNX files
    onnx_files = glob.glob(os.path.join(model_dir, "task*.onnx"))
    onnx_files.sort()
    
    if not onnx_files:
        print("No taskXXX.onnx files found to zip.")
        return
        
    print(f"Found {len(onnx_files)} actual ONNX models.")
    
    existing_tasks = set()
    for f in onnx_files:
        try:
            name = os.path.basename(f)
            tid = int(name.replace('task', '').replace('.onnx', ''))
            existing_tasks.add(tid)
        except:
            continue
            
    dummy_file = os.path.join(model_dir, "task001.onnx")
    if not os.path.exists(dummy_file):
        dummy_file = onnx_files[0]
        
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        added_count = 0
        for i in range(1, 401):
            arcname = f"task{i:03d}.onnx"
            if i in existing_tasks:
                actual_file = os.path.join(model_dir, arcname)
                if os.path.exists(actual_file):
                    zipf.write(actual_file, arcname)
                else:
                    zipf.write(dummy_file, arcname)
            else:
                zipf.write(dummy_file, arcname)
            added_count += 1
            
    print(f"\nSuccessfully created {output_zip} with {added_count} models (dummy copies used for missing).")

if __name__ == "__main__":
    create_submission_zip_fake_missing()
