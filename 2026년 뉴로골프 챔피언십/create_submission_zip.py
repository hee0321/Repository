"""
Create neurogolf-2026.zip for Kaggle submission.
- Embeds weights inside each .onnx file (no external .data files)
- Only includes task*.onnx files
"""
import zipfile
import os
import glob
import onnx
import tempfile
import shutil

def create_submission():
    output_zip = "neurogolf-2026.zip"
    onnx_files = sorted(glob.glob("task*.onnx"))
    onnx_files = [f for f in onnx_files if not f.endswith(".onnx.data")]
    
    print(f"Found {len(onnx_files)} ONNX models")
    
    # Create temp dir for re-saved models
    tmp_dir = os.path.join(".", "_tmp_submission")
    os.makedirs(tmp_dir, exist_ok=True)
    
    try:
        embedded_files = []
        total_size = 0
        
        for i, onnx_file in enumerate(onnx_files):
            try:
                model = onnx.load(onnx_file)
                out_path = os.path.join(tmp_dir, os.path.basename(onnx_file))
                # Save with weights embedded (no external data)
                onnx.save(model, out_path, save_as_external_data=False)
                fsize = os.path.getsize(out_path)
                total_size += fsize
                embedded_files.append(out_path)
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i+1}/{len(onnx_files)} | Running total: {total_size / 1024 / 1024:.1f} MB")
            except Exception as e:
                print(f"  [ERROR] {onnx_file}: {e}")
        
        print(f"\nTotal embedded size: {total_size / 1024 / 1024:.1f} MB")
        print(f"Creating {output_zip}...")
        
        # Create ZIP
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fpath in embedded_files:
                arcname = os.path.basename(fpath)
                zf.write(fpath, arcname)
        
        zip_size = os.path.getsize(output_zip)
        print(f"\n{'='*50}")
        print(f"  neurogolf-2026.zip created!")
        print(f"  Models: {len(embedded_files)}/400")
        print(f"  ZIP size: {zip_size / 1024 / 1024:.1f} MB")
        print(f"{'='*50}")
        
        # Report missing tasks
        task_ids = set()
        for f in embedded_files:
            try:
                tid = int(os.path.basename(f).replace("task", "").replace(".onnx", ""))
                task_ids.add(tid)
            except:
                pass
        missing = sorted([i for i in range(1, 401) if i not in task_ids])
        if missing:
            print(f"\n  Missing {len(missing)} tasks: {missing}")
            
    finally:
        # Cleanup temp dir
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    create_submission()
