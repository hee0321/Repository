"""Convert all existing .pth files to .onnx that are missing."""
import glob
import os
from export_onnx_v2 import export_to_onnx_v2
from export_onnx import export_to_onnx

for pth in sorted(glob.glob("best_model_task*.pth")):
    task_name = pth.replace("best_model_", "").replace(".pth", "")
    onnx_path = f"{task_name}.onnx"
    
    if os.path.exists(onnx_path):
        print(f"SKIP {onnx_path} (already exists)")
        continue
    
    print(f"Converting {pth} -> {onnx_path}")
    try:
        # Try V2 export first
        export_to_onnx_v2(pth, onnx_path)
        print(f"  OK (V2)")
    except Exception as e1:
        try:
            # Fallback to V1 export
            export_to_onnx(pth, onnx_path)
            print(f"  OK (V1)")
        except Exception as e2:
            print(f"  FAILED: V2={e1}, V1={e2}")

total = len(glob.glob("task*.onnx"))
print(f"\nTotal ONNX files: {total}/400")
