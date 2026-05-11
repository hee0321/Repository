import os
import zipfile

onnx_path = "test_tiny.onnx"
if not os.path.exists(onnx_path):
    print("Run check_size.py first.")
    exit()

zip_path = "test_tiny.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for i in range(400):
        zf.writestr(f"task{i:03d}.onnx", open(onnx_path, 'rb').read())

print(f"Total size of 400 models in ZIP: {os.path.getsize(zip_path)} bytes")
print(f"Target: 1474560 bytes")
