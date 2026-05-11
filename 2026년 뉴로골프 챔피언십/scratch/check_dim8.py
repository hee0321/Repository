import torch
import torch.nn as nn
import onnx
import os
import sys
import io
import zipfile

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

class TinyGolfNet(nn.Module):
    def __init__(self, hidden_dim=8):
        super(TinyGolfNet, self).__init__()
        self.conv1 = nn.Conv2d(10, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, 10, kernel_size=3, padding=1)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return torch.sigmoid(self.conv3(x))

hidden_dim = 8
model = TinyGolfNet(hidden_dim=hidden_dim)
dummy = torch.randn(1, 10, 30, 30)
onnx_path = f"test_dim{hidden_dim}.onnx"
torch.onnx.export(model, dummy, onnx_path, opset_version=10)
size = os.path.getsize(onnx_path)
print(f"Single model size (dim {hidden_dim}): {size} bytes")

zip_path = f"test_dim{hidden_dim}.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for i in range(400):
        zf.writestr(f"task{i:03d}.onnx", open(onnx_path, 'rb').read())
print(f"400 models ZIP size (dim {hidden_dim}): {os.path.getsize(zip_path)} bytes")
