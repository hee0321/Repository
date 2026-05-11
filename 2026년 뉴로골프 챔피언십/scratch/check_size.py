import torch
import torch.nn as nn
import onnx
import os
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

class TinyGolfNet(nn.Module):
    def __init__(self, hidden_dim=2):
        super(TinyGolfNet, self).__init__()
        self.conv1 = nn.Conv2d(10, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, 10, kernel_size=3, padding=1)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return torch.sigmoid(self.conv3(x))

model = TinyGolfNet(hidden_dim=2)
dummy = torch.randn(1, 10, 30, 30)
onnx_path = "test_tiny.onnx"
torch.onnx.export(model, dummy, onnx_path, opset_version=10)
print(f"Size: {os.path.getsize(onnx_path)} bytes")
