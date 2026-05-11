import torch
import torch.nn as nn
from model_v2 import NeuroGolfNetV2

m = NeuroGolfNetV2(32)
m.eval()
x = torch.randn(1, 10, 30, 30)

class M(nn.Module):
    def __init__(self, b):
        super().__init__()
        self.b = b
    def forward(self, x):
        return torch.sigmoid(self.b(x))

fm = M(m)
fm.eval()

# Test opset 18
torch.onnx.export(fm, x, 'test_v2_op18.onnx', opset_version=18,
                  input_names=['input'], output_names=['output'], dynamic_axes=None)
import os
print(f"opset18: {os.path.exists('test_v2_op18.onnx')} size={os.path.getsize('test_v2_op18.onnx') if os.path.exists('test_v2_op18.onnx') else 0}")

# Test opset 11 
try:
    torch.onnx.export(fm, x, 'test_v2_op11.onnx', opset_version=11,
                      input_names=['input'], output_names=['output'], dynamic_axes=None)
    print(f"opset11: {os.path.exists('test_v2_op11.onnx')} size={os.path.getsize('test_v2_op11.onnx') if os.path.exists('test_v2_op11.onnx') else 0}")
except Exception as e:
    print(f"opset11 failed: {e}")

# Force opset 10 on opset 18 file
import onnx
model = onnx.load('test_v2_op18.onnx')
model.opset_import[0].version = 10
onnx.save(model, 'test_v2_forced10.onnx')
print(f"forced10: {os.path.exists('test_v2_forced10.onnx')} size={os.path.getsize('test_v2_forced10.onnx')}")
