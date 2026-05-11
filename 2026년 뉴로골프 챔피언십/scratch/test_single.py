import os
import sys
import io

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import onnx

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from neurogolf_utils import load_task, grid_to_tensor

class NeuroGolfNetV1(nn.Module):
    def __init__(self, hidden_dim=12):
        super(NeuroGolfNetV1, self).__init__()
        self.conv1 = nn.Conv2d(10, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, 10, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return torch.sigmoid(self.conv3(x))

tid = 1
# Correct path to task file (it's in root)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
task_file = os.path.join(root_dir, f"task{tid:03d}.json")
onnx_path = os.path.join(root_dir, f"task{tid:03d}.onnx")

print(f"Loading {task_file}...")
task_data = load_task(task_file)
inputs = torch.cat([grid_to_tensor(p['input']) for p in task_data['train']], dim=0)
targets = torch.cat([grid_to_tensor(p['output']) for p in task_data['train']], dim=0)

model = NeuroGolfNetV1(hidden_dim=12)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Training...")
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(inputs)
    loss = criterion(out, targets)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

print("Exporting...")
model.eval()
dummy = torch.randn(1, 10, 30, 30)
torch.onnx.export(model, dummy, onnx_path, opset_version=10)
print(f"Success! {onnx_path} created.")
