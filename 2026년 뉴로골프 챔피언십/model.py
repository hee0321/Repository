import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuroGolfNet(nn.Module):
    """
    A minimal parameter-efficient network for ARC-AGI.
    Goal: High accuracy with lowest parameter count.
    """
    def __init__(self, hidden_dim=16):
        super(NeuroGolfNet, self).__init__()
        # Input: [1, 10, 30, 30]
        self.conv1 = nn.Conv2d(10, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, 10, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x: [B, 10, 30, 30]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x # Output: [B, 10, 30, 30]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test model size
    model = NeuroGolfNet(hidden_dim=8)
    params = count_parameters(model)
    print(f"Total trainable parameters: {params}")
    
    # Dummy input
    dummy_input = torch.randn(1, 10, 30, 30)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
