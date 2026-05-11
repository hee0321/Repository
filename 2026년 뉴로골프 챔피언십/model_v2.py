import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuroGolfNetV2(nn.Module):
    """
    Improved ARC-AGI network with residual connections and deeper layers.
    Much more capable than V1 while still keeping parameters reasonable.
    """
    def __init__(self, hidden_dim=32, num_blocks=3, kernel_size=3):
        super(NeuroGolfNetV2, self).__init__()
        padding = kernel_size // 2
        # Input: [B, 10, 30, 30]
        self.conv_in = nn.Conv2d(10, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(hidden_dim)
            )
            self.res_blocks.append(block)
        
        # Output projection
        self.conv_out = nn.Conv2d(hidden_dim, 10, kernel_size=1)
        
    def forward(self, x):
        # x: [B, 10, 30, 30]
        x = F.relu(self.bn1(self.conv_in(x)))
        
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = F.relu(x + residual)
        
        x = self.conv_out(x)
        return x  # Output: [B, 10, 30, 30]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    for dim in [16, 32, 48, 64]:
        model = NeuroGolfNetV2(hidden_dim=dim)
        params = count_parameters(model)
        print(f"Hidden {dim}: {params} parameters")
    
    dummy_input = torch.randn(1, 10, 30, 30)
    model = NeuroGolfNetV2(hidden_dim=32)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
