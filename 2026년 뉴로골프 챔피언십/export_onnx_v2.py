import torch
import torch.nn as nn
from model_v2 import NeuroGolfNetV2
import onnx
import os

def export_to_onnx_v2(model_path, output_path, hidden_dim=None):
    """Export V2 model to ONNX format with proper encoding handling."""
    state_dict = torch.load(model_path, weights_only=True)
    
    # Auto-detect hidden_dim, num_blocks, and kernel_size from V2 model weights
    if hidden_dim is None:
        if 'conv_in.weight' in state_dict:
            hidden_dim = state_dict['conv_in.weight'].shape[0]
        elif 'conv1.weight' in state_dict:
            hidden_dim = state_dict['conv1.weight'].shape[0]
        else:
            hidden_dim = 32
    
    kernel_size = 3
    if 'conv_in.weight' in state_dict:
        kernel_size = state_dict['conv_in.weight'].shape[2] # weight is [out, in, k, k]
    
    num_blocks = 3
    res_block_keys = [k for k in state_dict.keys() if k.startswith('res_blocks.')]
    if res_block_keys:
        max_idx = -1
        for k in res_block_keys:
            parts = k.split('.')
            if len(parts) > 1 and parts[1].isdigit():
                max_idx = max(max_idx, int(parts[1]))
        num_blocks = max_idx + 1
        
    model = NeuroGolfNetV2(hidden_dim=hidden_dim, num_blocks=num_blocks, kernel_size=kernel_size)
    model.load_state_dict(state_dict)
    model.eval()
    
    dummy_input = torch.randn(1, 10, 30, 30)
    
    class ModelWithSigmoid(nn.Module):
        def __init__(self, base_model):
            super(ModelWithSigmoid, self).__init__()
            self.base_model = base_model
        def forward(self, x):
            return torch.sigmoid(self.base_model(x))
            
    final_model = ModelWithSigmoid(model)
    final_model.eval()
    
    # Export with opset 18 (native for this PyTorch version)
    torch.onnx.export(
        final_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None
    )
    
    # Force Opset 10 for Kaggle compatibility and ensure weights are embedded
    onnx_model = onnx.load(output_path)
    onnx_model.opset_import[0].version = 10
    # Save with weights embedded in the main file
    onnx.save(onnx_model, output_path, save_as_external_data=False)
    
    print(f"V2 Model exported to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pth_file = sys.argv[1]
        onnx_file = pth_file.replace('.pth', '.onnx')
        export_to_onnx_v2(pth_file, onnx_file)
