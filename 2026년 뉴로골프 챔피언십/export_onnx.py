import torch
import torch.nn as nn
from model import NeuroGolfNet
import os

def export_to_onnx(model_path, output_path, hidden_dim=None):
    # 1. Load the PyTorch model
    state_dict = torch.load(model_path, weights_only=True)
    if hidden_dim is None:
        hidden_dim = state_dict['conv1.weight'].shape[0]
        
    model = NeuroGolfNet(hidden_dim=hidden_dim)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 2. Create dummy input
    dummy_input = torch.randn(1, 10, 30, 30)
    
    # 3. Add Sigmoid to the model for the final output (to get probabilities 0-1)
    # Most evaluators will threshold this at 0.5
    class ModelWithSigmoid(nn.Module):
        def __init__(self, base_model):
            super(ModelWithSigmoid, self).__init__()
            self.base_model = base_model
        def forward(self, x):
            return torch.sigmoid(self.base_model(x))
            
    final_model = ModelWithSigmoid(model)
    final_model.eval()
    
    # 4. Export to ONNX
    torch.onnx.export(
        final_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11, # Start with a compatible version
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None # Static shapes required
    )
    
    # 5. Force Opset 10 via metadata override (Desperate fix for Kaggle)
    import onnx
    onnx_model = onnx.load(output_path)
    onnx_model.opset_import[0].version = 10
    onnx.save(onnx_model, output_path)
    
    print(f"Model exported to {output_path} (Forced Opset 10)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pth_file = sys.argv[1]
        onnx_file = pth_file.replace('.pth', '.onnx')
        export_to_onnx(pth_file, onnx_file)
    else:
        # Default test for task001
        if os.path.exists("best_model_task001.pth"):
            export_to_onnx("best_model_task001.pth", "task001.onnx")
        else:
            print("best_model_task001.pth not found. Please run train.py first.")
