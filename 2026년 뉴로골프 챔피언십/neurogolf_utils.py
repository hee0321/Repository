import json
import torch
import numpy as np

def load_task(file_path):
    """Loads a task JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def grid_to_tensor(grid, target_size=(30, 30)):
    """
    Converts a grid (list of lists) to a [1, 10, 30, 30] tensor.
    - One-hot encoding for colors 0-9.
    - Zero-hot encoding for pixels outside the grid border.
    """
    height = len(grid)
    width = len(grid[0])
    
    # Initialize zero-hot tensor [10, 30, 30]
    tensor = torch.zeros((10, target_size[0], target_size[1]))
    
    for r in range(min(height, target_size[0])):
        for c in range(min(width, target_size[1])):
            color = grid[r][c]
            if 0 <= color <= 9:
                tensor[color, r, c] = 1.0
                
    return tensor.unsqueeze(0)  # Add batch dimension [1, 10, 30, 30]

def prepare_data_for_task(task_data):
    """
    Converts task data (train/test/arc-gen) into tensors.
    Returns (inputs, targets) as [N, 10, 30, 30] tensors.
    """
    inputs = []
    targets = []
    
    # Combine all available exemplars for the specific task
    all_pairs = task_data['train'] + task_data.get('test', []) + task_data.get('arc-gen', [])
    
    for pair in all_pairs:
        inputs.append(grid_to_tensor(pair['input']))
        targets.append(grid_to_tensor(pair['output']))
        
    return torch.cat(inputs, dim=0), torch.cat(targets, dim=0)

def tensor_to_grid(tensor, original_shape):
    """
    Converts a [1, 10, 30, 30] tensor back to a grid of integers.
    Uses argmax to find the color, but only for the original shape.
    """
    height, width = original_shape
    # Remove batch dim and get argmax across channels
    color_indices = torch.argmax(tensor[0], dim=0)
    
    grid = []
    for r in range(height):
        row = []
        for c in range(width):
            # Check if it was zero-hot (all channels zero)
            if torch.sum(tensor[0, :, r, c]) == 0:
                # This shouldn't happen for valid output cells, 
                # but we'll default to 0 or some marker.
                row.append(0)
            else:
                row.append(int(color_indices[r, c]))
        grid.append(row)
    return grid

if __name__ == "__main__":
    # Test with a dummy grid
    sample_grid = [[1, 2], [3, 4]]
    tensor = grid_to_tensor(sample_grid)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Non-zero values in first 2x2: {tensor[0, :, :2, :2].sum()}")
    print(f"Values outside border (at 5,5): {tensor[0, :, 5, 5].sum()}")
