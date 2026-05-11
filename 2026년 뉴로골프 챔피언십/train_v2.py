import torch
import torch.nn as nn
import torch.optim as optim
from neurogolf_utils import load_task, prepare_data_for_task, grid_to_tensor
from model_v2 import NeuroGolfNetV2, count_parameters
import os
import sys
import numpy as np

def augment_grid(grid):
    """Returns 8 variants of the grid (4 rotations * 2 flips)."""
    arr = np.array(grid)
    variants = []
    for rot in range(4):
        rotated = np.rot90(arr, k=rot)
        variants.append(rotated.tolist())
        variants.append(np.flipud(rotated).tolist())
    # Return unique ones only to save computation if grid is symmetric
    unique_variants = []
    seen = set()
    for v in variants:
        s = str(v)
        if s not in seen:
            unique_variants.append(v)
            seen.add(s)
    return unique_variants

def train_on_task_v2(task_file, hidden_dim=32, lr=0.005, max_epochs=3000, augment=True, num_blocks=3, kernel_size=3):
    """Improved training with data augmentation and better convergence."""
    # 1. Load Data
    task_data = load_task(task_file)
    
    inputs_list = []
    targets_list = []
    
    # Combine all available exemplars
    all_pairs = task_data['train'] + task_data.get('test', [])
    
    for pair in all_pairs:
        if augment:
            aug_in = augment_grid(pair['input'])
            aug_out = augment_grid(pair['output'])
            # zipped augmentation assumes input/output transformations match (standard in ARC)
            for i_grid, o_grid in zip(aug_in, aug_out):
                inputs_list.append(grid_to_tensor(i_grid))
                targets_list.append(grid_to_tensor(o_grid))
        else:
            inputs_list.append(grid_to_tensor(pair['input']))
            targets_list.append(grid_to_tensor(pair['output']))
            
    inputs = torch.cat(inputs_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    
    task_name = os.path.basename(task_file).replace('.json', '')
    
    # 2. Initialize Model
    model = NeuroGolfNetV2(hidden_dim=hidden_dim, num_blocks=num_blocks, kernel_size=kernel_size)
    params = count_parameters(model)
    log_file = f"log_{task_name}.txt"
    with open(log_file, "w") as f:
        f.write(f"Training Start: {task_file} | dim={hidden_dim}\n")
    
    print(f"--- Training V2 on {task_file} (dim={hidden_dim}, params={params}) ---", flush=True)
    
    # 3. Setup Training with better optimizer config
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 500  # Stop if no improvement for 500 epochs
    
    # 4. Training Loop
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        current_loss = loss.item()
        
        # Track best loss for early stopping
        if current_loss < best_loss - 1e-6:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Check accuracy every 50 epochs
        if epoch % 50 == 0 or epoch == max_epochs - 1:
            with torch.no_grad():
                model.eval()
                final_outputs = model(inputs)
                preds = (torch.sigmoid(final_outputs) > 0.5).float()
                correct = (preds == targets).all().item()
                
                if correct:
                    print(f"Epoch {epoch}: Loss {current_loss:.6f} - SUCCESS!", flush=True)
                    model_path = f"best_model_{task_name}.pth"
                    torch.save(model.state_dict(), model_path)
                    return {"success": True, "params": params, "loss": current_loss, "epoch": epoch, "hidden_dim": hidden_dim}
        
        # Early stopping
        if patience_counter >= patience_limit and epoch > 500:
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss {current_loss:.6f} (early stop - no improvement)", flush=True)
            break
                
        if epoch % 50 == 0:
            msg = f"Epoch {epoch}: Loss {current_loss:.6f}"
            print(msg, flush=True)
            with open(log_file, "a") as f:
                f.write(msg + "\n")
            
    print(f"Training finished without 100% accuracy. Best loss: {best_loss:.6f}", flush=True)
    return {"success": False, "params": params, "loss": best_loss, "epoch": epoch, "hidden_dim": hidden_dim}

if __name__ == "__main__":
    task_file = "task001.json"
    if len(sys.argv) > 1:
        task_file = sys.argv[1]
    result = train_on_task_v2(task_file)
    print(result)
