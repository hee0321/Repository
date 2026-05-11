import torch
import torch.nn as nn
import torch.optim as optim
from neurogolf_utils import load_task, prepare_data_for_task
from model import NeuroGolfNet, count_parameters

def train_on_task(task_file, hidden_dim=16, lr=0.01, max_epochs=2000):
    # 1. Load Data
    task_data = load_task(task_file)
    inputs, targets = prepare_data_for_task(task_data)
    
    # 2. Initialize Model
    model = NeuroGolfNet(hidden_dim=hidden_dim)
    params = count_parameters(model)
    print(f"--- Training on {task_file} ---")
    print(f"Model Parameters: {params}")
    
    # 3. Setup Training
    criterion = nn.BCEWithLogitsLoss() # Supports zero-hot for padding
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4. Training Loop
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        # Check Accuracy (Exact Match)
        if epoch % 50 == 0 or epoch == max_epochs - 1:
            with torch.no_grad():
                model.eval()
                final_outputs = model(inputs)
                # Threshold at 0.5 for binary classification per channel
                preds = (torch.sigmoid(final_outputs) > 0.5).float()
                
                # Exact match check: all elements in the tensor must match
                correct = (preds == targets).all().item()
                
                if correct:
                    print(f"Epoch {epoch}: Loss {loss.item():.6f} - SUCCESS: 100% Accuracy reached!")
                    import os
                    task_name = os.path.basename(task_file).replace('.json', '')
                    model_path = f"best_model_{task_name}.pth"
                    torch.save(model.state_dict(), model_path)
                    return {"success": True, "params": params, "loss": loss.item(), "epoch": epoch, "hidden_dim": hidden_dim}
                
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.6f}")
            
    print("Training finished without reaching 100% accuracy.")
    return {"success": False, "params": params, "loss": loss.item(), "epoch": max_epochs}

if __name__ == "__main__":
    import sys
    task_file = "task001.json"
    if len(sys.argv) > 1:
        task_file = sys.argv[1]
    
    # Save model state safely
    result = train_on_task(task_file, hidden_dim=8)
    print(result)
