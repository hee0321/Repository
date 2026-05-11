import matplotlib.pyplot as plt
import numpy as np
from neurogolf_utils import load_task

# ARC colors
COLOR_MAP = {
    0: "#000000", # Black
    1: "#0074D9", # Blue
    2: "#FF4136", # Red
    3: "#2ECC40", # Green
    4: "#FFDC00", # Yellow
    5: "#AAAAAA", # Grey
    6: "#F012BE", # Magenta
    7: "#FF851B", # Orange
    8: "#7FDBFF", # Azure
    9: "#870C25"  # Maroon
}

def plot_grid(ax, grid, title):
    grid = np.array(grid)
    rows, cols = grid.shape
    
    # Create an RGB image
    img = np.zeros((rows, cols, 3))
    for r in range(rows):
        for c in range(cols):
            hex_color = COLOR_MAP.get(grid[r, c], "#FFFFFF")
            rgb = [int(hex_color[i:i+2], 16) / 255.0 for i in (1, 3, 5)]
            img[r, c] = rgb
            
    ax.imshow(img, interpolation='nearest')
    ax.set_title(title)
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

def visualize_task(task_path):
    task = load_task(task_path)
    train_pairs = task['train']
    test_pairs = task['test']
    
    num_train = len(train_pairs)
    num_test = len(test_pairs)
    
    fig, axes = plt.subplots(num_train + num_test, 2, figsize=(8, 4 * (num_train + num_test)))
    
    for i, pair in enumerate(train_pairs):
        plot_grid(axes[i, 0], pair['input'], f"Train Input {i+1}")
        plot_grid(axes[i, 1], pair['output'], f"Train Output {i+1}")
        
    for i, pair in enumerate(test_pairs):
        idx = num_train + i
        plot_grid(axes[idx, 0], pair['input'], f"Test Input {i+1}")
        plot_grid(axes[idx, 1], pair['output'], f"Test Output {i+1}")
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    task_file = "task001.json"
    if len(sys.argv) > 1:
        task_file = sys.argv[1]
    visualize_task(task_file)
