import time
from train_v2 import train_on_task_v2
import torch
import sys

task = "task005.json"
if len(sys.argv) > 1:
    task = sys.argv[1]

# No thread restriction for this test
start = time.time()
print(f"Starting test training for 10 epochs on {task}...")
result = train_on_task_v2(task, max_epochs=10)
end = time.time()
print(f"Time taken for 10 epochs: {end - start:.2f}s")
