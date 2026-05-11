import torch
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def test_import(x):
    try:
        import torch
        import pandas
        from train_v2 import train_on_task_v2
        return f"Process {x}: Success"
    except Exception as e:
        return f"Process {x}: Error {str(e)}"

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(test_import, [1, 2]))
    print(results)
