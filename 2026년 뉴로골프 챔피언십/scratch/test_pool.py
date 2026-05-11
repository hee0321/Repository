from concurrent.futures import ProcessPoolExecutor
import time
import sys

def worker(i):
    print(f"Worker {i} starting...")
    time.sleep(1)
    return i

if __name__ == "__main__":
    print("Starting pool...")
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(worker, range(10)))
    print(f"Results: {results}")
