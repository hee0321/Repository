import json
import os

for f in ["task005.json", "task009.json", "task069.json"]:
    if os.path.exists(f):
        data = json.load(open(f))
        print(f"{f} - Train: {len(data.get('train', []))}, Test: {len(data.get('test', []))}, Gen: {len(data.get('arc-gen', []))}")
