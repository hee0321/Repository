import os
import json

def find_kaggle_key():
    for root, dirs, files in os.walk(r"c:\Users\centr\obit game"):
        for file in files:
            if file.endswith((".json", ".txt", ".md")):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if '"username"' in content and '"key"' in content:
                            print(f"FOUND KEY IN: {path}")
                            print(content)
                            return
                except:
                    continue

if __name__ == "__main__":
    find_kaggle_key()
