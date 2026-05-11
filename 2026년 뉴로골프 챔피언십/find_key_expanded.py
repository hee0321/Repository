import os
import json

def find_kaggle_key():
    search_paths = [r"c:\Users\centr", r"C:\Users\centr\.gemini\antigravity\brain"]
    for base_path in search_paths:
        print(f"Searching in: {base_path}")
        for root, dirs, files in os.walk(base_path):
            if ".gemini" in root and "logs" not in root: # Skip non-log gemini stuff to save time
                continue
            for file in files:
                if file.endswith((".json", ".txt", ".md")):
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if '"username"' in content and '"key"' in content and len(content) < 1000:
                                print(f"FOUND POTENTIAL KEY IN: {path}")
                                print(content)
                                # Don't return, keep looking for more
                    except:
                        continue

if __name__ == "__main__":
    find_kaggle_key()
