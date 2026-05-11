import json

def analyze_task(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    for i, example in enumerate(data['train']):
        print(f"--- Example {i+1} ---")
        input_grid = example['input']
        output_grid = example['output']
        
        rows = len(input_grid)
        cols = len(input_grid[0])
        
        # Find objects in input
        objects = []
        visited = set()
        for r in range(rows):
            for c in range(cols):
                if input_grid[r][c] != 0 and (r, c) not in visited:
                    # Find extent of this object
                    obj_cells = []
                    stack = [(r, c)]
                    visited.add((r, c))
                    while stack:
                        curr_r, curr_c = stack.pop()
                        obj_cells.append((curr_r, curr_c, input_grid[curr_r][curr_c]))
                        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and input_grid[nr][nc] != 0 and (nr, nc) not in visited:
                                visited.add((nr, nc))
                                stack.append((nr, nc))
                    objects.append(obj_cells)
        
        for j, obj in enumerate(objects):
            min_r = min(cell[0] for cell in obj)
            max_r = max(cell[0] for cell in obj)
            min_c = min(cell[1] for cell in obj)
            max_c = max(cell[1] for cell in obj)
            
            # Check shadow length in output
            # Shadow starts at max_r + 1
            shadow_len = 0
            for r in range(max_r + 1, rows):
                all_shadow = True
                for c in range(min_c, max_c + 1):
                    if output_grid[r][c] != 3:
                        all_shadow = False
                        break
                if all_shadow:
                    shadow_len += 1
                else:
                    break
            
            print(f"Object {j+1}: Rows {min_r}-{max_r}, Cols {min_c}-{max_c}, Colors {[c[2] for c in obj]}, Shadow Len: {shadow_len}")

analyze_task('c:/Users/centr/2026년 뉴로골프 챔피언십/task397.json')
