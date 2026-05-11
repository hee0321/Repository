import random
import math
from collections import deque

# Constants
WALL_N, WALL_E, WALL_S, WALL_W = 1, 2, 4, 8
DIRS = {"NORTH": (0, 1, WALL_N), "EAST": (1, 0, WALL_E), "SOUTH": (0, -1, WALL_S), "WEST": (-1, 0, WALL_W)}
TYPE_FACTORY, TYPE_SCOUT, TYPE_WORKER, TYPE_MINER = 0, 1, 2, 3

class BestEntryAgent:
    def __init__(self, config):
        self.config = config
        self.width = config.width
        self.memory_map = {} # (col, row) -> wall_bitfield
        self.known_mines = {} # (col, row) -> [energy, maxEnergy, owner]
        self.known_crystals = {} # (col, row) -> energy
        self.mining_nodes = set()
        self.planned_positions = {} # (col, row) -> uid
        self.step = 0
        self.player_id = -1

    def update_memory(self, obs):
        self.step = obs.step
        self.player_id = obs.player
        sb = obs.southBound
        
        # 1. Map Update
        for i, val in enumerate(obs.walls):
            if val != -1:
                row, col = sb + (i // self.width), i % self.width
                self.memory_map[(col, row)] = val
        
        # 2. Resource Update
        self.known_crystals = {(int(k.split(',')[0]), int(k.split(',')[1])): v for k, v in obs.crystals.items()}
        for pos_str, data in obs.mines.items():
            self.known_mines[tuple(map(int, pos_str.split(',')))] = data
        for pos_str in obs.miningNodes:
            self.mining_nodes.add(tuple(map(int, pos_str.split(','))))

    def get_bfs_dist(self, start, target, obs):
        """Standard BFS to find the real distance through walls."""
        if start == target: return 0
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            (curr_col, curr_row), dist = queue.popleft()
            if dist > 15: break # Max search depth for performance
            
            walls = self.memory_map.get((curr_col, curr_row), 0)
            for dc, dr, bit in DIRS.values():
                if not (walls & bit):
                    nxt = (curr_col + dc, curr_row + dr)
                    if nxt == target: return dist + 1
                    if nxt not in visited and 0 <= nxt[0] < self.width and nxt[1] >= obs.southBound:
                        visited.add(nxt)
                        queue.append((nxt, dist + 1))
        return 100 # Not reachable in search depth

    def get_potential(self, col, row, robot_data, obs):
        rtype, r_col, r_row, r_energy = robot_data[0], robot_data[1], robot_data[2], robot_data[3]
        potential = 0
        
        # 1. Global Northward Bias (Essential for survival)
        potential += (row - obs.southBound) * 2.0
        
        # 2. Southern Boundary Repulsion (High priority)
        dist_to_death = row - obs.southBound
        if dist_to_death <= 1: potential -= 5000
        elif dist_to_death <= 3: potential -= 200 / (dist_to_death + 0.1)
        
        # 3. Crystal/Node Attraction (BFS based)
        targets = []
        if rtype == TYPE_MINER: 
            targets = [(t, 100) for t in self.mining_nodes if t not in self.known_mines]
        if not targets: 
            targets = [(t, v) for t, v in self.known_crystals.items()]
            
        for (t_col, t_row), value in targets[:10]: # Check nearest 10 targets
            # Use Manhattan as heuristic, only BFS if close
            h_dist = abs(t_col - col) + abs(t_row - row)
            if h_dist <= 8:
                real_dist = self.get_bfs_dist((col, row), (t_col, t_row), obs)
                potential += (value * 5) / (real_dist + 1)
            else:
                potential += value / (h_dist + 1)
            
        # 4. Enemy Avoidance (Crush Hierarchy)
        for uid, e_data in obs.robots.items():
            if e_data[4] != self.player_id:
                e_type, e_col, e_row = e_data[0], e_data[1], e_data[2]
                dist = abs(e_col - col) + abs(e_row - row)
                if dist <= 2:
                    # Crush Hierarchy: Factory > Miner > Worker > Scout
                    # TYPE_FACTORY=0, TYPE_SCOUT=1, TYPE_WORKER=2, TYPE_MINER=3
                    if e_type == TYPE_FACTORY: # Enemy Factory is dangerous
                        potential -= 500 / (dist + 0.5)
                    elif e_type == TYPE_MINER and rtype != TYPE_FACTORY:
                        potential -= 200 / (dist + 0.5)
                    elif e_type == rtype and rtype != TYPE_FACTORY:
                        # Mutual destruction
                        potential -= 100 / (dist + 0.5)
                        
        # 5. Boundary Check
        if col < 0 or col >= self.width: potential -= 10000
        
        return potential

    def get_action(self, obs):
        self.update_memory(obs)
        actions = {}
        self.planned_positions = {}
        my_robots = {uid: d for uid, d in obs.robots.items() if d[4] == self.player_id}
        
        factory_uid = next((u for u, d in my_robots.items() if d[0] == TYPE_FACTORY), None)
        factory_data = my_robots[factory_uid] if factory_uid else None
        
        # 1. Factory Strategy (Survival & Production)
        if factory_data:
            f_col, f_row = factory_data[1], factory_data[2]
            f_walls = self.memory_map.get((f_col, f_row), 0)
            
            # Survival Move
            f_act = "IDLE"
            if factory_data[5] == 0: # move_cd
                if not (f_walls & WALL_N): f_act = "NORTH"
                elif factory_data[6] == 0: f_act = "JUMP_NORTH"
                elif not (f_walls & WALL_E) and f_col < self.width - 1: f_act = "EAST"
                elif not (f_walls & WALL_W) and f_col > 0: f_act = "WEST"
            
            if f_act != "IDLE":
                actions[factory_uid] = f_act
                dc, dr = (0, 1) if f_act == "NORTH" else (0, 2) if f_act == "JUMP_NORTH" else (1, 0) if f_act == "EAST" else (-1, 0)
                self.planned_positions[(f_col + dc, f_row + dr)] = factory_uid
            else:
                self.planned_positions[(f_col, f_row)] = factory_uid
                
            # Production
            if factory_data[7] == 0 and factory_uid not in actions:
                if (f_col, f_row + 1) not in self.planned_positions:
                    n_miners = sum(1 for d in my_robots.values() if d[0] == TYPE_MINER)
                    n_workers = sum(1 for d in my_robots.values() if d[0] == TYPE_WORKER)
                    n_scouts = sum(1 for d in my_robots.values() if d[0] == TYPE_SCOUT)
                    
                    if factory_data[3] >= self.config.minerCost and n_miners < 5:
                        actions[factory_uid] = "BUILD_MINER"
                    elif factory_data[3] >= self.config.workerCost and n_workers < 2:
                        actions[factory_uid] = "BUILD_WORKER"
                    elif factory_data[3] >= self.config.scoutCost and n_scouts < 2:
                        actions[factory_uid] = "BUILD_SCOUT"
                    elif factory_data[3] >= 1000:
                        actions[factory_uid] = "BUILD_MINER"

        # 2. Unit Strategy
        sorted_uids = sorted([u for u in my_robots if u != factory_uid], key=lambda u: my_robots[u][0], reverse=True)
        
        for uid in sorted_uids:
            data = my_robots[uid]
            rtype, col, row, energy, owner, m_cd = data[:6]
            
            if m_cd > 0:
                self.planned_positions[(col, row)] = uid
                continue
                
            if rtype == TYPE_MINER and (col, row) in self.mining_nodes and (col, row) not in self.known_mines:
                if energy >= 100:
                    actions[uid] = "TRANSFORM"
                    self.planned_positions[(col, row)] = uid
                    continue
            
            if rtype == TYPE_WORKER and factory_data:
                f_c, f_r = factory_data[1], factory_data[2]
                f_w = self.memory_map.get((f_c, f_r), 0)
                if (f_w & WALL_N) and col == f_c and row == f_r + 1:
                    actions[uid] = "REMOVE_SOUTH"
                    self.planned_positions[(col, row)] = uid
                    continue

            best_move = "IDLE"
            best_pot = self.get_potential(col, row, data, obs)
            
            walls = self.memory_map.get((col, row), 0)
            moves = list(DIRS.items())
            random.shuffle(moves)
            
            for d_name, (dc, dr, bit) in moves:
                if not (walls & bit):
                    nxt = (col + dc, row + dr)
                    if nxt not in self.planned_positions and 0 <= nxt[0] < self.width and nxt[1] >= obs.southBound:
                        pot = self.get_potential(nxt[0], nxt[1], data, obs)
                        if pot > best_pot:
                            best_pot = pot
                            best_move = d_name
            
            if best_move != "IDLE":
                actions[uid] = best_move
                dc, dr, _ = DIRS[best_move]
                self.planned_positions[(col + dc, row + dr)] = uid
            else:
                self.planned_positions[(col, row)] = uid
                
        return actions

agent_instance = None
def agent(obs, config):
    global agent_instance
    if agent_instance is None:
        agent_instance = BestEntryAgent(config)
    return agent_instance.get_action(obs)
