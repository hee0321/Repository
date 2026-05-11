from collections import deque

WALL_N, WALL_E, WALL_S, WALL_W = 1, 2, 4, 8
DIRS = {"NORTH":(0,1,WALL_N),"EAST":(1,0,WALL_E),"SOUTH":(0,-1,WALL_S),"WEST":(-1,0,WALL_W)}
T_FAC, T_SCT, T_WRK, T_MNR = 0, 1, 2, 3

class Agent:
    def __init__(self, cfg):
        self.w = cfg.width
        self.cfg = cfg
        self.pid = -1
        self.turn = 0
        self.prev_sb = 0
        self.taken = {}
        self.ghost = {}       # (col,row) -> {val, turn, alive}
        self.targets = {}     # uid -> (col,row) waypoint

    def wl(self, c, r, obs):
        i = (r - obs.southBound) * self.w + c
        if 0 <= i < len(obs.walls) and obs.walls[i] != -1: return obs.walls[i]
        return 0

    def sync(self, obs):
        self.pid = obs.player
        self.turn += 1
        sb = obs.southBound
        # Purge below south bound
        if sb > self.prev_sb:
            self.ghost = {p: v for p, v in self.ghost.items() if p[1] >= sb}
        # Track visible crystals
        vis = set()
        for k, v in obs.crystals.items():
            p = tuple(map(int, k.split(',')))
            vis.add(p)
            self.ghost[p] = {"val": v, "t": self.turn, "alive": True}
        # Fog check: mark unseen as dead
        my = {u: d for u, d in obs.robots.items() if d[4] == self.pid}
        for d in my.values():
            vr = 7 if d[0] == T_SCT else 5
            for gp in list(self.ghost):
                if abs(gp[0]-d[1]) <= vr and abs(gp[1]-d[2]) <= vr and gp not in vis:
                    self.ghost[gp]["alive"] = False
        # Expire old ghosts
        dead = [p for p, v in self.ghost.items() if not v["alive"] or self.turn - v["t"] > 25]
        for p in dead: del self.ghost[p]
        self.prev_sb = sb

    def bfs(self, sc, sr, uid, obs, depth=8):
        """BFS to find best reachable cell; returns first-step direction"""
        sb = obs.southBound
        q = deque([(sc, sr, None, 0)])
        seen = {(sc, sr)}
        best_s = (sr - sb) * 150.0
        best_d = None
        wp = self.targets.get(uid)

        while q:
            c, r, fd, dep = q.popleft()
            # Score
            ds = r - sb
            s = ds * 200.0
            if ds <= 2: s -= 2000000.0 / (ds + 0.1)
            # Crystal/ghost attraction
            for p, v in self.ghost.items():
                if not v["alive"]: continue
                d = abs(p[0]-c) + abs(p[1]-r)
                if d == 0: s += v["val"] * 600.0
                elif d < 8:
                    mult = 5.0 if p == wp else 1.0  # Waypoint bonus
                    s += v["val"] * 40.0 * mult / (d + 1)
            # Enemy penalty
            for ed in obs.robots.values():
                if ed[4] != self.pid:
                    d = abs(c-ed[1]) + abs(r-ed[2])
                    if d <= 3: s -= 50000.0 / (d*d + 1)
            # Ally spread
            for ad in obs.robots.values():
                if ad[4] == self.pid:
                    d = abs(c-ad[1]) + abs(r-ad[2])
                    if 0 < d <= 2: s -= 300.0 / d
            # Dead-end trap detection: count exits
            w = self.wl(c, r, obs)
            exits = sum(1 for _, (_, _, bit) in DIRS.items() if not (w & bit))
            if exits <= 1: s -= 50000.0   # Dead-end trap! Avoid!
            elif exits == 2: s -= 5000.0  # Narrow corridor, risky
            s -= dep * 8.0  # Prefer shorter paths
            if fd and s > best_s:
                best_s, best_d = s, fd
            if dep >= depth: continue
            for dn, (dc, dr, bit) in DIRS.items():
                if w & bit: continue
                nx, ny = c+dc, r+dr
                if (nx, ny) in seen or nx < 0 or nx >= self.w or ny <= sb: continue
                if dep == 0 and (nx, ny) in self.taken: continue
                seen.add((nx, ny))
                q.append((nx, ny, fd or dn, dep+1))
        return best_d

    def get_action(self, obs):
        self.sync(obs)
        acts = {}
        bots = {u: d for u, d in obs.robots.items() if d[4] == self.pid}
        self.taken = {(d[1], d[2]): u for u, d in bots.items()}
        fac = next((d for d in bots.values() if d[0] == T_FAC), None)

        # --- Waypoint cleanup & auction ---
        for u in list(self.targets):
            if u not in bots: del self.targets[u]; continue
            wp = self.targets[u]
            rd = bots[u]
            if (rd[1], rd[2]) == wp or wp not in self.ghost or not self.ghost[wp]["alive"]:
                del self.targets[u]
        # Assign new waypoints
        assigned = set(self.targets.values())
        avail = [(p, v["val"]) for p, v in self.ghost.items() if v["alive"] and p not in assigned]
        for u, d in sorted(bots.items(), key=lambda x: (0 if x[1][0]==T_SCT else 1)):
            if d[0] == T_FAC or u in self.targets or not avail: continue
            best_i, best_sc = -1, -1
            for i, (tp, tv) in enumerate(avail):
                sc = tv / (abs(tp[0]-d[1]) + abs(tp[1]-d[2]) + 1)
                if sc > best_sc: best_sc, best_i = sc, i
            if best_i >= 0:
                self.targets[u] = avail[best_i][0]
                avail.pop(best_i)

        # --- Scout energy transfer to factory ---
        if fac:
            for u, d in bots.items():
                if d[0] == T_SCT and d[3] > 0:
                    dx, dy = d[1]-fac[1], d[2]-fac[2]
                    if abs(dx)+abs(dy) == 1:
                        if dx == 1: acts[u] = "TRANSFER_WEST"
                        elif dx == -1: acts[u] = "TRANSFER_EAST"
                        elif dy == 1: acts[u] = "TRANSFER_SOUTH"
                        elif dy == -1: acts[u] = "TRANSFER_NORTH"

        # --- Miner transform ---
        for u, d in bots.items():
            if d[0] == T_MNR and d[5] == 0 and u not in acts:
                pk = str(d[1])+","+str(d[2])
                if hasattr(obs, 'miningNodes') and pk in obs.miningNodes:
                    acts[u] = "TRANSFORM"

        # --- Factory ---
        for u, d in bots.items():
            if d[0] != T_FAC: continue
            _, c, r, nrg, _, mcd, jcd, pcd = d
            dsb = r - obs.southBound
            if mcd > 0:
                if pcd == 0 and dsb > 5:
                    b = self._build(bots, nrg)
                    if b: acts[u] = b
                continue
            w = self.wl(c, r, obs)
            # JUMP: whenever north is blocked and jump ready
            if (w & WALL_N) and jcd == 0 and (c, r+2) not in self.taken:
                acts[u] = "JUMP_NORTH"; self._mv(u, c, r, c, r+2); continue
            # BFS move
            mv = self.bfs(c, r, u, obs, depth=6)
            if mv:
                dc, dr, _ = DIRS[mv]; acts[u] = mv; self._mv(u, c, r, c+dc, r+dr)
            else:
                # STUCK! Try any open direction including south
                for dn in ["NORTH","EAST","WEST","SOUTH"]:
                    dc, dr, bit = DIRS[dn]
                    nx, ny = c+dc, r+dr
                    if not (w & bit) and (nx, ny) not in self.taken and 0 <= nx < self.w and ny > obs.southBound:
                        acts[u] = dn; self._mv(u, c, r, nx, ny); break
                # Still stuck? Build if possible
                if u not in acts and pcd == 0:
                    b = self._build(bots, nrg)
                    if b: acts[u] = b

        # --- Other units ---
        for u, d in bots.items():
            if d[0] == T_FAC or d[5] > 0 or u in acts: continue
            rt, c, r = d[0], d[1], d[2]
            w = self.wl(c, r, obs)
            # Worker: break walls when stuck OR near factory
            if rt == T_WRK:
                near_fac = fac and abs(c-fac[1]) <= 2 and abs(r-fac[2]) <= 3
                if near_fac and (w & WALL_N):
                    acts[u] = "REMOVE_NORTH"; continue
            # BFS move
            dp = 10 if rt == T_SCT else 8
            mv = self.bfs(c, r, u, obs, depth=dp)
            if mv:
                dc, dr, _ = DIRS[mv]; acts[u] = mv; self._mv(u, c, r, c+dc, r+dr)
            else:
                # STUCK! Worker breaks wall, others force-move
                if rt == T_WRK:
                    for dn in ["NORTH","EAST","WEST"]:
                        _, _, bit = DIRS[dn]
                        if w & bit: acts[u] = "REMOVE_" + dn; break
                if u not in acts:
                    for dn in ["NORTH","EAST","WEST","SOUTH"]:
                        dc, dr, bit = DIRS[dn]
                        nx, ny = c+dc, r+dr
                        if not (w & bit) and (nx, ny) not in self.taken and 0 <= nx < self.w and ny > obs.southBound:
                            acts[u] = dn; self._mv(u, c, r, nx, ny); break
        return acts

    def _mv(self, u, oc, or_, nc, nr):
        if (oc, or_) in self.taken and self.taken[(oc, or_)] == u: del self.taken[(oc, or_)]
        self.taken[(nc, nr)] = u

    def _build(self, bots, nrg):
        units = [x[0] for x in bots.values() if x[0] != T_FAC]
        if len(units) >= 5: return None
        if sum(1 for u in units if u==T_SCT) < 2 and nrg >= self.cfg.scoutCost: return "BUILD_SCOUT"
        if sum(1 for u in units if u==T_WRK) < 1 and nrg >= self.cfg.workerCost: return "BUILD_WORKER"
        if sum(1 for u in units if u==T_MNR) < 2 and nrg >= self.cfg.minerCost: return "BUILD_MINER"
        return None

_a = None
def agent(obs, cfg):
    global _a
    try:
        if _a is None: _a = Agent(cfg)
        return _a.get_action(obs)
    except: return {}
