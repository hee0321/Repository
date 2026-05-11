# ============================================================
# MASTERMIND v47.0 - "THE ARCHITECTURE OF DOMINANCE"
# ============================================================
# "To win is not enough; one must dominate the trajectory of the game."
#
# IMPLEMENTED PRINCIPLES:
#   1. HIERARCHICAL AUTO-REGRESSIVE DECISION (Meta -> Target -> Source)
#   2. RELATIVE POTENTIAL-BASED REWARD (Focus on Leader-Gap)
#   3. CLUSTER-AWARE SPATIAL INTELLIGENCE (Contextual Importance)
#   4. MULTI-SOURCE COORDINATED STRIKES (Team Spirit Logic)
# ============================================================

import math
import random

# --- CONSTANTS ---
BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0

# Strategic Constants
MIN_FLEET_SIZE = 5
DEFENSE_BUFFER = 8
PROJECTION_HORIZON = 40
LEADER_GAP_THRESHOLD = 50 # Relative score difference to trigger Anti-Leader mode

GLOBAL_STATE = {
    'planet_history': {},
    'prev_planets': {}, 
    'planet_velocities': {},
    'step': 0,
    'clusters': [] # Spatial groupings
}

# --- UTILS ---
def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def seg_dist(p, v, w):
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0: return dist(p, v)
    t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
    return dist(p, (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1])))

def fleet_speed(ships):
    ships = max(1, ships)
    s = 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(s, MAX_SPEED)

def predict_pos(p_id, px, py, pr, steps, av, is_comet):
    if is_comet:
        vel = GLOBAL_STATE['planet_velocities'].get(p_id, (0, 0))
        return px + vel[0] * steps, py + vel[1] * steps
    orb_r = dist((px, py), (CENTER, CENTER))
    if orb_r + pr >= ROTATION_RADIUS_LIMIT: return px, py
    angle = math.atan2(py - CENTER, px - CENTER) + av * steps
    return CENTER + orb_r * math.cos(angle), CENTER + orb_r * math.sin(angle)

def calc_intercept(sx, sy, tp, fleet_ships, av):
    speed = fleet_speed(fleet_ships)
    prev_steps = dist((sx, sy), (tp['x'], tp['y'])) / speed
    steps = prev_steps
    fx, fy = tp['x'], tp['y']
    for _ in range(12):
        fx, fy = predict_pos(tp['id'], tp['x'], tp['y'], tp['radius'], steps, av, tp['is_comet'])
        steps = dist((sx, sy), (fx, fy)) / speed
        if abs(steps - prev_steps) < 0.01: break
        prev_steps = steps
    return math.atan2(fy - sy, fx - sx), steps, fx, fy

def sun_blocked(sx, sy, fx, fy, buffer=2.5):
    return seg_dist((CENTER, CENTER), (sx, sy), (fx, fy)) <= SUN_RADIUS + buffer

# --- CORE MODULES ---

def _update_global_state(obs):
    GLOBAL_STATE['step'] = obs.get("step", 0)
    comet_ids = set(obs.get("comet_planet_ids", []))
    planets = {p[0]: {
        'id': p[0], 'owner': p[1], 'x': p[2], 'y': p[3], 
        'radius': p[4], 'ships': p[5], 'production': p[6], 
        'is_comet': p[0] in comet_ids
    } for p in obs.get("planets", [])}
    
    prev_p = GLOBAL_STATE['prev_planets']
    for pid, p in planets.items():
        if pid in prev_p:
            dx, dy = p['x'] - prev_p[pid][0], p['y'] - prev_p[pid][1]
            GLOBAL_STATE['planet_velocities'][pid] = (dx, dy)
        prev_p[pid] = (p['x'], p['y'])
    
    # Contextual Clustering (Every 50 steps)
    if GLOBAL_STATE['step'] % 50 == 0:
        GLOBAL_STATE['clusters'] = _cluster_planets(planets)
        
    return planets

def _cluster_planets(planets):
    clusters = []
    p_list = list(planets.values())
    used = set()
    for p in p_list:
        if p['id'] in used: continue
        cluster = [p['id']]
        used.add(p['id'])
        for other in p_list:
            if other['id'] not in used and dist((p['x'], p['y']), (other['x'], other['y'])) < 25:
                cluster.append(other['id'])
                used.add(other['id'])
        clusters.append(cluster)
    return clusters

def _analyze_strategic_posture(planets, fleets, player):
    p_stats = {i: {'ships': 0, 'prod': 0} for i in range(4)}
    for p in planets.values():
        if p['owner'] >= 0:
            p_stats[p['owner']]['ships'] += p['ships']
            p_stats[p['owner']]['prod'] += p['production']
    for f in fleets:
        if f[1] >= 0: p_stats[f[1]]['ships'] += f[6]
    
    # Dominance Score (Potential Function)
    scores = {i: p_stats[i]['ships'] + p_stats[i]['prod'] * 25 for i in range(4)}
    leader_id = max(scores, key=scores.get)
    
    my_score = scores[player]
    leader_score = scores[leader_id]
    gap = leader_score - my_score
    
    posture = "NORMAL"
    if leader_id == player:
        posture = "STEALTH" if gap > LEADER_GAP_THRESHOLD else "CONSOLIDATE"
    elif gap > LEADER_GAP_THRESHOLD:
        posture = "ANTI_LEADER" # Focus on bringing down the giant
        
    return posture, leader_id, scores

def _project_combat(planets, fleets, player, av):
    incoming = {pid: [] for pid in planets}
    committed_friendly = {pid: 0 for pid in planets}
    for f in fleets:
        f_owner, f_ships, f_angle, f_x, f_y = f[1], f[6], f[4], f[2], f[3]
        f_speed = fleet_speed(f_ships)
        dest_id = None
        for t in range(1, PROJECTION_HORIZON):
            fx, fy = f_x + math.cos(f_angle)*f_speed*t, f_y + math.sin(f_angle)*f_speed*t
            for pid, p in planets.items():
                if abs(fx - p['x']) > 20 or abs(fy - p['y']) > 20: continue 
                ppx, ppy = predict_pos(pid, p['x'], p['y'], p['radius'], t, av, p['is_comet'])
                if (fx-ppx)**2 + (fy-ppy)**2 < (p['radius']+1.5)**2: 
                    dest_id = pid; break
            if dest_id is not None:
                if f_owner == player: committed_friendly[dest_id] += f_ships
                else: incoming[dest_id].append((f_owner, f_ships, t))
                break
    return incoming, committed_friendly

def _get_projected_state(pid, t_arr, planets, incoming):
    p = planets[pid]
    curr, owner = p['ships'], p['owner']
    events = sorted([(f[2], f[0], f[1]) for f in incoming[pid] if f[2] <= t_arr])
    lt = 0
    for t_ev, f_own, f_sh in events:
        if owner != -1: curr += p['production'] * (t_ev - lt)
        if f_own == owner: curr += f_sh
        else:
            curr -= f_sh
            if curr < 0: curr, owner = abs(curr), f_own
        lt = t_ev
    if owner != -1: curr += p['production'] * (t_arr - lt)
    return curr, owner

# --- THE AGENT ---

def agent(obs):
    try:
        # 1. Update State & Environment
        planets = _update_global_state(obs)
        player = obs.get("player", 0)
        av = obs.get("angular_velocity", 0.0)
        fleets = obs.get("fleets", [])
        my_planets = [p for p in planets.values() if p['owner'] == player]
        if not my_planets: return []
        
        # 2. Hierarchical Level 1: Strategic Posture (Meta-Strategy)
        posture, leader_id, player_scores = _analyze_strategic_posture(planets, fleets, player)
        
        # 3. Tactical Projection
        incoming, committed_friendly = _project_combat(planets, fleets, player, av)
        
        # 4. Hierarchical Level 2: Global Target Ranking (PBRS-inspired)
        targets = []
        for tp in planets.values():
            if tp['owner'] == player: continue
            
            # Base value: Production capacity
            val = tp['production'] * 10
            
            # Contextual Bonus: Clustering
            for cluster in GLOBAL_STATE['clusters']:
                if tp['id'] in cluster:
                    # High value if it helps dominate a cluster
                    owned_in_cluster = sum(1 for cid in cluster if planets[cid]['owner'] == player)
                    val += (len(cluster) - owned_in_cluster) * 5
                    break
            
            # Relative Reward: Anti-Leader
            if posture == "ANTI_LEADER" and tp['owner'] == leader_id:
                val *= 2.0
            
            # Neutral & Comet Bonuses
            if tp['owner'] == -1: val *= 4.0
            if tp['is_comet']: val *= 10.0
            
            targets.append((tp['id'], val))
            
        targets.sort(key=lambda x: x[1], reverse=True)
        
        # 5. Hierarchical Level 3: Multi-Source Allocation
        moves = []
        source_avail = {}
        for mp in my_planets:
            # Reserve logic
            p_gar, p_own = _get_projected_state(mp['id'], 35, planets, incoming)
            threat = sum(f[1] for f in incoming[mp['id']] if f[2] < 35)
            reserve = max(DEFENSE_BUFFER, threat - (p_gar if p_own == player else 0))
            if posture == "STEALTH": reserve += 25
            source_avail[mp['id']] = max(0, mp['ships'] - reserve)

        committed_this_turn = {pid: 0 for pid in planets}
        
        # Dispatch Loop
        for tid, t_val in targets:
            tp = planets[tid]
            
            # Estimate needed ships (average ETA ~25)
            p_gar, p_own = _get_projected_state(tid, 25, planets, incoming)
            needed = max(0, p_gar - committed_friendly[tid] - committed_this_turn[tid]) + 5
            
            if needed > sum(source_avail.values()): continue
            
            # Multi-source coordination
            potential_sources = []
            for sid, avail in source_avail.items():
                if avail < 2: continue
                mp = planets[sid]
                angle, eta, _, _ = calc_intercept(mp['x'], mp['y'], tp, avail, av)
                if sun_blocked(mp['x'], mp['y'], tp['x'], tp['y']): continue
                potential_sources.append({'id': sid, 'avail': avail, 'eta': eta, 'angle': angle})
            
            if not potential_sources: continue
            
            # Sort sources by proximity to target
            potential_sources.sort(key=lambda x: x['eta'])
            
            dispatched = 0
            for s in potential_sources:
                if dispatched >= needed: break
                take = min(s['avail'], needed - dispatched)
                if take >= MIN_FLEET_SIZE:
                    moves.append([s['id'], s['angle'], take])
                    source_avail[s['id']] -= take
                    committed_this_turn[tid] += take
                    dispatched += take
                    
        # 6. Consolidation & Defense
        for mp in my_planets:
            if source_avail[mp['id']] < 10: continue
            # Check for weak friendly planets
            for target_p in my_planets:
                if target_p['id'] == mp['id']: continue
                if target_p['ships'] < 15 and committed_friendly[target_p['id']] < 5:
                    angle, _, _, _ = calc_intercept(mp['x'], mp['y'], target_p, 10, av)
                    if not sun_blocked(mp['x'], mp['y'], target_p['x'], target_p['y']):
                        moves.append([mp['id'], angle, 10])
                        source_avail[mp['id']] -= 10
                        break
        
        return moves

    except Exception:
        return []

# ============================================================
# END OF MASTERMIND v47.0
# ============================================================
