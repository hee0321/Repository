# ============================================================
# MASTERMIND v49.0 - "V3: MAX SPEED & SNOWBALL"
# ============================================================
# "The first 100 turns decide the winner. Speed is the only currency."
#
# NEW IN V3:
#   1. NEUTRAL BLITZ (0-80 turns): 25x weight on neutral planets.
#   2. PRODUCTION SCALING: Production weighted exponentially (prod^1.5).
#   3. KILLER REVENGE: 10x weight on recently lost planets.
#   4. DYNAMIC SPEED INTERCEPT: Precision arrival time calculation.
#   5. TRIPLE DISPATCH: Each planet can launch up to 3 fleets per turn.
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
MIN_FLEET_SIZE = 4
DEFENSE_BUFFER = 6
PROJECTION_HORIZON = 40
REVENGE_WINDOW = 20

GLOBAL_STATE = {
    'planet_history': {}, # {pid: (owner, age)}
    'prev_planets': {}, 
    'planet_velocities': {},
    'step': 0,
    'clusters': []
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
    for _ in range(15):
        fx, fy = predict_pos(tp['id'], tp['x'], tp['y'], tp['radius'], steps, av, tp['is_comet'])
        steps = dist((sx, sy), (fx, fy)) / speed
        if abs(steps - prev_steps) < 0.01: break
        prev_steps = steps
    return math.atan2(fy - sy, fx - sx), steps, fx, fy

def sun_blocked(sx, sy, fx, fy, buffer=2.2):
    return seg_dist((CENTER, CENTER), (sx, sy), (fx, fy)) <= SUN_RADIUS + buffer

def try_waypoint(sx, sy, fx, fy, buffer=2.2):
    mx, my = (sx + fx) / 2, (sy + fy) / 2
    dx, dy = fx - sx, fy - sy
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6: return None
    px, py = -dy / length, dx / length
    offset = SUN_RADIUS + 8.0
    w1 = (mx + px * offset, my + py * offset)
    w2 = (mx - px * offset, my - py * offset)
    wp = w1 if dist(w1, (CENTER, CENTER)) > dist(w2, (CENTER, CENTER)) else w2
    if 2 < wp[0] < 98 and 2 < wp[1] < 98:
        if not sun_blocked(sx, sy, wp[0], wp[1], buffer): return wp
    return None

# --- CORE MODULES ---

def _update_global_state(obs, player):
    GLOBAL_STATE['step'] = obs.get("step", 0)
    comet_ids = set(obs.get("comet_planet_ids", []))
    planets = {p[0]: {
        'id': p[0], 'owner': p[1], 'x': p[2], 'y': p[3], 
        'radius': p[4], 'ships': p[5], 'production': p[6], 
        'is_comet': p[0] in comet_ids
    } for p in obs.get("planets", [])}
    
    # Track History for Revenge
    history = GLOBAL_STATE['planet_history']
    for pid, p in planets.items():
        prev_owner, age = history.get(pid, (-2, 0))
        if prev_owner == player and p['owner'] != player:
            history[pid] = (p['owner'], REVENGE_WINDOW)
        elif age > 0:
            history[pid] = (p['owner'], age - 1)
        else:
            history[pid] = (p['owner'], 0)

    # Velocity tracking for comets
    prev_p = GLOBAL_STATE['prev_planets']
    for pid, p in planets.items():
        if pid in prev_p:
            dx, dy = p['x'] - prev_p[pid][0], p['y'] - prev_p[pid][1]
            GLOBAL_STATE['planet_velocities'][pid] = (dx, dy)
        prev_p[pid] = (p['x'], p['y'])
    
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
    
    scores = {i: p_stats[i]['ships'] + p_stats[i]['prod'] * 20 for i in range(4)}
    leader_id = max(scores, key=scores.get)
    
    my_score = scores[player]
    leader_score = scores[leader_id]
    gap = leader_score - my_score
    
    posture = "NORMAL"
    if leader_id == player:
        posture = "DOMINANT"
    elif gap > 50:
        posture = "ANTI_LEADER"
        
    return posture, leader_id

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
        player = obs.get("player", 0)
        step = obs.get("step", 0)
        av = obs.get("angular_velocity", 0.0)
        fleets = obs.get("fleets", [])
        
        planets = _update_global_state(obs, player)
        my_planets = [p for p in planets.values() if p['owner'] == player]
        if not my_planets: return []
        
        posture, leader_id = _analyze_strategic_posture(planets, fleets, player)
        incoming, committed_friendly = _project_combat(planets, fleets, player, av)
        
        # Target Ranking
        targets = []
        history = GLOBAL_STATE['planet_history']
        
        for tp in planets.values():
            if tp['owner'] == player: continue
            
            # Base value: Production capacity ^ 1.5 (Exponential economy)
            val = (tp['production'] ** 1.5) * 10
            
            # Neutral Blitz (Early game priority)
            if tp['owner'] == -1:
                if step < 80: val *= 25.0
                else: val *= 5.0
            
            # Comet Priority
            if tp['is_comet']: val *= 12.0
            
            # Killer Revenge (10x weight)
            if history.get(tp['id'], (-2, 0))[1] > 0:
                val *= 10.0
                
            # Anti-Leader Bias
            if posture == "ANTI_LEADER" and tp['owner'] == leader_id:
                val *= 2.0
                
            # Orbital / Center Bias
            orb_r = dist((tp['x'], tp['y']), (CENTER, CENTER))
            if orb_r < 30: val *= 1.5
            
            targets.append((tp['id'], val))
            
        targets.sort(key=lambda x: x[1], reverse=True)
        
        # Source Allocation
        moves = []
        source_avail = {}
        for mp in my_planets:
            # Threat-aware Reserve
            threat_window = 35
            p_gar, p_own = _get_projected_state(mp['id'], threat_window, planets, incoming)
            nearby_threat = sum(f[1] for f in incoming[mp['id']] if f[2] < threat_window)
            reserve = max(DEFENSE_BUFFER, nearby_threat - (p_gar if p_own == player else 0))
            source_avail[mp['id']] = max(0, mp['ships'] - reserve)

        committed_this_turn = {pid: 0 for pid in planets}
        dispatches_per_source = {mp['id']: 0 for mp in my_planets}
        
        for tid, t_val in targets:
            tp = planets[tid]
            
            # Projection for "needed" ships
            avg_eta = 25
            p_gar, p_own = _get_projected_state(tid, avg_eta, planets, incoming)
            needed = max(0, p_gar - committed_friendly[tid] - committed_this_turn[tid])
            if tp['owner'] == -1: needed += 2
            else: needed += 5 # Buffer for combat
            
            if needed > sum(source_avail.values()): continue
            
            # Find best sources
            potential_sources = []
            for sid, avail in source_avail.items():
                if avail < MIN_FLEET_SIZE or dispatches_per_source[sid] >= 3: continue
                mp = planets[sid]
                angle, eta, fx, fy = calc_intercept(mp['x'], mp['y'], tp, avail, av)
                
                # Arrival turn check
                if step + eta >= 498: continue
                
                if sun_blocked(mp['x'], mp['y'], fx, fy):
                    wp = try_waypoint(mp['x'], mp['y'], fx, fy)
                    if wp: angle = math.atan2(wp[1]-mp['y'], wp[0]-mp['x'])
                    else: continue
                
                # Dynamic Score: Value / ETA
                score = t_val / (eta + 1)
                potential_sources.append({'id': sid, 'score': score, 'eta': eta, 'angle': angle, 'avail': avail})
            
            if not potential_sources: continue
            
            # Sort by ETA (Max Speed principle)
            potential_sources.sort(key=lambda x: x['eta'])
            
            total_sent_to_target = 0
            for s in potential_sources:
                if total_sent_to_target >= needed: break
                take = min(s['avail'], needed - total_sent_to_target)
                if take >= MIN_FLEET_SIZE:
                    # If this is the main source, and we have plenty, send a bit more to ensure capture
                    if total_sent_to_target == 0 and s['avail'] > take + 10:
                         take += 2
                         
                    moves.append([s['id'], s['angle'], int(take)])
                    source_avail[s['id']] -= take
                    committed_this_turn[tid] += take
                    dispatches_per_source[s['id']] += 1
                    total_sent_to_target += take
                    
        # Defense & Support
        for mp in my_planets:
            if source_avail[mp['id']] < 15 or dispatches_per_source[mp['id']] >= 3: continue
            for target_p in my_planets:
                if target_p['id'] == mp['id']: continue
                if target_p['ships'] < 10 and committed_friendly[target_p['id']] < 5:
                    angle, _, _, _ = calc_intercept(mp['x'], mp['y'], target_p, 10, av)
                    if not sun_blocked(mp['x'], mp['y'], target_p['x'], target_p['y']):
                        moves.append([mp['id'], angle, 10])
                        source_avail[mp['id']] -= 10
                        dispatches_per_source[mp['id']] += 1
                        break
        
        return moves

    except Exception:
        return []

# ============================================================
# END OF MASTERMIND v49.0
# ============================================================
