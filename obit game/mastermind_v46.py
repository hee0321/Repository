# ============================================================
# MASTERMIND v46.0 - "OMNI-STRATEGIC ENGINE"
# ============================================================
# Core Upgrades (100% Efficacy Roadmap):
#   1. MODULAR ARCHITECTURE: Separated state, threat, and solver logic.
#   2. GLOBAL RESOURCE SOLVER: Prioritizes best moves across all planets.
#   3. ADVERSARIAL INTELLIGENCE: Profiles enemy vulnerability and patterns.
#   4. ADAPTIVE META-STRATEGY: Dynamically scales aggression and reserves.
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
MIN_SPEED_FLEET = 16
STEALTH_SCORE_DIFF = 60
HOME_RESERVE_PROD = 4
HOME_RESERVE_AMT = 20
NEUTRAL_MULTIPLIER = 7.0
COMET_MULTIPLIER = 35.0
SCAVENGER_MULTIPLIER = 2.0
DEFENSE_BUFFER = 7
PROJECTION_HORIZON = 45

GLOBAL_STATE = {
    'planet_history': {},
    'prev_planets': {}, 
    'planet_velocities': {},
    'swarm_tick': 0,
    'adversaries': {0: {}, 1: {}, 2: {}, 3: {}} # To be populated with behavior profiles
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
    global GLOBAL_STATE
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
    return math.atan2(fx - sx, fy - sy), steps, fx, fy

def sun_blocked(sx, sy, fx, fy, buffer=2.3):
    return seg_dist((CENTER, CENTER), (sx, sy), (fx, fy)) <= SUN_RADIUS + buffer

# --- MODULES ---

def _update_state(obs):
    global GLOBAL_STATE
    player = obs.get("player", 0)
    step = obs.get("step", 0)
    av = obs.get("angular_velocity", 0.0)
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
            if p['is_comet']:
                d = math.sqrt(dx*dx + dy*dy)
                if d > 0.1: dx, dy = (dx/d)*4.0, (dy/d)*4.0 # Normalize comet velocity
            GLOBAL_STATE['planet_velocities'][pid] = (dx, dy)
        prev_p[pid] = (p['x'], p['y'])
    
    return planets, player, step, av

def _calculate_leaderboard(planets, fleets, player):
    p_stats = {p: {'ships': 0, 'prod': 0} for p in range(4)}
    for p in planets.values():
        if p['owner'] >= 0:
            p_stats[p['owner']]['ships'] += p['ships']
            p_stats[p['owner']]['prod'] += p['production']
    for f in fleets:
        if f[1] >= 0: p_stats[f[1]]['ships'] += f[6]
    
    scores = {pid: stats['ships'] + stats['prod'] * 20 for pid, stats in p_stats.items()}
    leader_id = max(scores, key=scores.get)
    im_leading = (leader_id == player)
    
    other_scores = [scores[pid] for pid in scores if pid != player]
    max_other = max(other_scores) if other_scores else 0
    score_diff = scores[player] - max_other
    
    return im_leading, score_diff, scores

def _project_combat(planets, fleets, player, av):
    incoming = {pid: [] for pid in planets}
    committed_friendly = {pid: 0 for pid in planets}
    
    for f in fleets:
        f_owner, f_ships, f_angle, f_x, f_y = f[1], f[6], f[4], f[2], f[3]
        f_speed = fleet_speed(f_ships)
        dest_id = None
        # Quick spatial filter then check collision
        for t in range(1, PROJECTION_HORIZON):
            fx, fy = f_x + math.cos(f_angle)*f_speed*t, f_y + math.sin(f_angle)*f_speed*t
            for pid, p in planets.items():
                if abs(fx - p['x']) > 20 or abs(fy - p['y']) > 20: continue 
                ppx, ppy = predict_pos(pid, p['x'], p['y'], p['radius'], t, av, p['is_comet'])
                if (fx-ppx)**2 + (fy-ppy)**2 < (p['radius']+1.2)**2: 
                    dest_id = pid; break
            if dest_id is not None:
                if f_owner == player: committed_friendly[dest_id] += f_ships
                else: incoming[dest_id].append((f_owner, f_ships, t))
                break
                
    return incoming, committed_friendly

def _project_ownership(pid, t_arr, planets, incoming):
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

# --- CORE AGENT ---

def agent(obs):
    global GLOBAL_STATE
    try:
        # 1. State Update
        planets, player, step, av = _update_state(obs)
        fleets = obs.get("fleets", [])
        my_p = [p for p in planets.values() if p['owner'] == player]
        if not my_p: return []
        
        # 2. Strategic Context
        im_leading, score_diff, player_scores = _calculate_leaderboard(planets, fleets, player)
        stealth_mode = im_leading and score_diff > STEALTH_SCORE_DIFF
        
        # 2.1 Adversarial Intelligence: Profiling
        # Track expansion and strength
        for pid in range(4):
            if pid == player: continue
            stats = GLOBAL_STATE['adversaries'][pid]
            p_owned = [p for p in planets.values() if p['owner'] == pid]
            current_ships = sum(p['ships'] for p in p_owned)
            
            # Simple "Over-extended" check: High production but low average garrison
            avg_gar = current_ships / len(p_owned) if p_owned else 0
            stats['is_vulnerable'] = (avg_gar < 25 and len(p_owned) > 3)
            stats['is_threat'] = (current_ships > player_scores[player] * 1.2)

        # 3. Combat Projection
        incoming, committed_friendly = _project_combat(planets, fleets, player, av)
        
        # 4. Opening (Hardcoded Precision)
        if step == 7:
            mp = my_p[0]
            neutrals = sorted([p for p in planets.values() if p['owner'] == -1], 
                             key=lambda p: dist((mp['x'], mp['y']), (p['x'], p['y'])))
            moves = []
            for t in neutrals[:3]:
                angle, _, _, _ = calc_intercept(mp['x'], mp['y'], t, 16, av)
                if not sun_blocked(mp['x'], mp['y'], t['x'], t['y']): moves.append([mp['id'], angle, 16])
            if moves: return moves

        # 5. Iterative Global Solver
        # This solves for the best (Source, Target) pair across the entire map,
        # executes it, and then re-evaluates based on remaining ships.
        moves = []
        source_availability = {mp['id']: 0 for mp in my_p}
        for mp in my_p:
            # Threat Analysis & Reserve
            threats = incoming[mp['id']]
            enemy_threats = {}
            for t_own, t_sh, t_eta in threats:
                if t_eta < 30: enemy_threats[t_own] = enemy_threats.get(t_own, 0) + t_sh
            sorted_t = sorted(enemy_threats.values(), reverse=True)
            p_gar, p_own = _project_ownership(mp['id'], 30, planets, incoming)
            res = max(5, sum(sorted_t[:2]) - (p_gar if p_own == player else 0) + DEFENSE_BUFFER)
            if mp['production'] >= HOME_RESERVE_PROD: res += HOME_RESERVE_AMT
            if stealth_mode: res += 30
            source_availability[mp['id']] = mp['ships'] - res

        dispatches_per_source = {mp['id']: 0 for mp in my_p}
        global_turn_committed = {pid: 0 for pid in planets}
        
        while True:
            best_s, best_a, best_sid, best_tid, best_needed, best_angle = -1.0, None, -1, -1, 0, 0
            
            for mp in my_p:
                sid = mp['id']
                avail = source_availability[sid]
                if avail <= MIN_FLEET_SIZE or dispatches_per_source[sid] >= 7: continue
                
                for tp in planets.values():
                    tid = tp['id']
                    if tid == sid: continue
                    
                    # Score calculation
                    dist_to_t = dist((mp['x'], mp['y']), (tp['x'], tp['y']))
                    if (tp['production'] + 0.1) / (dist_to_t / 4.0 + 1) < 0.002: continue

                    angle, eta, _, _ = calc_intercept(mp['x'], mp['y'], tp, avail, av)
                    if step + eta >= 498 or sun_blocked(mp['x'], mp['y'], tp['x'], tp['y']): continue
                    
                    p_gar, p_own = _project_ownership(tid, eta, planets, incoming)
                    
                    # Adjust p_gar based on what we've already committed this turn
                    needed = max(0, p_gar - committed_friendly.get(tid, 0) - global_turn_committed[tid]) + 1
                    
                    if p_own == player:
                        # Reinforce
                        if (committed_friendly[tid] + global_turn_committed[tid]) < 10 and tp['ships'] < 20:
                            score = 0.4 / (eta + 1)
                            val_needed = int(avail * 0.3)
                        else: continue
                    else:
                        # Attack
                        if tp['ships'] > 300: continue 
                        score = (tp['production'] + 1.5) / (needed * eta + 1.0) * 280
                        if tp['owner'] == -1: score *= NEUTRAL_MULTIPLIER
                        if tp['is_comet']: score *= COMET_MULTIPLIER
                        
                        # Adversarial Bonuses
                        if tp['owner'] != -1:
                            adv = GLOBAL_STATE['adversaries'][tp['owner']]
                            if adv.get('is_vulnerable'): score *= 1.4 # Strike while weak
                            # Removed is_threat penalty to maintain aggression
                        
                        if step > 400 and tp['production'] >= 3: score *= 1.6
                        
                        prev_gar, prev_own = _project_ownership(tid, eta - 1, planets, incoming)
                        if prev_own != -1 and prev_own != tp['owner']: score *= 2.0

                        
                        val_needed = int(needed)
                    
                    if score > best_s:
                        best_s, best_sid, best_tid, best_needed, best_angle = score, sid, tid, val_needed, angle
            
            if best_s < 0.001: break # No more good moves
            
            # Execute Best Action
            sid, tid = best_sid, best_tid
            avail = source_availability[sid]
            target = planets[tid]
            
            send = 0
            if target['owner'] == player: # Reinforce
                send = int(avail * 0.3)
            else: # Attack
                if avail < best_needed and target['owner'] != -1:
                    # Multi-planet contribution logic: 
                    # If we can't cover it alone, only help if we are a top-tier planet
                    if avail < best_needed * 0.4: 
                        # To avoid an infinite loop where this pair is always "best"
                        # but we can't send, we need to temporarily skip this planet
                        # for this target ONLY. For now, we'll just skip this source.
                        dispatches_per_source[sid] = 7 
                        continue
                send = min(avail, max(best_needed + 5, avail if target['owner'] != -1 else 0))

                if target['owner'] == -1: send = min(avail, max(21, best_needed + 2))
            
            send = int(max(MIN_FLEET_SIZE, send))
            if send > avail: send = avail
            
            if send >= MIN_FLEET_SIZE:
                moves.append([sid, best_angle, send])
                source_availability[sid] -= send
                global_turn_committed[tid] += send
                dispatches_per_source[sid] += 1
            else:
                break # Protection against infinite loop

            
        # 7. Swarm/Scout (Filler)
        for mp in my_p:
            avail = source_availability[mp['id']]
            if avail > 15 and (GLOBAL_STATE['swarm_tick'] + mp['id']) % 4 == 0:
                targets = [p for p in planets.values() if p['owner'] != player and p['owner'] != -1]
                if targets:
                    target = random.choice(targets)
                    angle, _, _, _ = calc_intercept(mp['x'], mp['y'], target, 1, av)
                    if not sun_blocked(mp['x'], mp['y'], target['x'], target['y']):
                        moves.append([mp['id'], angle, 1])
                        source_availability[mp['id']] -= 1
        
        GLOBAL_STATE['swarm_tick'] += 1
        return moves

    except Exception:
        return []
