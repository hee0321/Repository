# ============================================================
# MASTERMIND v39.0 - "CHAMPION'S GHOST"
# ============================================================
# The Ultimate Reverse-Engineered Shun_PI Strategy:
#   1. THE STEP 7 RULE: Hardcoded multi-split opening at Step 7.
#   2. HYDRA HARASSMENT: High-frequency small fleet saturation (up to 8 slots).
#   3. STRAIGHT-LINE PRECISION: Mathematical sun-grazing paths without detours.
#   4. VELOCITY-BASED INTERCEPT: Absolute precision for comets and rotating planets.
#   5. SIMULATED GARRISON v3: Perfect event-driven prediction.
# ============================================================

import math
import random

BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0

GLOBAL_STATE = {
    'planet_history': {},
    'prev_planets': {}, 
    'planet_velocities': {}
}

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
        if abs(steps - prev_steps) < 0.02: break
        prev_steps = steps
    return math.atan2(fy - sy, fx - sx), steps, fx, fy

def sun_blocked(sx, sy, fx, fy, buffer=2.2): # Shun_PI uses closer sun-grazing
    return seg_dist((CENTER, CENTER), (sx, sy), (fx, fy)) <= SUN_RADIUS + buffer

def agent(obs):
    global GLOBAL_STATE
    try:
        player = obs.get("player", 0)
        step = obs.get("step", 0)
        av = obs.get("angular_velocity", 0.0)
        comet_ids = set(obs.get("comet_planet_ids", []))
        
        # Planet & Velocity tracking
        planets = {p[0]: {'id': p[0], 'owner': p[1], 'x': p[2], 'y': p[3], 'radius': p[4], 'ships': p[5], 'production': p[6], 'is_comet': p[0] in comet_ids} for p in obs.get("planets", [])}
        prev_p = GLOBAL_STATE['prev_planets']
        for pid, p in planets.items():
            if pid in prev_p:
                GLOBAL_STATE['planet_velocities'][pid] = (p['x'] - prev_p[pid][0], p['y'] - prev_p[pid][1])
            prev_p[pid] = (p['x'], p['y'])
            
        all_p = list(planets.values())
        my_p = [p for p in all_p if p['owner'] == player]
        if not my_p: return []
        
        # Leader tracking
        player_strength = {p: {'ships': 0, 'prod': 0} for p in range(4)}
        for p in all_p:
            if p['owner'] >= 0:
                player_strength[p['owner']]['ships'] += p['ships']
                player_strength[p['owner']]['prod'] += p['production']
        for f in obs.get("fleets", []):
            if f[1] >= 0: player_strength[f[1]]['ships'] += f[6]
        
        leader_id = player
        max_p = -1
        for pid, stats in player_strength.items():
            pwr = stats['ships'] + stats['prod'] * 25
            if pwr > max_p: max_p = pwr; leader_id = pid
        am_leading = (leader_id == player)

        # Simulation & Projection
        incoming_fleets = {p['id']: [] for p in all_p}
        committed = {p['id']: 0 for p in all_p}
        for f in obs.get("fleets", []):
            f_owner, f_ships, f_angle, f_x, f_y = f[1], f[6], f[4], f[2], f[3]
            f_speed = fleet_speed(f_ships)
            dest_id = None
            for t in range(1, 45):
                fx, fy = f_x + math.cos(f_angle)*f_speed*t, f_y + math.sin(f_angle)*f_speed*t
                for p in all_p:
                    if abs(fx - p['x']) > 20 or abs(fy - p['y']) > 20: continue 
                    ppx, ppy = predict_pos(p['id'], p['x'], p['y'], p['radius'], t, av, p['is_comet'])
                    if (fx-ppx)**2 + (fy-ppy)**2 < (p['radius']+1.2)**2: dest_id = p['id']; break
                if dest_id is not None:
                    if f_owner == player: committed[dest_id] += f_ships
                    else: incoming_fleets[dest_id].append((f_owner, f_ships, t))
                    break

        def predict_garrison(p_id, t_arr):
            p = planets[p_id]
            curr, owner = p['ships'], p['owner']
            events = sorted([(f[2], f[0], f[1]) for f in incoming_fleets[p_id] if f[2] <= t_arr])
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

        # Defense Prep
        for p in all_p:
            threats = [f for f in incoming_fleets[p['id']] if f[1] >= 3]
            p['min_threat_eta'] = min(f[2] for f in threats) if threats else 999
            p['total_threat'] = sum(f[1] for f in threats)
            p_gar, p_own = predict_garrison(p['id'], min(p['min_threat_eta'], 30))
            p['defense_needed'] = max(0, p['total_threat'] - p_gar + 6) if (p_own != player and p['owner'] == player) else 0

        # === THE STEP 7 OPENING ===
        if step == 7:
            moves = []
            mp = my_p[0]
            # Find 3 closest neutrals
            neutrals = [p for p in all_p if p['owner'] == -1]
            neutrals.sort(key=lambda p: dist((mp['x'], mp['y']), (p['x'], p['y'])))
            targets = neutrals[:3]
            for t in targets:
                angle, _, _, _ = calc_intercept(mp['x'], mp['y'], t, 15, av)
                if not sun_blocked(mp['x'], mp['y'], t['x'], t['y']):
                    moves.append([mp['id'], angle, 15])
            if moves: return moves

        # Regular Logic
        my_p.sort(key=lambda p: p['ships'], reverse=True)
        moves = []
        min_threshold = 2 if step < 80 else 5
        
        for mp in my_p:
            res = max(2, mp['total_threat'] + 2)
            avail = mp['ships'] - res
            if avail <= min_threshold: continue
            
            dispatches = 0
            # HYDRA: Increased dispatch limit for saturation
            max_dispatches = 8 if avail > 100 else 4 
            
            while avail > min_threshold and dispatches < max_dispatches:
                best_t, best_s, best_a, best_needed = None, -1.0, 0, 0
                for tp in all_p:
                    if tp['id'] == mp['id']: continue
                    
                    dist_to_t = dist((mp['x'], mp['y']), (tp['x'], tp['y']))
                    if (tp['production'] + 0.1) / (dist_to_t / 4.0 + 1) < 0.003: continue

                    curr_send = avail if tp['owner'] != -1 else tp['ships'] + 4
                    angle, eta, _, _ = calc_intercept(mp['x'], mp['y'], tp, curr_send, av)
                    if step + eta >= 497: continue
                    if sun_blocked(mp['x'], mp['y'], tp['x'], tp['y']): continue # No detours, just straight lines
                    
                    p_gar, p_own = predict_garrison(tp['id'], eta)
                    if p_own == player:
                        if tp['defense_needed'] > 0 and eta < tp['min_threat_eta'] + 3:
                            needed = tp['defense_needed'] - committed[tp['id']]
                            if needed <= 0: continue
                            score = (tp['production'] * 20) / (eta + 1)
                            val_needed = int(needed)
                        else: continue
                    else:
                        needed = max(0, p_gar - committed.get(tp['id'], 0)) + 1
                        if avail < needed: continue
                        if committed.get(tp['id'], 0) > p_gar + 20: continue
                        
                        # SHUN_PI PRIORITY: Production / (Needed * Time)
                        score = (tp['production'] + 0.5) / (needed * eta + 0.1) * 150
                        if tp['owner'] == -1: score *= 7.5
                        if tp['is_comet']: score *= 20.0
                        
                        orb_r = dist((tp['x'], tp['y']), (CENTER, CENTER))
                        if orb_r < 35: score *= 2.0
                        if am_leading and step > 400: score *= 0.5 # Play safe late game
                        
                        val_needed = int(needed)
                    
                    if score > best_s: best_s, best_t, best_a, best_needed = score, tp, angle, val_needed
                
                if best_t:
                    if best_t['owner'] == -1: send = min(avail, max(18, best_needed + 4))
                    else:
                        send = min(avail, max(best_needed + 4, avail))
                        if dispatches == 0 and avail > best_needed * 3 + 20: send = best_needed + 5
                    
                    if dist((mp['x'], mp['y']), (best_t['x'], best_t['y'])) > 35: send = max(send, 22)
                    
                    send = max(min_threshold, int(send))
                    if send > avail: send = int(avail)
                    moves.append([mp['id'], best_a, send])
                    avail -= send
                    committed[best_t['id']] += send
                    dispatches += 1
                else:
                    # HYDRA HARASSMENT
                    if avail > 10:
                        enemies = [p for p in all_p if p['owner'] != player and p['owner'] != -1]
                        if enemies:
                            target = random.choice(enemies)
                            if dist((mp['x'], mp['y']), (target['x'], target['y'])) < 75:
                                angle, _, _, _ = calc_intercept(mp['x'], mp['y'], target, 1, av)
                                if not sun_blocked(mp['x'], mp['y'], target['x'], target['y']):
                                    moves.append([mp['id'], angle, 1])
                                    avail -= 1
                    break
        return moves
    except Exception: return []
