# ============================================================
# MASTERMIND v38.0 - "SINGULARITY"
# ============================================================
# The Ultimate Evolution for 100% Dominance:
#   1. VELOCITY TRACKING: Predicts non-circular movement (Comets) via position delta.
#   2. SMART GARRISON v2: Full event-based simulation of arrival state.
#   3. FFA STEALTH: If leading, play defensively until the "Wipeout" window.
#   4. SYMMETRY EXPLOIT: Increased pressure on mirror quadrants.
#   5. MICRO SNIPE: Targets low-health planets with high-speed precision fleets.
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
    'prev_planets': {}, # {id: (x, y)}
    'planet_velocities': {} # {id: (dx, dy)}
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
        # Linear extrapolation for comets (based on observed velocity)
        vel = GLOBAL_STATE['planet_velocities'].get(p_id, (0, 0))
        return px + vel[0] * steps, py + vel[1] * steps
    
    # Orbital rotation for normal planets
    orb_r = dist((px, py), (CENTER, CENTER))
    if orb_r + pr >= ROTATION_RADIUS_LIMIT: return px, py
    angle = math.atan2(py - CENTER, px - CENTER) + av * steps
    return CENTER + orb_r * math.cos(angle), CENTER + orb_r * math.sin(angle)

def calc_intercept(mp, tp, fleet_ships, av):
    speed = fleet_speed(fleet_ships)
    prev_steps = dist((mp['x'], mp['y']), (tp['x'], tp['y'])) / speed
    steps = prev_steps
    fx, fy = tp['x'], tp['y']
    for _ in range(12): # More iterations for precision
        fx, fy = predict_pos(tp['id'], tp['x'], tp['y'], tp['radius'], steps, av, tp['is_comet'])
        steps = dist((mp['x'], mp['y']), (fx, fy)) / speed
        if abs(steps - prev_steps) < 0.05: break
        prev_steps = steps
    return math.atan2(fy - mp['y'], fx - mp['x']), steps, fx, fy

def sun_blocked(sx, sy, fx, fy, buffer=2.4):
    return seg_dist((CENTER, CENTER), (sx, sy), (fx, fy)) <= SUN_RADIUS + buffer

def try_waypoint(sx, sy, fx, fy, buffer=2.4):
    mx, my = (sx + fx) / 2, (sy + fy) / 2
    dx, dy = fx - sx, fy - sy
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6: return None
    px, py = -dy / length, dx / length
    offset = SUN_RADIUS + 11.0
    w1 = (mx + px * offset, my + py * offset)
    w2 = (mx - px * offset, my - py * offset)
    wp = w1 if dist(w1, (CENTER, CENTER)) > dist(w2, (CENTER, CENTER)) else w2
    if 1 < wp[0] < 99 and 1 < wp[1] < 99:
        if not sun_blocked(sx, sy, wp[0], wp[1], buffer): return wp
    return None

def agent(obs):
    global GLOBAL_STATE
    try:
        player = obs.get("player", 0)
        step = obs.get("step", 0)
        av = obs.get("angular_velocity", 0.0)
        comet_ids = set(obs.get("comet_planet_ids", []))
        
        planets = {p[0]: {'id': p[0], 'owner': p[1], 'x': p[2], 'y': p[3], 'radius': p[4], 'ships': p[5], 'production': p[6], 'is_comet': p[0] in comet_ids} for p in obs.get("planets", [])}
        
        # Velocity Tracking Logic
        prev_p = GLOBAL_STATE['prev_planets']
        for pid, p in planets.items():
            if pid in prev_p:
                GLOBAL_STATE['planet_velocities'][pid] = (p['x'] - prev_p[pid][0], p['y'] - prev_p[pid][1])
            prev_p[pid] = (p['x'], p['y'])
            
        all_p = list(planets.values())
        my_p = [p for p in all_p if p['owner'] == player]
        if not my_p: return []
        
        # Strength Analysis
        player_strength = {p: {'ships': 0, 'prod': 0} for p in range(4)}
        for p in all_p:
            if p['owner'] >= 0:
                player_strength[p['owner']]['ships'] += p['ships']
                player_strength[p['owner']]['prod'] += p['production']
        for f in obs.get("fleets", []):
            if f[1] >= 0: player_strength[f[1]]['ships'] += f[6]
        
        is_ffa = len([pid for pid, s in player_strength.items() if s['ships'] > 0 or s['prod'] > 0]) >= 3
        leader_id = player
        max_p = -1
        for pid, stats in player_strength.items():
            pwr = stats['ships'] + stats['prod'] * 25
            if pwr > max_p: max_p = pwr; leader_id = pid
        am_leading = (leader_id == player)

        # History Tracking
        history = GLOBAL_STATE['planet_history']
        for pid, p in planets.items():
            prev = history.get(pid, (p['owner'], 0))
            if prev[0] == player and p['owner'] != player: history[pid] = (p['owner'], 35)
            elif prev[1] > 0: history[pid] = (p['owner'], prev[1] - 1)
            else: history[pid] = (p['owner'], 0)

        # Advanced Fleet Simulation
        committed = {p['id']: 0 for p in all_p}
        incoming_fleets = {p['id']: [] for p in all_p}
        for f in obs.get("fleets", []):
            f_owner, f_ships, f_angle, f_x, f_y = f[1], f[6], f[4], f[2], f[3]
            f_speed = fleet_speed(f_ships)
            dest_id = None
            for t in range(1, 45):
                fx, fy = f_x + math.cos(f_angle)*f_speed*t, f_y + math.sin(f_angle)*f_speed*t
                for p in all_p:
                    if abs(fx - p['x']) > 20 or abs(fy - p['y']) > 20: continue 
                    ppx, ppy = predict_pos(p['id'], p['x'], p['y'], p['radius'], t, av, p['is_comet'])
                    if (fx-ppx)**2 + (fy-ppy)**2 < (p['radius']+1.3)**2: dest_id = p['id']; break
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

        # Defense Logic
        for p in all_p:
            threats = [f for f in incoming_fleets[p['id']] if f[1] >= 3]
            p['min_threat_eta'] = min(f[2] for f in threats) if threats else 999
            p['total_threat'] = sum(f[1] for f in threats)
            p_gar, p_own = predict_garrison(p['id'], min(p['min_threat_eta'], 35))
            p['is_doomed'] = (p_own != player and p['owner'] == player)
            p['defense_needed'] = max(0, p['total_threat'] - p_gar + 6) if p['is_doomed'] else 0

        my_p.sort(key=lambda p: p['ships'], reverse=True)
        moves = []
        min_threshold = 2 if step < 80 else 5
        neutral_buffer = 4

        for mp in my_p:
            res = max(2, mp['total_threat'] + 2)
            if am_leading and step > 400: res += 15 # Play safe in late game lead
            avail = mp['ships'] - res
            if avail <= min_threshold: continue
            
            dispatches = 0
            while avail > min_threshold and dispatches < 4:
                best_t, best_s, best_a, best_needed = None, -1.0, 0, 0
                for tp in all_p:
                    if tp['id'] == mp['id']: continue
                    
                    dist_to_t = dist((mp['x'], mp['y']), (tp['x'], tp['y']))
                    if (tp['production'] + 0.1) / (dist_to_t / 4.0 + 1) < 0.004: continue

                    curr_send = avail if tp['owner'] != -1 else tp['ships'] + neutral_buffer
                    angle, eta, _, _ = calc_intercept(mp, tp, curr_send, av)
                    if step + eta >= 496: continue
                    
                    fx, fy = predict_pos(tp['id'], tp['x'], tp['y'], tp['radius'], eta, av, tp['is_comet'])
                    if sun_blocked(mp['x'], mp['y'], fx, fy):
                        wp = try_waypoint(mp['x'], mp['y'], fx, fy)
                        if wp: angle = math.atan2(wp[1]-mp['y'], wp[0]-mp['x'])
                        else: continue
                    
                    p_gar, p_own = predict_garrison(tp['id'], eta)
                    if p_own == player:
                        if tp['is_doomed'] and eta < tp['min_threat_eta'] + 3:
                            needed = tp['defense_needed'] - committed[tp['id']]
                            if needed <= 0: continue
                            score = (tp['production'] * 18) / (eta + 1)
                            val_needed = int(needed)
                        else: continue
                    else:
                        needed = max(0, p_gar - committed.get(tp['id'], 0)) + 1
                        if avail < needed: continue
                        if committed.get(tp['id'], 0) > p_gar + 20: continue
                        
                        score = (tp['production'] + 0.6) / (needed * eta + 1) * 120
                        if tp['owner'] == -1: score *= 7.0
                        if tp['is_comet']: score *= 18.0
                        if history.get(tp['id'], (0,0))[1] > 0: score *= 5.0
                        
                        # Cluster & Symmetry
                        nearby_friends = [p for p in my_p if dist((p['x'], p['y']), (tp['x'], tp['y'])) < 28]
                        score *= (1.1 + 0.12 * len(nearby_friends))
                        if dist((tp['x'], tp['y']), (CENTER, CENTER)) < 36: score *= 1.9
                        
                        if is_ffa and tp['owner'] != -1:
                            if not am_leading and tp['owner'] == leader_id: score *= 3.0
                            elif am_leading: score *= 0.7 # Avoid over-extension when leading
                        
                        val_needed = int(needed)
                    
                    if score > best_s: best_s, best_t, best_a, best_needed = score, tp, angle, val_needed
                
                if best_t:
                    if best_t['owner'] == -1: send = min(avail, max(19, best_needed + neutral_buffer))
                    else:
                        send = min(avail, max(best_needed + neutral_buffer, avail))
                        if dispatches == 0 and avail > best_needed * 2.5 + 20: send = best_needed + 6
                    
                    if dist((mp['x'], mp['y']), (best_t['x'], best_t['y'])) > 38: send = max(send, 22)
                    
                    send = max(min_threshold, int(send))
                    if send > avail: send = int(avail)
                    moves.append([mp['id'], best_a, send])
                    avail -= send
                    committed[best_t['id']] += send
                    dispatches += 1
                else:
                    # PHANTOM HARASS
                    if avail > 15:
                        enemies = [p for p in all_p if p['owner'] != player and p['owner'] != -1]
                        if enemies:
                            target = random.choice(enemies)
                            if dist((mp['x'], mp['y']), (target['x'], target['y'])) < 70:
                                angle, _, _, _ = calc_intercept(mp, target, 1, av)
                                if not sun_blocked(mp['x'], mp['y'], target['x'], target['y']):
                                    moves.append([mp['id'], angle, 1])
                                    avail -= 1
                    break
        return moves
    except Exception: return []
