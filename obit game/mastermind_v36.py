# ============================================================
# MASTERMIND v36.0 - "PHANTOM HARASS"
# ============================================================
# Improvements:
#   1. SMART THREAT FILTER: Ignore tiny threats (<3 ships) in defense logic.
#   2. HARASSMENT: Send small "phantom" fleets to harass enemies.
#   3. OMEGA MULTIPLIERS: High neutral (6.0) and comet (15.0) weight.
#   4. SPEED AWARENESS: Distant fleets must be large enough.
#   5. REVENGE FOCUS: Priority window 35 turns.
# ============================================================

import math
import random

BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0

GLOBAL_STATE = {'planet_history': {}}

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

def predict_pos(px, py, pr, steps, av):
    orb_r = dist((px, py), (CENTER, CENTER))
    if orb_r + pr >= ROTATION_RADIUS_LIMIT: return px, py
    angle = math.atan2(py - CENTER, px - CENTER) + av * steps
    return CENTER + orb_r * math.cos(angle), CENTER + orb_r * math.sin(angle)

def calc_intercept(sx, sy, tx, ty, tr, fleet_ships, av):
    speed = fleet_speed(fleet_ships)
    prev_steps = dist((sx, sy), (tx, ty)) / speed
    steps = prev_steps
    fx, fy = tx, ty
    for _ in range(10):
        fx, fy = predict_pos(tx, ty, tr, steps, av)
        steps = dist((sx, sy), (fx, fy)) / speed
        if abs(steps - prev_steps) < 0.1: break
        prev_steps = steps
    return math.atan2(fy - sy, fx - sx), steps, fx, fy

def sun_blocked(sx, sy, fx, fy, buffer=2.3):
    return seg_dist((CENTER, CENTER), (sx, sy), (fx, fy)) <= SUN_RADIUS + buffer

def try_waypoint(sx, sy, fx, fy, buffer=2.3):
    mx, my = (sx + fx) / 2, (sy + fy) / 2
    dx, dy = fx - sx, fy - sy
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6: return None
    px, py = -dy / length, dx / length
    offset = SUN_RADIUS + 10.0
    w1 = (mx + px * offset, my + py * offset)
    w2 = (mx - px * offset, my - py * offset)
    wp = w1 if dist(w1, (CENTER, CENTER)) > dist(w2, (CENTER, CENTER)) else w2
    if 2 < wp[0] < 98 and 2 < wp[1] < 98:
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
        all_p = list(planets.values())
        my_p = [p for p in all_p if p['owner'] == player]
        if not my_p: return []
        
        player_strength = {p: {'ships': 0, 'prod': 0} for p in range(4)}
        for p in all_p:
            if p['owner'] >= 0:
                player_strength[p['owner']]['ships'] += p['ships']
                player_strength[p['owner']]['prod'] += p['production']
        for f in obs.get("fleets", []):
            if f[1] >= 0: player_strength[f[1]]['ships'] += f[6]
        
        is_ffa = len([pid for pid, s in player_strength.items() if s['ships'] > 0 or s['prod'] > 0]) >= 3
        
        my_stats = player_strength.get(player, {'ships': 0, 'prod': 0})
        leader_id = player
        max_p = -1
        for pid, stats in player_strength.items():
            pwr = stats['ships'] + stats['prod'] * 20
            if pwr > max_p: max_p = pwr; leader_id = pid
        am_leading = (leader_id == player)

        history = GLOBAL_STATE['planet_history']
        for pid, p in planets.items():
            prev = history.get(pid, (p['owner'], 0))
            if prev[0] == player and p['owner'] != player: history[pid] = (p['owner'], 35)
            elif prev[1] > 0: history[pid] = (p['owner'], prev[1] - 1)
            else: history[pid] = (p['owner'], 0)

        committed = {p['id']: 0 for p in all_p}
        incoming = {p['id']: [] for p in all_p}
        for f in obs.get("fleets", []):
            f_owner, f_ships, f_angle, f_x, f_y = f[1], f[6], f[4], f[2], f[3]
            f_speed = fleet_speed(f_ships)
            for t in range(1, 41):
                fx, fy = f_x + math.cos(f_angle)*f_speed*t, f_y + math.sin(f_angle)*f_speed*t
                hit_id = None
                for p in all_p:
                    if abs(fx - p['x']) > 25 or abs(fy - p['y']) > 25: continue 
                    ppx, ppy = predict_pos(p['x'], p['y'], p['radius'], t, av)
                    if (fx-ppx)**2 + (fy-ppy)**2 < (p['radius']+1.3)**2: hit_id = p['id']; break
                if hit_id is not None:
                    if f_owner == player: committed[hit_id] += f_ships
                    else: incoming[hit_id].append((f_owner, f_ships, t))
                    break
        
        for p in all_p:
            threats = incoming[p['id']]
            # SMART FILTER: Ignore tiny threats for defense calculation
            effective_threats = [f for f in threats if f[1] >= 3]
            p['min_threat_eta'] = min(f[2] for f in effective_threats) if effective_threats else 999
            p['total_threat'] = sum(f[1] for f in effective_threats)
            
            expected = p['ships'] + (p['production'] * min(p['min_threat_eta'], 30))
            p['is_doomed'] = p['total_threat'] > expected + 2 and p['min_threat_eta'] < 25
            p['defense_needed'] = max(0, p['total_threat'] - expected + 5) if p['is_doomed'] else 0

        my_p.sort(key=lambda p: p['ships'], reverse=True)
        moves = []
        min_threshold = 3 if step < 60 else 5
        neutral_buffer = 4

        for mp in my_p:
            res = max(2, mp['total_threat'] + 2)
            avail = mp['ships'] - res
            if avail <= min_threshold: continue
            
            dispatches = 0
            while avail > min_threshold and dispatches < 3:
                best_t, best_s, best_a, best_needed = None, -1.0, 0, 0
                for tp in all_p:
                    if tp['id'] == mp['id']: continue
                    
                    score_estimate = (tp['production'] + 0.1) / (dist((mp['x'], mp['y']), (tp['x'], tp['y'])) / 4.0 + 1)
                    if score_estimate < 0.005: continue

                    curr_send = avail if tp['owner'] != -1 else tp['ships'] + neutral_buffer
                    angle, eta, fx, fy = calc_intercept(mp['x'], mp['y'], tp['x'], tp['y'], tp['radius'], curr_send, av)
                    if tp['owner'] == -1:
                        curr_send = min(avail, tp['ships'] + (tp['production'] * eta) + neutral_buffer)
                        angle, eta, fx, fy = calc_intercept(mp['x'], mp['y'], tp['x'], tp['y'], tp['radius'], curr_send, av)
                    
                    if step + eta >= 495: continue
                    if sun_blocked(mp['x'], mp['y'], fx, fy):
                        wp = try_waypoint(mp['x'], mp['y'], fx, fy)
                        if wp: angle = math.atan2(wp[1]-mp['y'], wp[0]-mp['x'])
                        else: continue
                    
                    if tp['owner'] == player:
                        if tp['is_doomed'] and eta < tp['min_threat_eta'] + 3:
                            needed = tp['defense_needed'] - committed[tp['id']]
                            if needed <= 0: continue
                            score = (tp['production'] * 15) / (eta + 1)
                            val_needed = int(needed)
                        else: continue
                    else:
                        req = tp['ships'] + (tp['production'] if tp['owner'] != -1 else 0) * eta
                        already = committed.get(tp['id'], 0)
                        needed = max(0, req - already) + 1
                        if avail < needed: continue
                        if already > req + 10: continue
                        
                        score = (tp['production'] + 0.2) / (eta + 1)
                        if tp['owner'] == -1: score *= 6.0
                        if tp['is_comet']: score *= 15.0
                        if tp['owner'] != -1 and tp['ships'] > 50: score *= 0.4
                        if history.get(tp['id'], (0,0))[1] > 0: score *= 4.0
                        
                        orb_r = dist((tp['x'], tp['y']), (CENTER, CENTER))
                        if orb_r < 35: score *= 2.0
                        
                        if is_ffa and tp['owner'] != -1:
                            if not am_leading and tp['owner'] == leader_id: score *= 2.0
                        
                        val_needed = int(needed)
                    
                    if score > best_s: best_s, best_t, best_a, best_needed = score, tp, angle, val_needed
                
                if best_t:
                    if best_t['owner'] == -1: send = min(avail, best_needed + neutral_buffer)
                    else:
                        send = min(avail, max(best_needed + neutral_buffer, avail))
                        if dispatches == 0 and avail > best_needed * 2 + 10: send = best_needed + neutral_buffer
                    
                    # Speed constraint
                    if dist((mp['x'], mp['y']), (best_t['x'], best_t['y'])) > 40:
                        send = max(send, 20)
                    
                    send = max(min_threshold, int(send))
                    if send > avail: send = int(avail)
                    moves.append([mp['id'], best_a, send])
                    avail -= send
                    committed[best_t['id']] += send
                    dispatches += 1
                else:
                    # HARASSMENT LOGIC: If no good target found, send a phantom fleet
                    if avail > 15:
                        enemies = [p for p in all_p if p['owner'] != player and p['owner'] != -1]
                        if enemies:
                            target = random.choice(enemies)
                            # Only harass if close enough
                            if dist((mp['x'], mp['y']), (target['x'], target['y'])) < 60:
                                angle, _, _, _ = calc_intercept(mp['x'], mp['y'], target['x'], target['y'], target['radius'], 1, av)
                                if not sun_blocked(mp['x'], mp['y'], target['x'], target['y']):
                                    moves.append([mp['id'], angle, 1])
                                    avail -= 1
                    break
        return moves
    except Exception: return []
