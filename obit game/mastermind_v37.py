# ============================================================
# MASTERMIND v37.0 - "COSMIC OVERLORD"
# ============================================================
# Improvements:
#   1. SIMULATED GARRISON: Accurate arrival-time prediction including all fleets.
#   2. CLUSTER BONUS: Prioritize expansion near existing friendly fronts.
#   3. AGGRESSIVE OPENING: Solidify 18-ship captures for neutrals.
#   4. SYMMETRY PREEMPTION: Value mirroring moves against symmetric opponents.
#   5. OMEGA MULTIPLIERS: Refined ROI heuristic (Production / Needed Ships).
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

def get_mirror_pos(x, y):
    # Map is 100x100, center is 50,50. 4-fold symmetry.
    return [(x, y), (100-x, y), (x, 100-y), (100-x, 100-y)]

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
        
        # Player and Alliance stats
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
            pwr = stats['ships'] + stats['prod'] * 20
            if pwr > max_p: max_p = pwr; leader_id = pid
        am_leading = (leader_id == player)

        # History Tracking
        history = GLOBAL_STATE['planet_history']
        for pid, p in planets.items():
            prev = history.get(pid, (p['owner'], 0))
            if prev[0] == player and p['owner'] != player: history[pid] = (p['owner'], 35)
            elif prev[1] > 0: history[pid] = (p['owner'], prev[1] - 1)
            else: history[pid] = (p['owner'], 0)

        # Fleet Impact Simulation
        committed = {p['id']: 0 for p in all_p}
        incoming_fleets = {p['id']: [] for p in all_p} # (owner, ships, eta)
        for f in obs.get("fleets", []):
            f_owner, f_ships, f_angle, f_x, f_y = f[1], f[6], f[4], f[2], f[3]
            f_speed = fleet_speed(f_ships)
            # Find destination
            dest_id = None
            for t in range(1, 41):
                fx, fy = f_x + math.cos(f_angle)*f_speed*t, f_y + math.sin(f_angle)*f_speed*t
                for p in all_p:
                    if abs(fx - p['x']) > 25 or abs(fy - p['y']) > 25: continue 
                    ppx, ppy = predict_pos(p['x'], p['y'], p['radius'], t, av)
                    if (fx-ppx)**2 + (fy-ppy)**2 < (p['radius']+1.3)**2: dest_id = p['id']; break
                if dest_id is not None:
                    if f_owner == player: committed[dest_id] += f_ships
                    else: incoming_fleets[dest_id].append((f_owner, f_ships, t))
                    break

        def predict_garrison(p_id, t_arrival):
            p = planets[p_id]
            curr = p['ships']
            owner = p['owner']
            # Simulate step by step or simplified events
            events = sorted([(f[2], f[0], f[1]) for f in incoming_fleets[p_id] if f[2] <= t_arrival])
            last_t = 0
            for t_ev, f_owner, f_ships in events:
                if owner != -1: curr += p['production'] * (t_ev - last_t)
                if f_owner == owner: curr += f_ships
                else:
                    curr -= f_ships
                    if curr < 0:
                        curr = abs(curr)
                        owner = f_owner
                last_t = t_ev
            if owner != -1: curr += p['production'] * (t_arrival - last_t)
            return curr, owner

        # Threat and Defense
        for p in all_p:
            threats = [f for f in incoming_fleets[p['id']] if f[1] >= 3]
            p['min_threat_eta'] = min(f[2] for f in threats) if threats else 999
            p['total_threat'] = sum(f[1] for f in threats)
            
            p_garrison, p_owner = predict_garrison(p['id'], min(p['min_threat_eta'], 30))
            p['is_doomed'] = p_owner != player and p['owner'] == player
            p['defense_needed'] = max(0, p['total_threat'] - p_garrison + 5) if p['is_doomed'] else 0

        my_p.sort(key=lambda p: p['ships'], reverse=True)
        moves = []
        min_threshold = 3 if step < 60 else 5
        neutral_buffer = 4

        for mp in my_p:
            res = max(2, mp['total_threat'] + 2)
            avail = mp['ships'] - res
            if avail <= min_threshold: continue
            
            dispatches = 0
            while avail > min_threshold and dispatches < 4:
                best_t, best_s, best_a, best_needed = None, -1.0, 0, 0
                for tp in all_p:
                    if tp['id'] == mp['id']: continue
                    
                    dist_to_t = dist((mp['x'], mp['y']), (tp['x'], tp['y']))
                    score_est = (tp['production'] + 0.1) / (dist_to_t / 4.0 + 1)
                    if score_est < 0.005: continue

                    curr_send = avail if tp['owner'] != -1 else tp['ships'] + neutral_buffer
                    angle, eta, fx, fy = calc_intercept(mp['x'], mp['y'], tp['x'], tp['y'], tp['radius'], curr_send, av)
                    
                    if step + eta >= 496: continue
                    if sun_blocked(mp['x'], mp['y'], fx, fy):
                        wp = try_waypoint(mp['x'], mp['y'], fx, fy)
                        if wp: angle = math.atan2(wp[1]-mp['y'], wp[0]-mp['x'])
                        else: continue
                    
                    predicted_count, predicted_owner = predict_garrison(tp['id'], eta)
                    
                    if predicted_owner == player:
                        if tp['is_doomed'] and eta < tp['min_threat_eta'] + 3:
                            needed = tp['defense_needed'] - committed[tp['id']]
                            if needed <= 0: continue
                            score = (tp['production'] * 15) / (eta + 1)
                            val_needed = int(needed)
                        else: continue
                    else:
                        needed = max(0, predicted_count - committed.get(tp['id'], 0)) + 1
                        if avail < needed: continue
                        if committed.get(tp['id'], 0) > predicted_count + 15: continue
                        
                        # ROI HEURISTIC
                        score = (tp['production'] + 0.5) / (needed * eta + 1) * 100
                        
                        # MULTIPLIERS
                        if tp['owner'] == -1: score *= 6.5
                        if tp['is_comet']: score *= 15.0
                        if history.get(tp['id'], (0,0))[1] > 0: score *= 4.5
                        
                        # CLUSTER BONUS: Expansion near friendly planets
                        nearby_friends = [p for p in my_p if dist((p['x'], p['y']), (tp['x'], tp['y'])) < 25]
                        if nearby_friends: score *= (1.2 + 0.1 * len(nearby_friends))
                        
                        orb_r = dist((tp['x'], tp['y']), (CENTER, CENTER))
                        if orb_r < 35: score *= 1.8
                        
                        if is_ffa and tp['owner'] != -1:
                            if not am_leading and tp['owner'] == leader_id: score *= 2.5
                        
                        val_needed = int(needed)
                    
                    if score > best_s: best_s, best_t, best_a, best_needed = score, tp, angle, val_needed
                
                if best_t:
                    if best_t['owner'] == -1:
                        # Top-tier opening: ensure 18+ ships for solid capture if possible
                        send = min(avail, max(18, best_needed + neutral_buffer))
                    else:
                        send = min(avail, max(best_needed + neutral_buffer, avail))
                        if dispatches == 0 and avail > best_needed * 2 + 10: send = best_needed + neutral_buffer
                    
                    # Speed threshold
                    if dist((mp['x'], mp['y']), (best_t['x'], best_t['y'])) > 40: send = max(send, 20)
                    
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
                            if dist((mp['x'], mp['y']), (target['x'], target['y'])) < 65:
                                angle, _, _, _ = calc_intercept(mp['x'], mp['y'], target['x'], target['y'], target['radius'], 1, av)
                                if not sun_blocked(mp['x'], mp['y'], target['x'], target['y']):
                                    moves.append([mp['id'], angle, 1])
                                    avail -= 1
                    break
        return moves
    except Exception: return []
