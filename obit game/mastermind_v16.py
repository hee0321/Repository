import math

# ============================================================
# MASTERMIND v16.0 LEGEND — The Perfected v13
# ============================================================
# v13 was the peak. v14/v15 failed by over-complicating.
# v16 RESTORES v13 and adds only the most reliable features:
#   1. RESTORED: Continuous step-by-step threat projection.
#   2. RESTORED: Aggressive scoring multipliers.
#   3. ADDED: Waypoint Sun Detour (if direct path is blocked).
#   4. ADDED: Basic step 480+ endgame safety.
# ============================================================

BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0

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
    steps = dist((sx, sy), (tx, ty)) / speed
    fx, fy = tx, ty
    for _ in range(15):
        fx, fy = predict_pos(tx, ty, tr, steps, av)
        steps = dist((sx, sy), (fx, fy)) / speed
    return math.atan2(fy - sy, fx - sx), steps, fx, fy

def sun_blocked(sx, sy, fx, fy):
    return seg_dist((CENTER, CENTER), (sx, sy), (fx, fy)) <= SUN_RADIUS + 2.0

def try_waypoint(sx, sy, tx, ty):
    mx, my = (sx + tx) / 2, (sy + ty) / 2
    dx, dy = tx - sx, ty - sy
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6: return None
    px, py = -dy / length, dx / length
    offset = SUN_RADIUS + 8.5
    w1 = (mx + px * offset, my + py * offset)
    w2 = (mx - px * offset, my - py * offset)
    wp = w1 if dist(w1, (CENTER, CENTER)) > dist(w2, (CENTER, CENTER)) else w2
    if 2 < wp[0] < 98 and 2 < wp[1] < 98:
        if not sun_blocked(sx, sy, wp[0], wp[1]): return wp
    return None

def agent(obs):
    if not hasattr(agent, '_s'): agent._s = {}
    player = obs.get("player", 0)
    step = obs.get("step", 0)
    if player not in agent._s or step == 0: agent._s[player] = {'prev_owners': {}}
    state = agent._s[player]
    
    av = obs.get("angular_velocity", 0.0)
    comet_ids = set(obs.get("comet_planet_ids", []))
    
    planets = {}
    for p in obs.get("planets", []):
        planets[p[0]] = {
            'id': p[0], 'owner': p[1], 'x': p[2], 'y': p[3],
            'radius': p[4], 'ships': p[5], 'production': p[6],
            'is_comet': p[0] in comet_ids
        }
    
    fleets = []
    for f in obs.get("fleets", []):
        fleets.append({'id': f[0], 'owner': f[1], 'x': f[2], 'y': f[3], 'angle': f[4], 'ships': f[6]})
    
    all_p = list(planets.values())
    my_p = [p for p in all_p if p['owner'] == player]
    if not my_p: return []
    other_p = [p for p in all_p if p['owner'] != player]
    
    revenge_ids = set()
    for pid, p in planets.items():
        if pid in state['prev_owners'] and state['prev_owners'][pid] == player and p['owner'] != player:
            revenge_ids.add(pid)
        state['prev_owners'][pid] = p['owner']
    
    # --- PRECISION THREAT TRACKING (v13 Restore) ---
    incoming_threats = {p['id']: 0 for p in my_p}
    enemy_reinforcements = {}
    committed_targets = {}
    
    for f in fleets:
        speed = fleet_speed(f['ships'])
        # Scan every step to never miss a fleet (CRITICAL)
        for check_steps in range(1, 30):
            fx = f['x'] + math.cos(f['angle']) * speed * check_steps
            fy = f['y'] + math.sin(f['angle']) * speed * check_steps
            hit_id = None
            for p in all_p:
                ppx, ppy = predict_pos(p['x'], p['y'], p['radius'], check_steps, av)
                if dist((fx, fy), (ppx, ppy)) < p['radius'] + 1.2:
                    hit_id = p['id']; break
            if hit_id is not None:
                if f['owner'] == player:
                    committed_targets[hit_id] = committed_targets.get(hit_id, 0) + f['ships']
                else:
                    if hit_id in incoming_threats: incoming_threats[hit_id] += f['ships']
                    elif planets[hit_id]['owner'] == f['owner']:
                        enemy_reinforcements[hit_id] = enemy_reinforcements.get(hit_id, 0) + f['ships']
                break
    
    defense_reserve = {p['id']: max(2, incoming_threats.get(p['id'], 0) + 2) for p in my_p}
    my_p.sort(key=lambda p: p['ships'], reverse=True)
    moves = []
    
    for mp in my_p:
        reserve = defense_reserve.get(mp['id'], 2)
        avail = mp['ships'] - reserve
        if avail <= 0: continue
        
        dispatches = 0
        while avail > 5 and dispatches < 2:
            best_target, best_score, best_angle, best_needed = None, -1.0, 0, 0
            for tp in other_p:
                angle, eta, fx, fy = calc_intercept(mp['x'], mp['y'], tp['x'], tp['y'], tp['radius'], avail, av)
                if step + eta >= 498: continue
                
                # Waypoint Sun Avoidance
                if sun_blocked(mp['x'], mp['y'], fx, fy):
                    wp = try_waypoint(mp['x'], mp['y'], fx, fy)
                    if wp:
                        angle = math.atan2(wp[1] - mp['y'], wp[0] - mp['x'])
                    else: continue
                
                required = tp['ships'] + (tp['production'] if tp['owner'] != -1 else 0) * eta
                required += enemy_reinforcements.get(tp['id'], 0)
                already = committed_targets.get(tp['id'], 0)
                needed = max(0, required - already) + 1
                
                if avail < needed: continue
                if already > required + 10: continue
                
                # Scoring (v13 Aggressive)
                score = (tp['production'] + 0.1) / (eta + 1)
                if tp['owner'] == -1: score *= 2.5
                if tp['id'] in revenge_ids: score *= 8.0
                if tp['is_comet']: score *= 5.0
                if tp['owner'] != -1 and tp['ships'] > 50: score *= 0.5
                
                if score > best_score:
                    best_score, best_target, best_angle, best_needed = score, tp, angle, int(needed)
            
            if best_target:
                send = min(avail, max(best_needed + 3, avail))
                if dispatches == 0 and avail > best_needed * 2 + 10: send = best_needed + 5
                moves.append([mp['id'], best_angle, int(send)])
                avail -= send
                committed_targets[best_target['id']] = committed_targets.get(best_target['id'], 0) + send
                dispatches += 1
            else: break
            
    return moves
