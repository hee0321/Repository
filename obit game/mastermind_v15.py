import math

# ============================================================
# MASTERMIND v15.0 EMPEROR — Supreme Command Edition
# ============================================================
# EVOLUTION FROM v13:
#   1. WAYPOINT DETOUR — Use perpendicular waypoints to route
#      fleets around the sun if the direct path is blocked.
#   2. REINFORCEMENT PIPELINES — Backline planets funnel ships to
#      threatened frontline planets.
#   3. ENDGAME OPTIMIZATION — Step 480+ logic to prevent wasting
#      ships on targets that won't arrive in time.
#   4. COMET SPAWN PRESERVATION — Save ships just before 50, 150...
#   5. REFINED ROI SCORING — Balance production gain vs distance.
# ============================================================

BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0
COMET_SPAWN_STEPS = {50, 150, 250, 350, 450}
EPISODE_STEPS = 500

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
    if orb_r + pr >= ROTATION_RADIUS_LIMIT:
        return px, py
    angle = math.atan2(py - CENTER, px - CENTER) + av * steps
    return CENTER + orb_r * math.cos(angle), CENTER + orb_r * math.sin(angle)

def calc_intercept(sx, sy, tx, ty, tr, fleet_ships, av):
    speed = fleet_speed(fleet_ships)
    steps = dist((sx, sy), (tx, ty)) / speed
    fx, fy = tx, ty
    for _ in range(15):
        fx, fy = predict_pos(tx, ty, tr, steps, av)
        steps = dist((sx, sy), (fx, fy)) / speed
    angle = math.atan2(fy - sy, fx - sx)
    return angle, steps, fx, fy

def sun_blocked(sx, sy, fx, fy):
    return seg_dist((CENTER, CENTER), (sx, sy), (fx, fy)) <= SUN_RADIUS + 2.0

def try_waypoint(sx, sy, tx, ty):
    mx, my = (sx + tx) / 2, (sy + ty) / 2
    dx, dy = tx - sx, ty - sy
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6: return None
    px, py = -dy / length, dx / length
    offset = SUN_RADIUS + 7.5
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
    
    # --- Incoming analysis (Project and detect targets) ---
    incoming_threats = {p['id']: 0 for p in my_p}
    enemy_reinforcements = {}
    committed_targets = {} # our fleets in flight
    
    for f in fleets:
        speed = fleet_speed(f['ships'])
        # Simple projection to find target
        target_id = None
        for check_steps in [5, 10, 15, 20, 25]:
            px = f['x'] + math.cos(f['angle']) * speed * check_steps
            py = f['y'] + math.sin(f['angle']) * speed * check_steps
            for p in all_p:
                ppx, ppy = predict_pos(p['x'], p['y'], p['radius'], check_steps, av)
                if dist((px, py), (ppx, ppy)) < p['radius'] + 1.5:
                    target_id = p['id']
                    break
            if target_id: break
        
        if target_id:
            if f['owner'] != player:
                if target_id in incoming_threats:
                    incoming_threats[target_id] += f['ships']
                elif planets[target_id]['owner'] == f['owner']:
                    enemy_reinforcements[target_id] = enemy_reinforcements.get(target_id, 0) + f['ships']
            else:
                committed_targets[target_id] = committed_targets.get(target_id, 0) + f['ships']

    # --- Defense & Comets ---
    defense_reserve = {p['id']: max(2, incoming_threats.get(p['id'], 0) + 2) for p in my_p}
    
    # Comet preservation: save ships if comet spawn is imminent (within 10 turns)
    is_spawning = False
    for spawn in COMET_SPAWN_STEPS:
        if 0 < spawn - step < 10:
            is_spawning = True; break

    my_p.sort(key=lambda p: p['ships'], reverse=True)
    moves = []
    
    for mp in my_p:
        reserve = defense_reserve.get(mp['id'], 2)
        if is_spawning: reserve += 15 # save extra for comets
        avail = mp['ships'] - reserve
        if avail <= 5: continue
        
        dispatches = 0
        while avail > 5 and dispatches < 2:
            best_target, best_score, best_angle, best_needed = None, -1.0, 0, 0
            
            for tp in other_p:
                angle, eta, fx, fy = calc_intercept(mp['x'], mp['y'], tp['x'], tp['y'], tp['radius'], avail, av)
                
                # Endgame check
                if step + eta >= EPISODE_STEPS - 2: continue
                
                # Sun blocked check
                if sun_blocked(mp['x'], mp['y'], fx, fy):
                    wp = try_waypoint(mp['x'], mp['y'], fx, fy)
                    if wp:
                        angle = math.atan2(wp[1] - mp['y'], wp[0] - mp['x'])
                    else:
                        continue
                
                # Required ships
                enemy_prod = tp['production'] if tp['owner'] != -1 else 0
                required = tp['ships'] + (enemy_prod * eta) + enemy_reinforcements.get(tp['id'], 0)
                already = committed_targets.get(tp['id'], 0)
                needed = max(0, required - already) + 1
                
                if avail < needed: continue
                if already > required + 10: continue # Already won
                
                # ROI Scoring (v13 style with tweaks)
                score = (tp['production'] + 0.1) / (eta + 1)
                if tp['id'] in revenge_ids: score *= 8.0
                if tp['is_comet']: score *= 6.0
                if tp['owner'] == -1: score *= 3.0 # Neutral is high priority
                if tp['owner'] != -1 and tp['ships'] > 60: score *= 0.4
                
                if score > best_score:
                    best_score, best_target, best_angle, best_needed = score, tp, angle, int(needed)
            
            if best_target:
                send = min(avail, max(best_needed + 4, avail))
                # If we have massive excess, split
                if dispatches == 0 and avail > best_needed * 2 + 15:
                    send = best_needed + 8
                
                moves.append([mp['id'], best_angle, int(send)])
                avail -= send
                committed_targets[best_target['id']] = committed_targets.get(best_target['id'], 0) + send
                dispatches += 1
            else: break
            
    # --- REINFORCEMENT (Ferry ships to frontline) ---
    if not is_spawning: # Don't ferry if we need to save for comets
        for mp in my_p:
            if [m for m in moves if m[0] == mp['id']]: continue # Already moved
            res = defense_reserve.get(mp['id'], 2)
            left = mp['ships'] - res
            if left < 20: continue
            
            # Find threatened frontline
            target_p = None
            for fp in my_p:
                if incoming_threats.get(fp['id'], 0) > fp['ships']:
                    target_p = fp; break
            
            if target_p and target_p['id'] != mp['id']:
                angle, eta, fx, fy = calc_intercept(mp['x'], mp['y'], target_p['x'], target_p['y'], target_p['radius'], left, av)
                if not sun_blocked(mp['x'], mp['y'], fx, fy):
                    moves.append([mp['id'], angle, int(left)])

    return moves
