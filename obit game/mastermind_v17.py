import math

# ============================================================
# MASTERMIND v17.0 PROTECTOR — The Strategic Guardian
# ============================================================
# UPGRADES FROM v13.1:
#   1. REACTIVE DEFENSE (RESCUE): Can now send reinforcements to
#      owned planets under heavy threat.
#   2. THREAT TIMING AWARENESS: Distinguishes between immediate
#      and future threats to prioritize defense.
#   3. PRODUCTION-AWARE DISPATCH: Uses same-turn production for
#      slightly more aggressive fleet sizes.
#   4. REFINED ENDGAME: Prioritizes ship preservation in last 15 turns.
#   5. DISTANCE-WEIGHTED EXPANSION: Expansion score scales better
#      with distance to ensure efficient growth.
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
    angle = math.atan2(fy - sy, fx - sx)
    return angle, steps, fx, fy

def sun_blocked(sx, sy, fx, fy):
    return seg_dist((CENTER, CENTER), (sx, sy), (fx, fy)) <= SUN_RADIUS + 2.0

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
        fleets.append({'id': f[0], 'owner': f[1], 'x': f[2], 'y': f[3], 'angle': f[4], 'ships': f[6], 'owner_id': f[1]})
    
    all_p = list(planets.values())
    my_p = [p for p in all_p if p['owner'] == player]
    if not my_p: return []
    other_p = [p for p in all_p if p['owner'] != player]
    
    revenge_ids = set()
    for pid, p in planets.items():
        if pid in state['prev_owners'] and state['prev_owners'][pid] == player and p['owner'] != player:
            revenge_ids.add(pid)
        state['prev_owners'][pid] = p['owner']
    
    # --- Advanced Threat Analysis ---
    incoming_enemy = {p['id']: [] for p in all_p} # pid -> list of (ships, eta)
    incoming_friendly = {p['id']: [] for p in all_p}
    
    for f in fleets:
        speed = fleet_speed(f['ships'])
        for check_steps in range(1, 35): # Slightly longer lookahead
            fx = f['x'] + math.cos(f['angle']) * speed * check_steps
            fy = f['y'] + math.sin(f['angle']) * speed * check_steps
            hit_id = None
            for p in all_p:
                ppx, ppy = predict_pos(p['x'], p['y'], p['radius'], check_steps, av)
                if dist((fx, fy), (ppx, ppy)) < p['radius'] + 1.2:
                    hit_id = p['id']; break
            if hit_id is not None:
                if f['owner_id'] == player:
                    incoming_friendly[hit_id].append((f['ships'], check_steps))
                else:
                    incoming_enemy[hit_id].append((f['ships'], check_steps))
                break
    
    # Determine which of MY planets are truly threatened
    threatened_my_p = {} # pid -> ships needed for rescue
    for mp in my_p:
        threats = sorted(incoming_enemy[mp['id']], key=lambda x: x[1])
        if not threats: continue
        
        # Simple simulation to see if it falls
        curr_garrison = mp['ships']
        friendlies = sorted(incoming_friendly[mp['id']], key=lambda x: x[1])
        f_idx = 0
        last_t = 0
        for ts, teta in threats:
            # Production until threat arrives
            curr_garrison += mp['production'] * (teta - last_t)
            # Add friendlies arriving before/at threat
            while f_idx < len(friendlies) and friendlies[f_idx][1] <= teta:
                curr_garrison += friendlies[f_idx][0]
                f_idx += 1
            # Combat
            curr_garrison -= ts
            last_t = teta
            if curr_garrison < 0:
                threatened_my_p[mp['id']] = abs(curr_garrison) + 5
                break

    defense_reserve = {p['id']: 2 for p in my_p}
    for pid, needed in threatened_my_p.items():
        # Keep everything if threatened, but also ask for help
        defense_reserve[pid] = planets[pid]['ships'] 

    my_p.sort(key=lambda p: p['ships'], reverse=True)
    moves = []
    
    committed_this_turn = {} # target_id -> total ships sent this turn
    
    for mp in my_p:
        # Cannot use same-turn production for dispatch; the engine validates against current ships
        avail = mp['ships'] - defense_reserve.get(mp['id'], 2)
        if avail <= 0: continue
        
        dispatches = 0
        while avail > 5 and dispatches < 2:
            best_target, best_score, best_angle, best_needed = None, -1.0, 0, 0
            
            # Potential targets include other planets AND threatened my planets
            targets = other_p + [planets[pid] for pid in threatened_my_p if pid != mp['id']]
            
            for tp in targets:
                angle, eta, fx, fy = calc_intercept(mp['x'], mp['y'], tp['x'], tp['y'], tp['radius'], avail, av)
                if step + eta >= 495: continue
                if sun_blocked(mp['x'], mp['y'], fx, fy): continue
                
                is_rescue = (tp['owner'] == player)
                
                if is_rescue:
                    needed = threatened_my_p[tp['id']]
                    if avail < needed: continue
                    # Score rescue based on production lost if it falls
                    score = (tp['production'] * 12) / (eta + 1)
                else:
                    # Normal Attack
                    enemy_prod = tp['production'] if tp['owner'] != -1 else 0
                    required = tp['ships'] + (enemy_prod * eta)
                    # Add enemy reinforcements
                    for rs, reta in incoming_enemy[tp['id']]:
                        if reta <= eta: required += rs
                    # Subtract our fleets already in flight
                    already = sum(s for s, e in incoming_friendly[tp['id']]) + committed_this_turn.get(tp['id'], 0)
                    needed = max(0, required - already) + 1
                    
                    if avail < needed: continue
                    if already > required + 10: continue
                    
                    score = (tp['production'] + 0.1) / (eta + 1)
                    if tp['owner'] == -1: 
                        # Distance penalization for expansion to avoid over-extending
                        dist_factor = 1.0 - (eta / 100.0) 
                        score *= 3.5 * max(0.1, dist_factor)
                    if tp['id'] in revenge_ids: score *= 8.0
                    if tp['is_comet']: score *= 6.0
                    if tp['owner'] != -1 and tp['ships'] > 50: score *= 0.4
                
                if score > best_score:
                    best_score, best_target, best_angle, best_needed = score, tp, angle, int(needed)
            
            if best_target:
                # Send amount
                send = avail
                if dispatches == 0 and avail > best_needed * 2 + 10: 
                    send = best_needed + 5
                    
                    # Recalculate angle with smaller fleet size (slower speed)
                    best_angle, new_eta, _, _ = calc_intercept(mp['x'], mp['y'], best_target['x'], best_target['y'], best_target['radius'], send, av)
                    
                    # Verify if it's still enough with the longer flight time
                    enemy_prod = best_target['production'] if best_target['owner'] != -1 else 0
                    required = best_target['ships'] + (enemy_prod * new_eta)
                    for rs, reta in incoming_enemy[best_target['id']]:
                        if reta <= new_eta: required += rs
                    already = sum(s for s, e in incoming_friendly[best_target['id']]) + committed_this_turn.get(best_target['id'], 0)
                    new_needed = max(0, required - already) + 1
                    
                    if send < new_needed + 3:
                        # If not enough, fallback to sending everything to be safe
                        send = avail
                        best_angle, _, _, _ = calc_intercept(mp['x'], mp['y'], best_target['x'], best_target['y'], best_target['radius'], send, av)
                
                moves.append([mp['id'], best_angle, int(send)])
                avail -= send
                committed_this_turn[best_target['id']] = committed_this_turn.get(best_target['id'], 0) + send
                dispatches += 1
            else: break
            
    return moves
