import math

# ============================================================
# MASTERMIND v13.0 OVERLORD — Competition-Grade Strategic AI
# ============================================================
# KEY IMPROVEMENTS OVER v12:
#   1. INCOMING FLEET TRACKING — Account for enemy fleets heading
#      toward our planets and their planets when calculating defense
#      and attack requirements.
#   2. DUPLICATE TARGET AVOIDANCE — Track which planets we've already
#      committed fleets to, so we don't waste multiple attacks on
#      the same target.
#   3. DEFENSIVE RESERVE — Keep enough ships on high-production planets
#      to survive incoming enemy fleets.
#   4. ARRIVAL-TIME OWNERSHIP — Predict garrison at arrival time
#      including production AND incoming fleet reinforcements.
#   5. MULTI-DISPATCH PER PLANET — If a planet has excess ships after
#      the first dispatch, it can send to a second target.
#   6. COMET SPAWN AWARENESS — Know comet spawn steps.
# ============================================================

BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0
COMET_SPAWN_STEPS = {50, 150, 250, 350, 450}

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def seg_dist(p, v, w):
    """Min distance from point p to segment v-w."""
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0: return dist(p, v)
    t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
    return dist(p, (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1])))

def fleet_speed(ships):
    ships = max(1, ships)
    s = 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(s, MAX_SPEED)

def predict_pos(px, py, pr, steps, av):
    """Predict planet position after `steps` turns of rotation."""
    orb_r = dist((px, py), (CENTER, CENTER))
    if orb_r + pr >= ROTATION_RADIUS_LIMIT:
        return px, py
    angle = math.atan2(py - CENTER, px - CENTER) + av * steps
    return CENTER + orb_r * math.cos(angle), CENTER + orb_r * math.sin(angle)

def calc_intercept(sx, sy, tx, ty, tr, fleet_ships, av):
    """Iterative intercept: returns (angle, flight_time, fx, fy)."""
    speed = fleet_speed(fleet_ships)
    steps = dist((sx, sy), (tx, ty)) / speed
    fx, fy = tx, ty
    for _ in range(15):
        fx, fy = predict_pos(tx, ty, tr, steps, av)
        steps = dist((sx, sy), (fx, fy)) / speed
    angle = math.atan2(fy - sy, fx - sx)
    return angle, steps, fx, fy

def sun_blocked(sx, sy, fx, fy):
    """Check if fleet path crosses the sun."""
    return seg_dist((CENTER, CENTER), (sx, sy), (fx, fy)) <= SUN_RADIUS + 2.0

def agent(obs):
    # --- Per-player state ---
    if not hasattr(agent, '_s'):
        agent._s = {}
    player = obs.get("player", 0)
    step = obs.get("step", 0)
    if player not in agent._s or step == 0:
        agent._s[player] = {'prev_owners': {}}
    state = agent._s[player]
    
    av = obs.get("angular_velocity", 0.0)
    comet_ids = set(obs.get("comet_planet_ids", []))
    
    # --- Parse planets ---
    planets = {}
    for p in obs.get("planets", []):
        planets[p[0]] = {
            'id': p[0], 'owner': p[1], 'x': p[2], 'y': p[3],
            'radius': p[4], 'ships': p[5], 'production': p[6],
            'is_comet': p[0] in comet_ids
        }
    
    # --- Parse fleets (critical for defense & attack prediction) ---
    fleets = []
    for f in obs.get("fleets", []):
        fleets.append({
            'id': f[0], 'owner': f[1], 'x': f[2], 'y': f[3],
            'angle': f[4], 'from_planet': f[5], 'ships': f[6]
        })
    
    all_p = list(planets.values())
    my_planets = [p for p in all_p if p['owner'] == player]
    if not my_planets:
        return []
    other_planets = [p for p in all_p if p['owner'] != player]
    
    # --- Revenge detection ---
    revenge_ids = set()
    for pid, p in planets.items():
        if pid in state['prev_owners'] and state['prev_owners'][pid] == player and p['owner'] != player:
            revenge_ids.add(pid)
        state['prev_owners'][pid] = p['owner']
    
    # --- Calculate incoming threats to MY planets ---
    # For each of my planets, estimate how many enemy ships are heading toward it
    incoming_threats = {p['id']: 0 for p in my_planets}
    incoming_friendly = {p['id']: 0 for p in my_planets}
    
    for f in fleets:
        # Estimate where this fleet is heading by projecting its trajectory
        speed = fleet_speed(f['ships'])
        # Project fleet forward by ~20 steps
        for check_steps in range(1, 25):
            proj_x = f['x'] + math.cos(f['angle']) * speed * check_steps
            proj_y = f['y'] + math.sin(f['angle']) * speed * check_steps
            # Check if it would hit any of my planets
            for mp in my_planets:
                mpx, mpy = predict_pos(mp['x'], mp['y'], mp['radius'], check_steps, av)
                if dist((proj_x, proj_y), (mpx, mpy)) < mp['radius'] + 1.0:
                    if f['owner'] != player:
                        incoming_threats[mp['id']] += f['ships']
                    else:
                        incoming_friendly[mp['id']] += f['ships']
                    break
            else:
                continue
            break
    
    # --- Calculate incoming enemy reinforcements to THEIR planets ---
    enemy_reinforcements = {}
    for f in fleets:
        if f['owner'] == player:
            continue
        speed = fleet_speed(f['ships'])
        for check_steps in range(1, 25):
            proj_x = f['x'] + math.cos(f['angle']) * speed * check_steps
            proj_y = f['y'] + math.sin(f['angle']) * speed * check_steps
            for op in other_planets:
                if op['owner'] == f['owner']:
                    opx, opy = predict_pos(op['x'], op['y'], op['radius'], check_steps, av)
                    if dist((proj_x, proj_y), (opx, opy)) < op['radius'] + 1.0:
                        enemy_reinforcements[op['id']] = enemy_reinforcements.get(op['id'], 0) + f['ships']
                        break
            else:
                continue
            break
    
    # --- Calculate defense reserves ---
    defense_reserve = {}
    for mp in my_planets:
        threat = incoming_threats.get(mp['id'], 0)
        if threat > 0:
            # Need to keep enough to survive the attack
            defense_reserve[mp['id']] = threat + 2
        else:
            # Minimum garrison
            defense_reserve[mp['id']] = 2
    
    # --- Track committed targets to avoid duplicate attacks ---
    committed_targets = {}  # target_id -> total ships committed
    
    # Also track our own fleets already in transit toward targets
    for f in fleets:
        if f['owner'] == player:
            speed = fleet_speed(f['ships'])
            for check_steps in range(1, 25):
                proj_x = f['x'] + math.cos(f['angle']) * speed * check_steps
                proj_y = f['y'] + math.sin(f['angle']) * speed * check_steps
                for op in other_planets:
                    opx, opy = predict_pos(op['x'], op['y'], op['radius'], check_steps, av)
                    if dist((proj_x, proj_y), (opx, opy)) < op['radius'] + 1.0:
                        committed_targets[op['id']] = committed_targets.get(op['id'], 0) + f['ships']
                        break
                else:
                    continue
                break
    
    # --- Sort planets: highest ships first for strongest dispatches ---
    my_planets.sort(key=lambda p: p['ships'], reverse=True)
    
    moves = []
    
    for mp in my_planets:
        reserve = defense_reserve.get(mp['id'], 2)
        avail = mp['ships'] - reserve
        if avail <= 0:
            continue
        
        # Can dispatch to multiple targets if enough ships
        max_dispatches = 2
        dispatches = 0
        
        while avail > 5 and dispatches < max_dispatches:
            best_target = None
            best_score = -float('inf')
            best_angle = 0
            best_ships_needed = 0
            
            for tp in other_planets:
                # 1. Calculate intercept
                angle, flight_time, fx, fy = calc_intercept(
                    mp['x'], mp['y'], tp['x'], tp['y'], tp['radius'], avail, av
                )
                
                # 2. Sun avoidance
                if sun_blocked(mp['x'], mp['y'], fx, fy):
                    continue
                
                # 3. Calculate ships needed at arrival
                enemy_prod = tp['production'] if tp['owner'] != -1 else 0
                garrison_at_arrival = tp['ships'] + (enemy_prod * flight_time)
                
                # Account for enemy reinforcements heading there
                reinf = enemy_reinforcements.get(tp['id'], 0)
                garrison_at_arrival += reinf
                
                # Subtract our already-committed ships
                already_committed = committed_targets.get(tp['id'], 0)
                effective_garrison = max(0, garrison_at_arrival - already_committed)
                
                ships_needed = effective_garrison + 1
                
                if avail < ships_needed:
                    continue
                
                # 4. Scoring
                score = (tp['production'] + 0.1) / (flight_time + 1)
                
                # Already fully committed? Skip
                if already_committed > garrison_at_arrival + 5:
                    continue
                
                # Priority multipliers
                if tp['id'] in revenge_ids:
                    score *= 8.0
                if tp['is_comet']:
                    score *= 5.0
                if tp['owner'] == -1:
                    score *= 2.5  # Neutral expansion
                
                # Penalize heavily defended enemy planets
                if tp['owner'] != -1 and tp['ships'] > 50:
                    score *= 0.5
                
                if score > best_score:
                    best_score = score
                    best_target = tp
                    best_angle = angle
                    best_ships_needed = int(ships_needed)
            
            if best_target:
                # Send enough to capture + a buffer, but not wastefully
                send = min(avail, max(best_ships_needed + 3, avail))
                # If first dispatch and we have a lot of ships, consider splitting
                if dispatches == 0 and avail > best_ships_needed * 2 + 10:
                    send = best_ships_needed + 5  # Send just enough with buffer
                
                moves.append([mp['id'], best_angle, int(send)])
                mp['ships'] -= send
                avail -= send
                committed_targets[best_target['id']] = committed_targets.get(best_target['id'], 0) + send
                dispatches += 1
            else:
                break
    
    return moves
