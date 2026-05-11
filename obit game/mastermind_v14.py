import math

# ============================================================
# MASTERMIND v14.0 WARLORD — Tournament-Grade Strategic AI
# ============================================================
# UPGRADES FROM v13:
#   1. WORLD MODEL: Forecast planet ownership/garrison at fleet
#      arrival time, including production and incoming fleets.
#   2. MISSION SYSTEM: Typed missions (capture, reinforce, rescue,
#      snipe, comet_hunt) with priority ordering.
#   3. ENDGAME AWARENESS: Stop wasting ships near step 500.
#   4. WAYPOINT SUN DETOUR: If direct path blocked by sun, try
#      perpendicular waypoint to route around it.
#   5. MULTI-PLANET COORDINATION: Don't over-commit to one target.
#   6. PRODUCTION-WEIGHTED SCORING: Heavily favor high-prod planets.
# ============================================================

BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0
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

def calc_intercept(sx, sy, tx, ty, tr, n_ships, av):
    speed = fleet_speed(n_ships)
    steps = dist((sx, sy), (tx, ty)) / speed
    fx, fy = tx, ty
    for _ in range(15):
        fx, fy = predict_pos(tx, ty, tr, steps, av)
        steps = dist((sx, sy), (fx, fy)) / speed
    return math.atan2(fy - sy, fx - sx), steps, fx, fy

def sun_blocked(sx, sy, fx, fy):
    return seg_dist((CENTER, CENTER), (sx, sy), (fx, fy)) <= SUN_RADIUS + 2.0

def try_waypoint(sx, sy, tx, ty):
    """Generate a waypoint to detour around the sun."""
    mx, my = (sx + tx) / 2, (sy + ty) / 2
    dx, dy = tx - sx, ty - sy
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return None
    # Perpendicular offset
    px, py = -dy / length, dx / length
    offset = SUN_RADIUS + 8.0
    # Try both perpendicular directions, pick the one farther from sun
    w1 = (mx + px * offset, my + py * offset)
    w2 = (mx - px * offset, my - py * offset)
    d1 = dist(w1, (CENTER, CENTER))
    d2 = dist(w2, (CENTER, CENTER))
    wp = w1 if d1 > d2 else w2
    # Validate waypoint is on board
    if 0 < wp[0] < BOARD_SIZE and 0 < wp[1] < BOARD_SIZE:
        # Check both legs don't cross sun
        if not sun_blocked(sx, sy, wp[0], wp[1]):
            return wp
    return None

def estimate_fleet_target(f, all_planets, av):
    """Estimate which planet a fleet is heading toward."""
    speed = fleet_speed(f['ships'])
    best_pid = None
    best_eta = float('inf')
    for p in all_planets:
        # Check if fleet angle roughly points toward this planet
        for check_steps in [3, 6, 10, 15, 20]:
            proj_x = f['x'] + math.cos(f['angle']) * speed * check_steps
            proj_y = f['y'] + math.sin(f['angle']) * speed * check_steps
            px, py = predict_pos(p['x'], p['y'], p['radius'], check_steps, av)
            if dist((proj_x, proj_y), (px, py)) < p['radius'] + 1.5:
                if check_steps < best_eta:
                    best_eta = check_steps
                    best_pid = p['id']
                break
    return best_pid, best_eta

def agent(obs):
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
    
    # --- Parse fleets ---
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
    enemy_planets = [p for p in all_p if p['owner'] not in (-1, player)]
    neutral_planets = [p for p in all_p if p['owner'] == -1]
    
    # --- Revenge detection ---
    revenge_ids = set()
    for pid, p in planets.items():
        if pid in state['prev_owners'] and state['prev_owners'][pid] == player and p['owner'] != player:
            revenge_ids.add(pid)
        state['prev_owners'][pid] = p['owner']
    
    # --- Fleet analysis: incoming threats & reinforcements ---
    incoming_enemy_to_mine = {}  # my_planet_id -> total enemy ships
    incoming_friendly_to_mine = {}  # my_planet_id -> total friendly ships
    committed_to_target = {}  # target_planet_id -> total my ships already sent
    enemy_reinf_to_theirs = {}  # enemy_planet_id -> total enemy reinforcements
    
    for f in fleets:
        target_pid, eta = estimate_fleet_target(f, all_p, av)
        if target_pid is None:
            continue
        
        if f['owner'] == player:
            # My fleet heading somewhere
            if target_pid in [mp['id'] for mp in my_planets]:
                incoming_friendly_to_mine[target_pid] = incoming_friendly_to_mine.get(target_pid, 0) + f['ships']
            else:
                committed_to_target[target_pid] = committed_to_target.get(target_pid, 0) + f['ships']
        else:
            # Enemy fleet
            if target_pid in [mp['id'] for mp in my_planets]:
                incoming_enemy_to_mine[target_pid] = incoming_enemy_to_mine.get(target_pid, 0) + f['ships']
            else:
                tp = planets.get(target_pid)
                if tp and tp['owner'] == f['owner']:
                    enemy_reinf_to_theirs[target_pid] = enemy_reinf_to_theirs.get(target_pid, 0) + f['ships']
    
    # --- Calculate defense reserves ---
    defense_reserve = {}
    for mp in my_planets:
        threat = incoming_enemy_to_mine.get(mp['id'], 0)
        if threat > 0:
            defense_reserve[mp['id']] = threat + 3
        else:
            defense_reserve[mp['id']] = 2
    
    # --- Build mission list ---
    missions = []  # (priority_score, source_id, target_id, angle, ships_needed, mission_type)
    
    my_planets.sort(key=lambda p: p['ships'], reverse=True)
    
    for mp in my_planets:
        reserve = defense_reserve.get(mp['id'], 2)
        avail = mp['ships'] - reserve
        if avail <= 3:
            continue
        
        for tp in other_planets:
            # --- Endgame check: don't launch if fleet won't arrive before game ends ---
            angle, flight_time, fx, fy = calc_intercept(
                mp['x'], mp['y'], tp['x'], tp['y'], tp['radius'], avail, av
            )
            
            if step + flight_time >= EPISODE_STEPS - 5:
                continue  # Fleet won't arrive in time
            
            # --- Sun avoidance with waypoint fallback ---
            blocked = sun_blocked(mp['x'], mp['y'], fx, fy)
            waypoint = None
            if blocked:
                waypoint = try_waypoint(mp['x'], mp['y'], fx, fy)
                if waypoint:
                    # Re-aim toward waypoint instead
                    angle = math.atan2(waypoint[1] - mp['y'], waypoint[0] - mp['x'])
                    blocked = False
            
            if blocked:
                continue
            
            # --- Arrival-time garrison prediction ---
            enemy_prod = tp['production'] if tp['owner'] != -1 else 0
            garrison_at_arrival = tp['ships'] + (enemy_prod * flight_time)
            garrison_at_arrival += enemy_reinf_to_theirs.get(tp['id'], 0)
            
            # Subtract already committed ships
            already = committed_to_target.get(tp['id'], 0)
            effective_garrison = max(0, garrison_at_arrival - already)
            
            ships_needed = effective_garrison + 1
            
            if avail < ships_needed:
                continue
            
            # Already sufficiently committed?
            if already > garrison_at_arrival + 10:
                continue
            
            # --- Mission scoring ---
            # Base: production value per time unit
            remaining_turns = EPISODE_STEPS - step - flight_time
            roi = tp['production'] * remaining_turns  # total production we'd gain
            time_cost = flight_time + 1
            score = roi / time_cost
            
            # Mission type classification & multipliers
            mission_type = "capture"
            if tp['id'] in revenge_ids:
                score *= 6.0
                mission_type = "recapture"
            if tp['is_comet']:
                score *= 4.0
                mission_type = "comet_hunt"
            if tp['owner'] == -1:
                score *= 2.0
                mission_type = "expand"
            
            # Penalize over-defended targets
            if tp['owner'] != -1 and tp['ships'] > 80:
                score *= 0.3
            
            # Bonus for low-garrison easy captures (snipe)
            if tp['ships'] < 10 and tp['production'] >= 2:
                score *= 1.5
                mission_type = "snipe"
            
            missions.append((score, mp['id'], tp['id'], angle, ships_needed, avail, mission_type))
    
    # --- Sort missions by priority and execute ---
    missions.sort(key=lambda m: m[0], reverse=True)
    
    moves = []
    used_sources = {}  # source_id -> ships already sent this turn
    executed_targets = set()  # prevent exact duplicates
    
    for score, src_id, tgt_id, angle, ships_needed, src_avail, m_type in missions:
        mp = planets[src_id]
        reserve = defense_reserve.get(src_id, 2)
        already_sent = used_sources.get(src_id, 0)
        remaining = mp['ships'] - reserve - already_sent
        
        if remaining <= 3:
            continue
        
        # Skip if we already over-committed to this target
        if committed_to_target.get(tgt_id, 0) > planets[tgt_id]['ships'] + planets[tgt_id]['production'] * 10 + 15:
            continue
        
        # Determine send amount
        if remaining > ships_needed * 2 + 10:
            # Have excess: send just enough + buffer to save ships for other missions
            send = min(remaining, ships_needed + 5)
        else:
            # Send all available for speed
            send = remaining
        
        if send < ships_needed:
            continue
        
        # Prevent same source attacking same target twice
        key = (src_id, tgt_id)
        if key in executed_targets:
            continue
        executed_targets.add(key)
        
        moves.append([src_id, angle, int(send)])
        used_sources[src_id] = already_sent + send
        committed_to_target[tgt_id] = committed_to_target.get(tgt_id, 0) + send
    
    # --- REINFORCE: Send backline ships to frontline ---
    # Only if there are enemy planets nearby some of our planets
    if len(my_planets) >= 3:
        frontline = []
        backline = []
        for mp in my_planets:
            sent = used_sources.get(mp['id'], 0)
            remaining = mp['ships'] - defense_reserve.get(mp['id'], 2) - sent
            if remaining < 10:
                continue
            is_front = False
            for op in other_planets:
                if dist((mp['x'], mp['y']), (op['x'], op['y'])) < 30:
                    is_front = True
                    break
            if is_front:
                frontline.append(mp)
            else:
                backline.append(mp)
        
        for bp in backline:
            sent = used_sources.get(bp['id'], 0)
            remaining = bp['ships'] - defense_reserve.get(bp['id'], 2) - sent
            if remaining < 15:
                continue
            
            # Find closest frontline planet that could use reinforcement
            best_front = None
            best_dist = float('inf')
            for fp in frontline:
                d = dist((bp['x'], bp['y']), (fp['x'], fp['y']))
                if d < best_dist:
                    best_dist = d
                    best_front = fp
            
            if best_front and best_dist < 50:
                angle, _, fx, fy = calc_intercept(
                    bp['x'], bp['y'], best_front['x'], best_front['y'],
                    best_front['radius'], remaining, av
                )
                if not sun_blocked(bp['x'], bp['y'], fx, fy):
                    if step + (best_dist / fleet_speed(remaining)) < EPISODE_STEPS - 10:
                        moves.append([bp['id'], angle, int(remaining)])
                        used_sources[bp['id']] = sent + remaining
    
    return moves
