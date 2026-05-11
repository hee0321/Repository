import math

# ============================================================
# MASTERMIND v12.5 SUPREME — The Tactical Master
# ------------------------------------------------------------
# 1. FUNNELING: Backline planets send ships to frontline.
# 2. SIEGE LOGIC: Multi-planet attacks for faster capture.
# 3. COMET CAPTURE: Balanced 3.0x priority for comet planets.
# 4. REVENGE: 5.0x priority for recently lost planets.
# 5. SUN AVOIDANCE: High-precision ray casting.
# ============================================================

BOARD_SIZE = 100.0
CENTER = (BOARD_SIZE / 2.0, BOARD_SIZE / 2.0)
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def point_to_segment_distance(p, v, w):
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0: return distance(p, v)
    t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
    projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return distance(p, projection)

def get_fleet_speed(ships):
    ships = max(1, ships)
    speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(speed, MAX_SPEED)

def predict_planet_position(p, steps, angular_velocity):
    orbital_r = distance((p['x'], p['y']), CENTER)
    if orbital_r + p['radius'] >= ROTATION_RADIUS_LIMIT:
        return p['x'], p['y']
    dx, dy = p['x'] - CENTER[0], p['y'] - CENTER[1]
    cur_angle = math.atan2(dy, dx)
    future_angle = cur_angle + angular_velocity * steps
    return CENTER[0] + orbital_r * math.cos(future_angle), CENTER[1] + orbital_r * math.sin(future_angle)

def calculate_intercept(source, target, fleet_size, angular_velocity):
    speed = get_fleet_speed(fleet_size)
    steps = distance((source['x'], source['y']), (target['x'], target['y'])) / speed
    tx, ty = target['x'], target['y']
    for _ in range(15): 
        tx, ty = predict_planet_position(target, steps, angular_velocity)
        steps = distance((source['x'], source['y']), (tx, ty)) / speed
    return math.atan2(ty - source['y'], tx - source['x']), steps, tx, ty

def agent(obs):
    if not hasattr(agent, "state"): agent.state = {}
    player = obs.get("player", 0)
    if player not in agent.state or obs.get("step", 0) == 0:
        agent.state[player] = {'last_owners': {}}
    
    p_state = agent.state[player]
    angular_velocity = obs.get("angular_velocity", 0.0)
    comet_ids = set(obs.get("comet_planet_ids", []))
    moves = []

    all_p = []
    for p in obs.get("planets", []):
        all_p.append({
            'id': p[0], 'owner': p[1], 'x': p[2], 'y': p[3], 'radius': p[4], 
            'ships': p[5], 'production': p[6], 'is_comet': p[0] in comet_ids
        })
    
    my_p = [p for p in all_p if p['owner'] == player]
    if not my_p: return []
    other_p = [p for p in all_p if p['owner'] != player]

    # 1. Update Revenge State
    revenge_targets = []
    for p in all_p:
        pid = p['id']
        if pid in p_state['last_owners'] and p_state['last_owners'][pid] == player and p['owner'] != player:
            revenge_targets.append(pid)
        p_state['last_owners'][pid] = p['owner']

    # 2. Classify Planets: Frontline vs Backline
    frontline = []
    backline = []
    for mp in my_p:
        is_front = False
        for op in other_p:
            if distance((mp['x'], mp['y']), (op['x'], op['y'])) < 35.0:
                is_front = True
                break
        if is_front: frontline.append(mp)
        else: backline.append(mp)

    # 3. Funneling: Backline to closest Frontline
    for mp in backline:
        avail = mp['ships'] - 5 # Keep a few for emergency
        if avail < 10: continue # Only funnel significant amounts
        
        target_front = None
        min_dist = float('inf')
        for fp in frontline:
            d = distance((mp['x'], mp['y']), (fp['x'], fp['y']))
            if d < min_dist:
                min_dist, target_front = d, fp
        
        if target_front:
            angle, _, fx, fy = calculate_intercept(mp, target_front, avail, angular_velocity)
            if point_to_segment_distance(CENTER, (mp['x'], mp['y']), (fx, fy)) > SUN_RADIUS + 1.0:
                moves.append([mp['id'], angle, int(avail)])
                mp['ships'] -= avail

    # 4. Siege Logic & Expansion (Frontline only attacks)
    for mp in frontline:
        avail = mp['ships'] - 2
        if avail <= 0: continue

        best_t, best_s, best_a = None, -float('inf'), 0
        
        for tp in other_p:
            angle, flight_time, fx, fy = calculate_intercept(mp, tp, avail, angular_velocity)
            if point_to_segment_distance(CENTER, (mp['x'], mp['y']), (fx, fy)) <= SUN_RADIUS + 2.0:
                continue
            
            enemy_prod = tp['production'] if tp['owner'] != -1 else 0
            needed = tp['ships'] + (enemy_prod * flight_time) + 1
            if avail < needed: continue
            
            score = (tp['production'] + 0.1) / (flight_time + 1)
            if tp['id'] in revenge_targets: score *= 5.0
            if tp['is_comet']: score *= 3.0
            if tp['owner'] == -1: score *= 2.0

            if score > best_s:
                best_s, best_t, best_a = score, tp, angle

        if best_t:
            moves.append([mp['id'], best_a, int(avail)])
            mp['ships'] -= avail

    return moves
