import math

# --- CONSTANTS ---
BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0

# Strategic Constants
MIN_FLEET_SIZE = 4
DEFENSE_BUFFER = 4
PROJECTION_HORIZON = 40
REVENGE_WINDOW = 20

# Persistent state for history and tracking
GLOBAL_STATE = {
    'planet_history': {}, # {pid: (owner, age)}
    'prev_planets': {}, 
    'planet_velocities': {},
    'step': 0,
    'incoming_threats': {} # {pid: total_ships}
}

# --- UTILITY FUNCTIONS ---

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def point_to_segment_distance(p, v, w):
    """Minimum distance from point p to line segment v-w."""
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0: return distance(p, v)
    t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
    projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return distance(p, projection)

def get_fleet_speed(ships):
    """Calculate fleet speed based on ship count."""
    ships = max(1, ships)
    speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(speed, MAX_SPEED)

def predict_planet_position(planet, steps, angular_velocity):
    """Predicts the position of a planet after N steps."""
    is_comet = planet.get('is_comet', False)
    if is_comet:
        # Linear prediction for comets based on tracked velocity
        vel = GLOBAL_STATE['planet_velocities'].get(planet['id'], (0, 0))
        return planet['x'] + vel[0] * steps, planet['y'] + vel[1] * steps
        
    orbital_r = distance((planet['x'], planet['y']), (CENTER, CENTER))
    if orbital_r + planet['radius'] >= ROTATION_RADIUS_LIMIT:
        return planet['x'], planet['y']
    
    current_angle = math.atan2(planet['y'] - CENTER, planet['x'] - CENTER)
    future_angle = current_angle + angular_velocity * steps
    return CENTER + orbital_r * math.cos(future_angle), CENTER + orbital_r * math.sin(future_angle)

def calculate_intercept(source, target, fleet_size, angular_velocity, max_iterations=12):
    """Iteratively calculate intercept angle and flight time."""
    speed = get_fleet_speed(fleet_size)
    steps = distance((source['x'], source['y']), (target['x'], target['y'])) / speed
    
    fx, fy = target['x'], target['y']
    for _ in range(max_iterations):
        fx, fy = predict_planet_position(target, steps, angular_velocity)
        steps = distance((source['x'], source['y']), (fx, fy)) / speed
        
    angle = math.atan2(fy - source['y'], fx - source['x'])
    return angle, steps, fx, fy

def sun_blocked(sx, sy, fx, fy, buffer=2.2):
    """Check if the path from (sx, sy) to (fx, fy) is blocked by the sun."""
    return point_to_segment_distance((CENTER, CENTER), (sx, sy), (fx, fy)) <= SUN_RADIUS + buffer

def find_waypoint(sx, sy, fx, fy):
    """Calculate a detour waypoint to avoid the sun."""
    mx, my = (sx + fx) / 2, (sy + fy) / 2
    dx, dy = fx - sx, fy - sy
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6: return None
    
    # Perpendicular vector
    px, py = -dy / length, dx / length
    offset = SUN_RADIUS + 8.0
    w1 = (mx + px * offset, my + py * offset)
    w2 = (mx - px * offset, my - py * offset)
    
    # Pick the one furthest from the sun to ensure clearance
    wp = w1 if distance(w1, (CENTER, CENTER)) > distance(w2, (CENTER, CENTER)) else w2
    if 2 < wp[0] < 98 and 2 < wp[1] < 98:
        if not sun_blocked(sx, sy, wp[0], wp[1]): return wp
    return None

# --- AGENT LOGIC ---

def agent(obs):
    """
    MASTERMIND V3 (Optimized Advanced Agent)
    Max Speed Intercept & Aggressive Early Snowball
    """
    try:
        player = obs.get("player", 0)
        step = obs.get("step", 0)
        angular_velocity = obs.get("angular_velocity", 0.0)
        comet_ids = set(obs.get("comet_planet_ids", []))
        
        # 1. Update Persistent State
        GLOBAL_STATE['step'] = step
        planets = {p[0]: {
            'id': p[0], 'owner': p[1], 'x': p[2], 'y': p[3], 
            'radius': p[4], 'ships': p[5], 'production': p[6],
            'is_comet': p[0] in comet_ids
        } for p in obs.get("planets", [])}
        
        # Track Revenge
        history = GLOBAL_STATE['planet_history']
        for pid, p in planets.items():
            prev_owner, age = history.get(pid, (-2, 0))
            if prev_owner == player and p['owner'] != player:
                history[pid] = (p['owner'], REVENGE_WINDOW)
            elif age > 0:
                history[pid] = (p['owner'], age - 1)
            else:
                history[pid] = (p['owner'], 0)
                
        # Track comet velocities
        prev_p = GLOBAL_STATE['prev_planets']
        for pid, p in planets.items():
            if pid in prev_p:
                dx, dy = p['x'] - prev_p[pid][0], p['y'] - prev_p[pid][1]
                GLOBAL_STATE['planet_velocities'][pid] = (dx, dy)
            prev_p[pid] = (p['x'], p['y'])
            
        # --- NEW: Fleet Projection (Threat Tracking) ---
        incoming_threats = {pid: 0 for pid in planets}
        committed_friendly = {pid: 0 for pid in planets}
        for f in obs.get("fleets", []):
            f_owner, f_ships, f_angle, f_x, f_y = f[1], f[6], f[4], f[2], f[3]
            f_speed = get_fleet_speed(f_ships)
            # Project flight path
            for t in range(1, PROJECTION_HORIZON):
                fx, fy = f_x + math.cos(f_angle)*f_speed*t, f_y + math.sin(f_angle)*f_speed*t
                hit_id = None
                for pid, p in planets.items():
                    # Quick bounding box check
                    if abs(fx - p['x']) > 20 or abs(fy - p['y']) > 20: continue
                    ppx, ppy = predict_planet_position(p, t, angular_velocity)
                    if (fx-ppx)**2 + (fy-ppy)**2 < (p['radius']+1.5)**2:
                        hit_id = pid
                        break
                if hit_id is not None:
                    if f_owner == player: committed_friendly[hit_id] += f_ships
                    else: incoming_threats[hit_id] += f_ships
                    break
        
        # 2. Strategic Posture Analysis
        p_stats = {i: {'ships': 0, 'prod': 0} for i in range(4)}
        for p in planets.values():
            if p['owner'] >= 0:
                p_stats[p['owner']]['ships'] += p['ships']
                p_stats[p['owner']]['prod'] += p['production']
        
        scores = {i: p_stats[i]['ships'] + p_stats[i]['prod'] * 20 for i in range(4)}
        leader_id = max(scores, key=scores.get)
        posture = "ANTI_LEADER" if (scores[leader_id] - scores[player]) > 50 else "NORMAL"
        
        # 3. Target Evaluation
        targets = []
        for tp in planets.values():
            if tp['owner'] == player: continue
            
            # Base Score: Prod^1.5 favors fast growth
            score = (tp['production'] ** 1.5) * 10
            
            # Aggressive Snowball (Early Game Blitz)
            if tp['owner'] == -1:
                score *= (25.0 if step < 80 else 5.0)
            
            # Special Multipliers
            if tp['is_comet']: score *= 12.0
            if history.get(tp['id'], (-2, 0))[1] > 0: score *= 10.0 # Killer Revenge
            if posture == "ANTI_LEADER" and tp['owner'] == leader_id: score *= 2.0
            
            # Distance / Position Weighting
            orb_r = distance((tp['x'], tp['y']), (CENTER, CENTER))
            if orb_r < 30: score *= 1.5 # Center control
            
            targets.append((tp['id'], score))
            
        targets.sort(key=lambda x: x[1], reverse=True)
        
        # 4. Dispatch Moves
        moves = []
        my_planets = [p for p in planets.values() if p['owner'] == player]
        source_avail = {}
        for p in my_planets:
            # Threat-aware Reserve: keep enough to match incoming threats minus existing garrison
            # We assume current garrison + some production can help.
            reserve = max(DEFENSE_BUFFER, incoming_threats[p['id']] - p['ships'] // 2)
            source_avail[p['id']] = max(0, p['ships'] - int(reserve))
            
        dispatches = {p['id']: 0 for p in my_planets} 
        committed_this_turn = {pid: 0 for pid in planets}

        for tid, t_score in targets:
            tp = planets[tid]
            
            # Find best source and calculate needed ships
            for sid, avail in source_avail.items():
                if avail < MIN_FLEET_SIZE or dispatches[sid] >= 3: continue
                mp = planets[sid]
                
                # Intercept logic
                angle, eta, fx, fy = calculate_intercept(mp, tp, avail, angular_velocity)
                if step + eta >= 498: continue # Game ending
                
                # Sun Collision Avoidance
                if sun_blocked(mp['x'], mp['y'], fx, fy):
                    wp = find_waypoint(mp['x'], mp['y'], fx, fy)
                    if wp: angle = math.atan2(wp[1] - mp['y'], wp[0] - mp['x'])
                    else: continue
                
                # Calculate required ships
                enemy_prod = tp['production'] if tp['owner'] != -1 else 0
                # needed = (Current Ships) + (Production while we travel) + (Incoming enemy fleets) - (Our fleets already on the way)
                needed = tp['ships'] + (enemy_prod * eta) + incoming_threats[tid] - committed_friendly[tid] - committed_this_turn.get(tid, 0)
                needed += (2 if tp['owner'] == -1 else 5) # Safety buffer
                
                if avail >= needed:
                    send = int(needed)
                    moves.append([sid, angle, send])
                    source_avail[sid] -= send
                    committed_this_turn[tid] += send
                    dispatches[sid] += 1
                    break # One target per source-check loop or one source per target? 
                    # Usually we want one source to target if it's enough.

        return moves
    except Exception:
        return []

# Compatibility with advanced_agent definition
def advanced_agent(obs):
    return agent(obs)
