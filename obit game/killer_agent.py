import math

# ============================================================
# KILLER AGENT v10.2 — The Overlord (FINAL)
# Strategy: Hyper-Aggression with Revenge Counter-Play.
# Features:
#  - math.ceil Engine Sync (Perfect Intercepts)
#  - Revenge Logic (Instant recapture of lost planets)
#  - Max Velocity (100% ship dispatch for speed 6.0)
#  - Strategic Neutral Priority (2.5x Neutral, 4.0x Comet)
# ============================================================

BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0

# Persistent state to track ownership changes
LAST_OWNERS = {}

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
    orbital_r = distance((p['x'], p['y']), (CENTER, CENTER))
    if orbital_r + p['radius'] >= ROTATION_RADIUS_LIMIT:
        return p['x'], p['y']
    dx, dy = p['x'] - CENTER, p['y'] - CENTER
    cur_angle = math.atan2(dy, dx)
    future_angle = cur_angle + angular_velocity * steps
    return CENTER + orbital_r * math.cos(future_angle), CENTER + orbital_r * math.sin(future_angle)

def calculate_intercept(source, target, fleet_size, angular_velocity):
    speed = get_fleet_speed(fleet_size)
    steps = math.ceil(distance((source['x'], source['y']), (target['x'], target['y'])) / speed)
    tx, ty = target['x'], target['y']
    for _ in range(15): 
        tx, ty = predict_planet_position(target, steps, angular_velocity)
        steps = math.ceil(distance((source['x'], source['y']), (tx, ty)) / speed)
    return math.atan2(ty - source['y'], tx - source['x']), steps, tx, ty

def killer_agent(obs):
    global LAST_OWNERS
    player = obs.get("player", 0)
    angular_velocity = obs.get("angular_velocity", 0.0)
    comet_ids = set(obs.get("comet_planet_ids", []))
    moves = []

    planets_dict = {}
    for p in obs.get("planets", []):
        pid = p[0]
        planets_dict[pid] = {
            'id': pid, 'owner': p[1], 'x': p[2], 'y': p[3], 'radius': p[4], 
            'ships': p[5], 'production': p[6], 'is_comet': pid in comet_ids
        }
    
    all_p = list(planets_dict.values())
    my_p = [p for p in all_p if p['owner'] == player]
    if not my_p: return []

    # Detect recently lost planets for Revenge Priority
    revenge_targets = []
    for pid, p in planets_dict.items():
        if pid in LAST_OWNERS and LAST_OWNERS[pid] == player and p['owner'] != player:
            revenge_targets.append(pid)
        LAST_OWNERS[pid] = p['owner']

    other_p = [p for p in all_p if p['owner'] != player]
    my_p.sort(key=lambda x: x['ships'], reverse=True)

    for mp in my_p:
        avail = mp['ships'] - 2
        if avail <= 0: continue

        best_t = None
        best_s = -float('inf')
        best_a = 0
        
        for tp in other_p:
            angle, flight_time, fx, fy = calculate_intercept(mp, tp, avail, angular_velocity)
            if point_to_segment_distance((CENTER, CENTER), (mp['x'], mp['y']), (fx, fy)) <= SUN_RADIUS + 2.0:
                continue
            
            enemy_prod = tp['production'] if tp['owner'] != -1 else 0
            ships_needed = tp['ships'] + (enemy_prod * flight_time) + 1
            
            if avail < ships_needed: continue
            
            # SCORING
            score = (tp['production'] + 0.1) / (flight_time + 1)
            
            # Apply Priorities
            if tp['id'] in revenge_targets: score *= 10.0 # REVENGE!
            elif tp['is_comet']: score *= 4.0
            elif tp['owner'] == -1: score *= 2.5
                
            if score > best_s:
                best_s, best_t, best_a = score, tp, angle

        if best_t:
            moves.append([mp['id'], best_a, int(avail)])
            mp['ships'] -= avail

    return moves
