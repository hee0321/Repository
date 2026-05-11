# ============================================================
# MASTERMIND v24.0 NEBULA "Hyper-Singularity"
# ============================================================
# HYBRID EVOLUTION:
#   1. WAYPOINT DETOUR: Surgical paths around the sun.
#   2. PRECISION TRACKER: 40-step projection for threat detection.
#   3. REVENGE INSTINCT: 20x priority for recapturing territory.
#   4. BALANCED EXPANSION: Low threshold (6 ships) for early growth.
# ============================================================

import math

BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0

# Persistent state across steps
GLOBAL_STATE = {
    'planet_history': {}, # id: (owner, revenge_ticks)
    'player_id': -1
}

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
    for _ in range(10): 
        fx, fy = predict_pos(tx, ty, tr, steps, av)
        steps = dist((sx, sy), (fx, fy)) / speed
    return math.atan2(fy - sy, fx - sx), steps, fx, fy

def sun_blocked(sx, sy, fx, fy, buffer=2.2):
    return seg_dist((CENTER, CENTER), (sx, sy), (fx, fy)) <= SUN_RADIUS + buffer

def try_waypoint(sx, sy, fx, fy, buffer=2.2):
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
        
        enemies = [p for p in all_p if p['owner'] not in [-1, player]]
        enemy_owners = set(p['owner'] for p in enemies)
        is_ffa = len(enemy_owners) > 1
        
        # --- PERSISTENT REVENGE TRACKING ---
        history = GLOBAL_STATE['planet_history']
        for pid, p in planets.items():
            prev_owner, ticks = history.get(pid, (p['owner'], 0))
            if prev_owner == player and p['owner'] != player:
                history[pid] = (p['owner'], 15) # Strong revenge
            elif ticks > 0:
                history[pid] = (p['owner'], ticks - 1)
            else:
                history[pid] = (p['owner'], 0)
        
        # --- PRECISION THREAT PROJECTION ---
        committed = {p['id']: 0 for p in all_p}
        incoming_ships = {p['id']: [] for p in all_p} # (owner, ships, eta)
        
        for f in obs.get("fleets", []):
            f_owner, f_ships, f_angle, f_x, f_y = f[1], f[6], f[4], f[2], f[3]
            f_speed = fleet_speed(f_ships)
            # 40-step simulation
            for t in range(1, 41):
                fx = f_x + math.cos(f_angle) * f_speed * t
                fy = f_y + math.sin(f_angle) * f_speed * t
                hit_p = None
                for p in all_p:
                    ppx, ppy = predict_pos(p['x'], p['y'], p['radius'], t, av)
                    if (fx-ppx)**2 + (fy-ppy)**2 < (p['radius'] + 1.5)**2:
                        hit_p = p; break
                if hit_p:
                    if f_owner == player: committed[hit_p['id']] += f_ships
                    else: incoming_ships[hit_p['id']].append((f_owner, f_ships, t))
                    break
        
        # --- DEFENSE CALCULATIONS ---
        for p in all_p:
            threats = incoming_ships[p['id']]
            p['min_threat_eta'] = min(f[2] for f in threats) if threats else 999
            p['total_threat'] = sum(f[1] for f in threats)
            # Simple simulation of survival
            expected = p['ships'] + (p['production'] * min(p['min_threat_eta'], 30))
            p['is_doomed'] = p['total_threat'] > (expected + 5) and p['min_threat_eta'] < 25
            p['defense_needed'] = max(0, p['total_threat'] - expected + 10) if p['is_doomed'] else 0
            
            # Roles
            enemy_dists = [dist((p['x'], p['y']), (ep['x'], ep['y'])) for ep in enemies]
            near_enemy = min(enemy_dists) if enemy_dists else 999
            p['role'] = 'Frontline' if near_enemy < 55 else 'Backline'
            p['local_danger'] = len([d for d in enemy_dists if d < 45])

        # --- MOVE GENERATION ---
        my_p.sort(key=lambda p: p['ships'], reverse=True)
        moves = []
        
        min_ships = 6 if step < 100 else (16 if is_ffa else 12)
        
        for mp in my_p:
            # Reserve logic
            base_res = int(mp['total_threat'] + mp['local_danger'] * 5 + 5)
            if mp['role'] == 'Backline': base_res = 4
            if mp['is_doomed'] and mp['min_threat_eta'] < 6: base_res = 0 # Evacuate
            
            avail = mp['ships'] - base_res
            if avail < min_ships: continue
            
            sub_steps = 0
            while avail >= min_ships and sub_steps < 10:
                best_t, best_s, best_a, best_size = None, -1.0, 0, 0
                
                for tp in all_p:
                    if tp['id'] == mp['id']: continue
                    dist_val = dist((mp['x'], mp['y']), (tp['x'], tp['y']))
                    
                    # Intercept
                    angle, eta, fx, fy = calc_intercept(mp['x'], mp['y'], tp['x'], tp['y'], tp['radius'], avail, av)
                    if step + eta >= 496: continue
                    
                    is_waypoint = False
                    if sun_blocked(mp['x'], mp['y'], fx, fy):
                        wp = try_waypoint(mp['x'], mp['y'], fx, fy)
                        if wp:
                            angle = math.atan2(wp[1] - mp['y'], wp[0] - mp['x'])
                            is_waypoint = True
                        else: continue
                    
                    if tp['owner'] == player:
                        # Rescue or Logistics
                        if tp['is_doomed'] and eta < tp['min_threat_eta'] + 3:
                            score = 800000.0 / (dist_val + 1)
                            size = min(avail, int(tp['defense_needed'] - committed[tp['id']]))
                        elif mp['role'] == 'Backline' and tp['role'] == 'Frontline':
                            score = 15000.0 / (dist_val + 1)
                            size = avail # Full dump
                        else: continue
                    else:
                        # Combat or Capture
                        req = tp['ships'] + (tp['production'] if tp['owner'] != -1 else 0) * eta
                        
                        # Predator Snipe check
                        enemy_inc = [f for f in incoming_ships[tp['id']] if f[0] != player]
                        is_snipe = False
                        if enemy_inc:
                            e_force = sum(f[1] for f in enemy_inc)
                            e_eta = min(f[2] for f in enemy_inc)
                            if e_force > tp['ships'] and eta > e_eta:
                                is_snipe = True
                                req = max(2, e_force - tp['ships'])
                        
                        p_mult = 1.3 if (tp['owner'] == -1 and not is_snipe) else 2.5
                        needed = (req + 6) * p_mult - committed[tp['id']]
                        
                        if avail < needed or needed <= 0: continue
                        
                        score = ((tp['production'] + 0.5) * 20000.0) / (dist_val + 20.0)
                        if tp['owner'] == -1: score *= 18.0
                        if tp['is_comet']: score *= 250.0
                        if is_snipe: score *= 60.0
                        if is_waypoint: score *= 0.7 # Penalty for indirect path
                        
                        # Revenge
                        _, r_ticks = history.get(tp['id'], (-1, 0))
                        if r_ticks > 0: score *= 20.0
                        
                        if tp['owner'] == -1 and not is_snipe: size = int(needed)
                        else: size = min(avail, max(int(needed), int(avail // 1.1)))
                    
                    if score > best_s:
                        best_s, best_t, best_a, best_size = score, tp, angle, size
                
                if best_t:
                    final_size = int(max(min_ships, best_size))
                    if final_size > avail: final_size = int(avail)
                    moves.append([mp['id'], best_a, final_size])
                    avail -= final_size
                    committed[best_t['id']] += final_size
                    sub_steps += 1
                else: break
        return moves
    except Exception: return []
