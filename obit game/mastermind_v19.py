import math

# ============================================================
# MASTERMIND v19.32 NEBULA — The Grandmaster (Rank 1 Hunter)
# ============================================================
# KOVI-KILLER STRATEGIES:
#   1. MULTI-FRONT HARASSMENT: Pokes ALL enemy planets with 16-ship fleets.
#   2. PERFECT SYNC STRIKE: Coordinated arrival from multiple hubs on the SAME tick.
#   3. TEMPORAL SPEED SCALING: 16 -> 24 -> 32 for late-game speed edge.
#   4. ROI & VACUUM: Maintained Architect core for stable growth.
# ============================================================

BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

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
    for _ in range(5): 
        fx, fy = predict_pos(tx, ty, tr, steps, av)
        steps = dist((sx, sy), (fx, fy)) / speed
    return math.atan2(fy - sy, fx - sx), steps, fx, fy

def sun_blocked(sx, sy, fx, fy, buffer=1.1):
    vx, vy = fx - sx, fy - sy
    l2 = vx*vx + vy*vy
    if l2 == 0: return False
    t = max(0, min(1, ((CENTER-sx)*vx + (CENTER-sy)*vy) / l2))
    dx, dy = CENTER - (sx + t*vx), CENTER - (sy + t*vy)
    return (dx*dx + dy*dy) <= (SUN_RADIUS + buffer)**2

def agent(obs):
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
        # --- 1:1 & FFA ADAPTIVE SCALING ---
        if is_ffa:
            min_ships = 12 if step < 100 else (24 if step < 300 else 48)
        else: # 1:1 Optimization: Much faster start
            min_ships = 4 if step < 50 else (12 if step < 200 else 24)
        
        # 1. DATA STRUCTURING
        committed = {p['id']: 0 for p in all_p}
        incoming = {p['id']: [] for p in all_p}
        for f in obs.get("fleets", []):
            f_owner, f_ships, f_angle, f_x, f_y = f[1], f[6], f[4], f[2], f[3]
            if f_owner == player:
                for p in all_p:
                    if dist((f_x, f_y), (p['x'], p['y'])) < 15: committed[p['id']] += f_ships; break
            f_speed = fleet_speed(f_ships)
            for t in range(5, 100, 5): 
                fx, fy = f_x + math.cos(f_angle)*f_speed*t, f_y + math.sin(f_angle)*f_speed*t
                for p in all_p:
                    if (fx-p['x'])**2 + (fy-p['y'])**2 < (p['radius']+10)**2:
                        incoming[p['id']].append((f_owner, f_ships, t)); break
        
        for mp in my_p:
            enemy_dists = [dist((mp['x'], mp['y']), (ep['x'], ep['y'])) for ep in enemies]
            nearest_enemy_dist = min(enemy_dists) if enemy_dists else 999
            mp['role'] = 'Frontline' if nearest_enemy_dist < 48 else 'Backline'
            mp['threat'] = sum(f[1] for f in incoming[mp['id']] if f[0] != player)
            mp['local_enemy_count'] = len([d for d in enemy_dists if d < 60])

        my_p.sort(key=lambda p: p['ships'], reverse=True)
        moves, assigned_this_turn = [], set()
        
        for mp in my_p:
            min_eta_list = [f[2] for f in incoming[mp['id']] if f[0] != player]
            min_eta = min(min_eta_list) if min_eta_list else 999
            expected_ships = mp['ships'] + (mp['production'] * min_eta)
            is_doomed = mp['threat'] > expected_ships and min_eta < 20
            
            # --- ADAPTIVE RESERVE ---
            # FFA needs high defense, 1:1 needs map control early
            if is_ffa:
                base_mult = 3.0
                min_res = 10
            else:
                base_mult = 1.2 if step < 100 else 2.5 # Low early reserve for 1:1
                min_res = 4 if step < 100 else 10
                
            base_reserve = int(mp['threat'] + (mp['local_enemy_count'] * 3) + 5)
            reserve = 0 if (is_doomed and min_eta <= 2) else (min_res if mp['role'] == 'Backline' else int(base_reserve * base_mult))
            avail = mp['ships'] - reserve
            if avail < min_ships: continue 
            
            attempts = 0
            while avail >= min_ships and attempts < 10:
                best_t, best_s, best_send = None, -1.0, 0
                
                for tp in all_p:
                    if tp['id'] == mp['id'] or tp['id'] in assigned_this_turn: continue
                    dist_val = dist((mp['x'], mp['y']), (tp['x'], tp['y']))
                    
                    if tp['owner'] == player:
                        if is_doomed: score = 100000.0 / (dist_val + 1); send_size = avail
                        elif mp['role'] == 'Backline' and tp['role'] == 'Frontline': score = 20000.0 / (dist_val + 1); send_size = avail
                        elif tp['ships'] + committed[tp['id']] < 60: score = 10000.0 / (dist_val + 1); send_size = min(avail, 40)
                        else: continue
                    else:
                        angle, eta, fx, fy = calc_intercept(mp['x'], mp['y'], tp['x'], tp['y'], tp['radius'], min(avail, 100), av)
                        if step + eta >= 496 or sun_blocked(mp['x'], mp['y'], fx, fy): continue
                        
                        prod = tp['production'] + 0.1
                        dist_factor = (dist_val + 10.0)
                        
                        # --- ADAPTIVE PERSONNEL ---
                        req = tp['ships'] + (tp['production'] if tp['owner'] != -1 else 0) * eta
                        p_mult = 1.3 if (not is_ffa and tp['owner'] == -1) else 2.0 # 2x for attacks, 1.3x for 1:1 neutrals
                        needed = (req + 5) * p_mult - committed[tp['id']]
                        if needed <= 0: continue
                        
                        # 1:1 Strategy: Starve opponent
                        enemy_dist_val = 999
                        if enemies:
                            enemy_dist_val = min(dist((tp['x'], tp['y']), (ep['x'], ep['y'])) for ep in enemies)
                        starve_bonus = 3.0 if (not is_ffa and tp['owner'] == -1 and enemy_dist_val < 30) else 1.0
                        
                        score = (prod * 1000.0 * starve_bonus) / (dist_factor * (needed + 5.0))
                        
                        if tp['owner'] == -1: score *= 10.0 
                        if tp['is_comet']: score *= 50.0 
                        if not is_ffa and tp['owner'] != -1: score *= 3.0 
                        
                        # --- FASTER FLEETS ---
                        send_size = min(avail, max(int(needed), avail // 1.2)) if step < 350 else avail
                    
                    if score > best_s: best_s, best_t, best_send = score, tp, send_size
                
                if best_target := best_t:
                    final_send = best_send
                    if final_send > avail: final_send = avail
                    if final_send >= min_ships:
                        if best_target['owner'] == player: ang = math.atan2(best_target['y']-mp['y'], best_target['x']-mp['x'])
                        else: ang, _, _, _ = calc_intercept(mp['x'], mp['y'], best_target['x'], best_target['y'], best_target['radius'], final_send, av)
                        moves.append([mp['id'], ang, int(final_send)])
                        avail -= final_send
                        committed[best_target['id']] += final_send
                        assigned_this_turn.add(best_target['id'])
                    attempts += 1
                else: break
        return moves
    except: return []
