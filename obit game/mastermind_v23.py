# ============================================================
# MASTERMIND v23.0 NEBULA "The Revenant"
# ============================================================
# CORE UPDATES:
#   1. REVENGE SYSTEM: 15x priority for recapturing lost territory.
#   2. ADAPTIVE EARLY-GAME: Aggressive 4-ship expansion in 1:1.
#   3. SOLAR LOCK: 2.2 radius buffer for fleet safety.
#   4. PREDATOR SNIPE 2.0: Dynamic force calculation for snipes.
# ============================================================

import math

BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0

# Global Persistent State
PLANET_HISTORY = {} # {id: (owner, ticks_since_lost)}

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

def sun_blocked(sx, sy, fx, fy, buffer=2.2):
    vx, vy = fx - sx, fy - sy
    l2 = vx*vx + vy*vy
    if l2 == 0: return False
    t = max(0, min(1, ((CENTER-sx)*vx + (CENTER-sy)*vy) / l2))
    dx, dy = CENTER - (sx + t*vx), CENTER - (sy + t*vy)
    return (dx*dx + dy*dy) <= (SUN_RADIUS + buffer)**2

def agent(obs):
    global PLANET_HISTORY
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
        
        # --- REVENGE LOGIC: TRACKING ---
        for pid, p in planets.items():
            prev_owner, ticks = PLANET_HISTORY.get(pid, (p['owner'], 0))
            if prev_owner == player and p['owner'] != player:
                PLANET_HISTORY[pid] = (p['owner'], 10) # Set revenge counter to 10
            elif ticks > 0:
                PLANET_HISTORY[pid] = (p['owner'], ticks - 1)
            else:
                PLANET_HISTORY[pid] = (p['owner'], 0)
        
        # --- ADAPTIVE THRESHOLDS ---
        if is_ffa:
            min_ships = 16 
        else: # 1:1 Speed Expansion
            if step < 60: min_ships = 4
            elif step < 150: min_ships = 12
            else: min_ships = 24
        
        # 1. DATA STRUCTURING
        committed = {p['id']: 0 for p in all_p}
        incoming = {p['id']: [] for p in all_p}
        for f in obs.get("fleets", []):
            f_owner, f_ships, f_angle, f_x, f_y = f[1], f[6], f[4], f[2], f[3]
            if f_owner == player:
                for p in all_p:
                    if dist((f_x, f_y), (p['x'], p['y'])) < 20: 
                        committed[p['id']] += f_ships; break
            f_speed = fleet_speed(f_ships)
            for t in range(5, 100, 5): 
                fx, fy = f_x + math.cos(f_angle)*f_speed*t, f_y + math.sin(f_angle)*f_speed*t
                for p in all_p:
                    if (fx-p['x'])**2 + (fy-p['y'])**2 < (p['radius']+12)**2:
                        incoming[p['id']].append((f_owner, f_ships, t)); break
        
        for p in all_p:
            threat_list = [f for f in incoming[p['id']] if f[0] != player]
            p['min_threat_eta'] = min(f[2] for f in threat_list) if threat_list else 999
            p['threat'] = sum(f[1] for f in threat_list)
            p['is_doomed'] = p['threat'] > (p['ships'] + p['production']*p['min_threat_eta']) and p['min_threat_eta'] < 30
            p['needed_defense'] = (p['threat'] - (p['ships'] + p['production']*p['min_threat_eta']) + 10) if p['is_doomed'] else 0
            
            enemy_dists = [dist((p['x'], p['y']), (ep['x'], ep['y'])) for ep in enemies]
            nearest_enemy_dist = min(enemy_dists) if enemy_dists else 999
            p['role'] = 'Frontline' if nearest_enemy_dist < 55 else 'Backline'
            p['local_threat_level'] = len([d for d in enemy_dists if d < 45])

        my_p.sort(key=lambda p: p['ships'], reverse=True)
        moves = []
        
        for mp in my_p:
            # --- VACUUM & DEFENSE ---
            base_reserve = int(mp['threat'])
            ffa_mult = 1.4 if is_ffa else 1.1
            threat_buffer = mp['local_threat_level'] * 6
            
            reserve = int((base_reserve + threat_buffer + 5) * ffa_mult) if mp['role'] == 'Frontline' else 4
            if mp['is_doomed'] and mp['min_threat_eta'] < 5: reserve = 0 # Evacuate/Fight to the end
            
            avail = mp['ships'] - reserve
            if avail < min_ships: continue 
            
            sub_attempts = 0
            while avail >= min_ships and sub_attempts < 12:
                best_t, best_s, best_send = None, -1.0, 0
                
                for tp in all_p:
                    if tp['id'] == mp['id']: continue
                    dist_val = dist((mp['x'], mp['y']), (tp['x'], tp['y']))
                    
                    if tp['owner'] == player:
                        speed = fleet_speed(avail)
                        eta = dist_val / speed
                        def_needed = tp['needed_defense'] - committed[tp['id']]
                        
                        if tp['is_doomed'] and eta < tp['min_threat_eta'] + 2 and def_needed > 0:
                            score = 600000.0 / (dist_val + 1)
                            send_size = min(avail, int(def_needed))
                        elif mp['role'] == 'Backline' and tp['role'] == 'Frontline':
                            score = 25000.0 / (dist_val + 5)
                            send_size = avail # Springboard
                        else: continue
                    else:
                        angle, eta, fx, fy = calc_intercept(mp['x'], mp['y'], tp['x'], tp['y'], tp['radius'], min(avail, 80), av)
                        if step + eta >= 495 or sun_blocked(mp['x'], mp['y'], fx, fy): continue
                        
                        req = tp['ships'] + (tp['production'] if tp['owner'] != -1 else 0) * eta
                        
                        # --- PREDATOR SNIPE 2.0 ---
                        enemy_incoming = [f for f in incoming[tp['id']] if f[0] not in (player, -1)]
                        is_snipe = False
                        if enemy_incoming:
                            enemy_force = sum(f[1] for f in enemy_incoming)
                            enemy_eta = min(f[2] for f in enemy_incoming)
                            if enemy_force > tp['ships'] and eta > enemy_eta:
                                is_snipe = True
                                req = max(2, enemy_force - tp['ships'])
                        
                        p_mult = 1.2 if (tp['owner'] == -1 and not is_snipe) else 2.5
                        needed = (req + 5) * p_mult - committed[tp['id']]
                        
                        if needed <= 0 or avail < int(needed): continue
                        
                        score = ((tp['production'] + 1.0) * 15000.0) / (dist_val + 15.0)
                        if tp['owner'] == -1 and not is_snipe: score *= 15.0 
                        if tp['is_comet']: score *= 200.0 
                        if is_snipe: score *= 50.0 
                        
                        # --- REVENGE MULTIPLIER ---
                        _, revenge_ticks = PLANET_HISTORY.get(tp['id'], (-1, 0))
                        if revenge_ticks > 0: score *= 15.0
                        
                        if tp['owner'] == -1 and not is_snipe:
                            send_size = int(needed)
                        else:
                            send_size = min(avail, max(int(needed), int(avail // 1.1)))
                    
                    if score > best_s: best_s, best_t, best_send = score, tp, send_size
                
                if best_target := best_t:
                    final_send = int(best_send)
                    if final_send < min_ships: final_send = min_ships
                    if final_send > avail: final_send = int(avail)
                    
                    if final_send >= min_ships:
                        if best_target['owner'] == player: 
                            ang = math.atan2(best_target['y']-mp['y'], best_target['x']-mp['x'])
                        else: 
                            ang, _, _, _ = calc_intercept(mp['x'], mp['y'], best_target['x'], best_target['y'], best_target['radius'], final_send, av)
                        
                        moves.append([mp['id'], ang, final_send])
                        avail -= final_send
                        committed[best_target['id']] += final_send
                    sub_attempts += 1
                else: break
        return moves
    except Exception:
        return []
