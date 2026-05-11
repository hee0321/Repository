# ============================================================
# MASTERMIND v41.0 - "APEX LEGEND"
# ============================================================
# Refined from v40 analysis:
#   1. CONSOLIDATED DISPATCHES: Limit to 5 per planet to avoid over-extension.
#   2. SAFE FUNNELING: Move ships to front only if surplus > 50, and only 20%.
#   3. AGGRESSIVE OPENING v2: Send 22 ships to top 2 neutrals for high-certainty capture.
#   4. ROBUST SUN-GRAZING: Buffer 2.3 for maximum safety.
#   5. FFA BALANCE: Dynamic threat scaling based on global fleet count.
# ============================================================

import math
import random

BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0

GLOBAL_STATE = {
    'planet_history': {},
    'prev_planets': {}, 
    'planet_velocities': {},
    'swarm_tick': 0
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

def predict_pos(p_id, px, py, pr, steps, av, is_comet):
    global GLOBAL_STATE
    if is_comet:
        vel = GLOBAL_STATE['planet_velocities'].get(p_id, (0, 0))
        return px + vel[0] * steps, py + vel[1] * steps
    orb_r = dist((px, py), (CENTER, CENTER))
    if orb_r + pr >= ROTATION_RADIUS_LIMIT: return px, py
    angle = math.atan2(py - CENTER, px - CENTER) + av * steps
    return CENTER + orb_r * math.cos(angle), CENTER + orb_r * math.sin(angle)

def calc_intercept(sx, sy, tp, fleet_ships, av):
    speed = fleet_speed(fleet_ships)
    prev_steps = dist((sx, sy), (tp['x'], tp['y'])) / speed
    steps = prev_steps
    fx, fy = tp['x'], tp['y']
    for _ in range(15):
        fx, fy = predict_pos(tp['id'], tp['x'], tp['y'], tp['radius'], steps, av, tp['is_comet'])
        steps = dist((sx, sy), (fx, fy)) / speed
        if abs(steps - prev_steps) < 0.02: break
        prev_steps = steps
    return math.atan2(fy - sy, fx - sx), steps, fx, fy

def sun_blocked(sx, sy, fx, fy, buffer=2.3):
    return seg_dist((CENTER, CENTER), (sx, sy), (fx, fy)) <= SUN_RADIUS + buffer

def agent(obs):
    global GLOBAL_STATE
    try:
        player = obs.get("player", 0)
        step = obs.get("step", 0)
        av = obs.get("angular_velocity", 0.0)
        comet_ids = set(obs.get("comet_planet_ids", []))
        
        planets = {p[0]: {'id': p[0], 'owner': p[1], 'x': p[2], 'y': p[3], 'radius': p[4], 'ships': p[5], 'production': p[6], 'is_comet': p[0] in comet_ids} for p in obs.get("planets", [])}
        prev_p = GLOBAL_STATE['prev_planets']
        for pid, p in planets.items():
            if pid in prev_p:
                GLOBAL_STATE['planet_velocities'][pid] = (p['x'] - prev_p[pid][0], p['y'] - prev_p[pid][1])
            prev_p[pid] = (p['x'], p['y'])
            
        all_p = list(planets.values())
        my_p = [p for p in all_p if p['owner'] == player]
        if not my_p: return []
        
        player_strength = {p: {'ships': 0, 'prod': 0} for p in range(4)}
        global_fleets = 0
        for p in all_p:
            if p['owner'] >= 0:
                player_strength[p['owner']]['ships'] += p['ships']
                player_strength[p['owner']]['prod'] += p['production']
        for f in obs.get("fleets", []):
            if f[1] >= 0: player_strength[f[1]]['ships'] += f[6]
            global_fleets += 1
        
        leader_id = player
        max_p = -1
        for pid, stats in player_strength.items():
            pwr = stats['ships'] + stats['prod'] * 25
            if pwr > max_p: max_p = pwr; leader_id = pid
        am_leading = (leader_id == player)

        incoming_fleets = {p['id']: [] for p in all_p}
        committed = {p['id']: 0 for p in all_p}
        for f in obs.get("fleets", []):
            f_owner, f_ships, f_angle, f_x, f_y = f[1], f[6], f[4], f[2], f[3]
            f_speed = fleet_speed(f_ships)
            dest_id = None
            for t in range(1, 45):
                fx, fy = f_x + math.cos(f_angle)*f_speed*t, f_y + math.sin(f_angle)*f_speed*t
                for p in all_p:
                    if abs(fx - p['x']) > 20 or abs(fy - p['y']) > 20: continue 
                    ppx, ppy = predict_pos(p['id'], p['x'], p['y'], p['radius'], t, av, p['is_comet'])
                    if (fx-ppx)**2 + (fy-ppy)**2 < (p['radius']+1.2)**2: dest_id = p['id']; break
                if dest_id is not None:
                    if f_owner == player: committed[dest_id] += f_ships
                    else: incoming_fleets[dest_id].append((f_owner, f_ships, t))
                    break

        def project_ownership(p_id, t_arr):
            p = planets[p_id]
            curr, owner = p['ships'], p['owner']
            events = sorted([(f[2], f[0], f[1]) for f in incoming_fleets[p_id] if f[2] <= t_arr])
            lt = 0
            for t_ev, f_own, f_sh in events:
                if owner != -1: curr += p['production'] * (t_ev - lt)
                if f_own == owner: curr += f_sh
                else:
                    curr -= f_sh
                    if curr < 0: curr, owner = abs(curr), f_own
                lt = t_ev
            if owner != -1: curr += p['production'] * (t_arr - lt)
            return curr, owner

        enemies = [p for p in all_p if p['owner'] != player and p['owner'] != -1]
        frontline_planets = []
        if enemies:
            for mp in my_p:
                min_enemy_dist = min(dist((mp['x'], mp['y']), (ep['x'], ep['y'])) for ep in enemies)
                if min_enemy_dist < 40: frontline_planets.append(mp)
        
        moves = []
        if step == 7:
            mp = my_p[0]
            neutrals = [p for p in all_p if p['owner'] == -1]
            neutrals.sort(key=lambda p: dist((mp['x'], mp['y']), (p['x'], p['y'])))
            for t in neutrals[:2]:
                angle, _, _, _ = calc_intercept(mp['x'], mp['y'], t, 22, av) # Heavier capture
                if not sun_blocked(mp['x'], mp['y'], t['x'], t['y']):
                    moves.append([mp['id'], angle, 22])
            if moves: return moves

        for mp in my_p:
            threats = [f for f in incoming_fleets[mp['id']] if f[1] >= 3]
            min_threat_eta = min(f[2] for f in threats) if threats else 999
            total_threat = sum(f[1] for f in threats)
            p_gar, p_own = project_ownership(mp['id'], min(min_threat_eta, 30))
            
            res = max(4, total_threat - p_gar + 6) if p_own != player else 4
            if am_leading: res += 10 + (global_fleets // 5)
            
            avail = mp['ships'] - res
            if avail <= 4: continue
            
            dispatches = 0
            while avail > 4 and dispatches < 5: # Consolidated dispatches
                best_t, best_s, best_a, best_needed = None, -1.0, 0, 0
                for tp in all_p:
                    if tp['id'] == mp['id']: continue
                    
                    dist_to_t = dist((mp['x'], mp['y']), (tp['x'], tp['y']))
                    if (tp['production'] + 0.1) / (dist_to_t / 4.0 + 1) < 0.003: continue

                    angle, eta, _, _ = calc_intercept(mp['x'], mp['y'], tp, avail, av)
                    if step + eta >= 498: continue
                    if sun_blocked(mp['x'], mp['y'], tp['x'], tp['y']): continue
                    
                    p_gar, p_own = project_ownership(tp['id'], eta)
                    if p_own == player:
                        # Defensive / Supply chain
                        if tp in frontline_planets and mp not in frontline_planets and avail > 50:
                            score = 0.4 / (eta + 1)
                            val_needed = int(avail * 0.2) # Conservative funneling
                        else: continue
                    else:
                        needed = max(0, p_gar - committed.get(tp['id'], 0)) + 1
                        if avail < needed: continue
                        
                        score = (tp['production'] + 0.9) / (needed * eta + 1.0) * 200
                        if tp['owner'] == -1: score *= 7.0
                        if tp['is_comet']: score *= 20.0
                        
                        nearby_enemies = [p for p in enemies if dist((p['x'], p['y']), (tp['x'], tp['y'])) < 25]
                        score *= (1.0 + 0.1 * len(nearby_enemies))
                        
                        val_needed = int(needed)
                    
                    if score > best_s: best_s, best_t, best_a, best_needed = score, tp, angle, val_needed
                
                if best_t:
                    send = min(avail, max(best_needed + 4, avail if best_t['owner'] != -1 else 0))
                    if best_t['owner'] == -1: send = min(avail, max(20, best_needed + 2))
                    
                    send = int(max(4, send))
                    moves.append([mp['id'], best_a, send])
                    avail -= send
                    committed[best_t['id']] += send
                    dispatches += 1
                else:
                    # HARASSMENT
                    GLOBAL_STATE['swarm_tick'] += 1
                    if avail > 10 and GLOBAL_STATE['swarm_tick'] % 4 == 0:
                        targets = [p for p in all_p if p['owner'] != player and p['owner'] != -1]
                        if targets:
                            target = random.choice(targets)
                            angle, _, _, _ = calc_intercept(mp['x'], mp['y'], target, 1, av)
                            if not sun_blocked(mp['x'], mp['y'], target['x'], target['y']):
                                moves.append([mp['id'], angle, 1])
                                avail -= 1
                    break
        return moves
    except Exception: return []
