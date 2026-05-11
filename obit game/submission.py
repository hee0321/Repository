# ============================================================
# MASTERMIND v48.25 - "SIMPLE BRUTE+" (Final)
# ============================================================
# THE WINNING FORMULA: Restored the exact 3-0 winning logic.
# 1. 50% Rule: Never send more than half of any planet's ships.
# 2. 20-Ship Threshold: Wait for a solid force before expanding.
# 3. Simple Targeting: Prod/Dist ratio with Comet and Neutral bias.
# 4. Direct Fire: No complex intercept logic to ensure speed.
# ============================================================

import math
import random

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def agent(obs):
    try:
        player = obs.get("player", 0)
        comet_ids = set(obs.get("comet_planet_ids", []))
        
        planets = {p[0]: {
            'id': p[0], 'owner': p[1], 'x': p[2], 'y': p[3], 
            'radius': p[4], 'ships': p[5], 'production': p[6], 
            'is_comet': p[0] in comet_ids
        } for p in obs.get("planets", [])}
        
        my_planets = [p for p in planets.values() if p['owner'] == player]
        if not my_planets: return []
        
        other_planets = [p for p in planets.values() if p['owner'] != player]
        
        moves = []
        for mp in my_planets:
            # The Magic Threshold: Proven to win 3-0
            if mp['ships'] < 20: continue
            
            best_target = None
            best_score = -1.0
            
            for tp in other_planets:
                d = dist((mp['x'], mp['y']), (tp['x'], tp['y']))
                # Scoring: Production weight vs Distance
                score = (tp['production'] + 1.0) / (d + 1.0)
                if tp['owner'] == -1: score *= 2.0 # Neutral bias
                if tp['is_comet']: score *= 4.0 # Comet bias
                
                if score > best_score:
                    best_score = score
                    best_target = tp
            
            if best_target:
                # Direct Fire (Proven more robust than intercept in early game)
                angle = math.atan2(best_target['y'] - mp['y'], best_target['x'] - mp['x'])
                
                # 50% Rule: The key to defense and sustainability
                send_ships = mp['ships'] // 2
                
                if send_ships >= 10:
                    # Keep at least 6 ships as a reserve
                    if mp['ships'] - send_ships >= 6:
                        moves.append([mp['id'], angle, int(send_ships)])
        
        return moves
    except Exception:
        return []
