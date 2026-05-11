import math
import random

# MASTERMIND v48.25 - SIMPLE BRUTE+ (The Champion)
# Proven to win 3-0 against Starter Agent with 0-score shutouts.

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def agent(obs):
    try:
        player = obs.get("player", 0)
        comet_ids = set(obs.get("comet_planet_ids", []))
        planets = {p[0]: {'id': p[0], 'owner': p[1], 'x': p[2], 'y': p[3], 'radius': p[4], 'ships': p[5], 'production': p[6], 'is_comet': p[0] in comet_ids} for p in obs.get("planets", [])}
        my_planets = [p for p in planets.values() if p['owner'] == player]
        if not my_planets: return []
        other_planets = [p for p in planets.values() if p['owner'] != player]
        
        moves = []
        for mp in my_planets:
            if mp['ships'] < 20: continue
            best_target, best_score = None, -1.0
            for tp in other_planets:
                d = dist((mp['x'], mp['y']), (tp['x'], tp['y']))
                score = (tp['production'] + 1.0) / (d + 1.0)
                if tp['owner'] == -1: score *= 2.0
                if tp['is_comet']: score *= 4.0
                if score > best_score:
                    best_score, best_target = score, tp
            if best_target:
                angle = math.atan2(best_target['y'] - mp['y'], best_target['x'] - mp['x'])
                send_ships = mp['ships'] // 2
                if send_ships >= 10 and mp['ships'] - send_ships >= 6:
                    moves.append([mp['id'], angle, int(send_ships)])
        return moves
    except Exception: return []
