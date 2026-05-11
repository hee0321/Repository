import math

# Constants from the environment
BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def point_to_segment_distance(p, v, w):
    """Minimum distance from point p to line segment v-w."""
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0:
        return distance(p, v)
    t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
    projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return distance(p, projection)

def get_fleet_speed(ships):
    """Calculate fleet speed based on ship count."""
    ships = max(1, ships)
    speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(speed, MAX_SPEED)

def predict_planet_position(planet, steps, angular_velocity):
    """Predicts the position of a planet after a given number of steps."""
    orbital_r = distance((planet['x'], planet['y']), (CENTER, CENTER))
    # Check if planet is orbiting or static
    if orbital_r + planet['radius'] >= ROTATION_RADIUS_LIMIT:
        return planet['x'], planet['y']
    
    # Calculate future position for orbiting planets
    dx = planet['x'] - CENTER
    dy = planet['y'] - CENTER
    current_angle = math.atan2(dy, dx)
    future_angle = current_angle + angular_velocity * steps
    
    future_x = CENTER + orbital_r * math.cos(future_angle)
    future_y = CENTER + orbital_r * math.sin(future_angle)
    return future_x, future_y

def calculate_intercept(source, target, fleet_size, angular_velocity, max_iterations=10):
    """Iteratively calculate the intercept angle and distance to a moving target."""
    speed = get_fleet_speed(fleet_size)
    estimated_distance = distance((source['x'], source['y']), (target['x'], target['y']))
    estimated_steps = estimated_distance / speed
    
    target_x, target_y = target['x'], target['y']
    
    for _ in range(max_iterations):
        target_x, target_y = predict_planet_position(target, estimated_steps, angular_velocity)
        new_distance = distance((source['x'], source['y']), (target_x, target_y))
        estimated_steps = new_distance / speed
        
    angle = math.atan2(target_y - source['y'], target_x - source['x'])
    return angle, estimated_steps, target_x, target_y

def advanced_agent(obs):
    """
    Advanced Agent for Orbit Wars.
    Includes: Target value calculation, moving planet intercept, and sun collision avoidance.
    """
    moves = []
    player = obs.get("player", 0)
    angular_velocity = obs.get("angular_velocity", 0.0)
    
    # Parse planets into dictionaries for easier access
    planets = []
    for p in obs.get("planets", []):
        planets.append({
            'id': p[0], 'owner': p[1], 'x': p[2], 'y': p[3], 
            'radius': p[4], 'ships': p[5], 'production': p[6]
        })
        
    my_planets = [p for p in planets if p['owner'] == player]
    other_planets = [p for p in planets if p['owner'] != player]
    
    for mp in my_planets:
        # Keep a small reserve of ships for defense
        available_ships = mp['ships'] - 5 
        
        if available_ships <= 0:
            continue
            
        best_target = None
        best_score = -float('inf')
        best_angle = 0
        ships_to_send = 0
        
        for tp in other_planets:
            # 1. Calculate Intercept and Flight Time
            angle, flight_time, future_x, future_y = calculate_intercept(mp, tp, available_ships, angular_velocity)
            
            # 2. Sun Collision Avoidance
            # Check if the line segment from source to future target intersects the sun
            sun_dist = point_to_segment_distance((CENTER, CENTER), (mp['x'], mp['y']), (future_x, future_y))
            if sun_dist <= SUN_RADIUS + 1.0: # add a small buffer
                continue # Skip this target because the path goes through the sun
                
            # 3. Calculate Required Ships to Capture
            # If owned by an opponent, they will produce ships while our fleet travels
            enemy_production = tp['production'] if tp['owner'] != -1 else 0
            ships_needed = tp['ships'] + (enemy_production * flight_time) + 1
            
            if available_ships < ships_needed:
                continue # We don't have enough ships to guarantee capture in one go
                
            # 4. Target Evaluation Score
            # We want high production, low ships needed, and short distance
            score = tp['production'] / (ships_needed + flight_time + 1)
            
            if score > best_score:
                best_score = score
                best_target = tp
                best_angle = angle
                # Send just enough ships to capture, maybe slightly more to be safe
                ships_to_send = min(available_ships, int(ships_needed) + 3)
                
        if best_target and ships_to_send > 0:
            moves.append([mp['id'], best_angle, ships_to_send])
            mp['ships'] -= ships_to_send # Update local count so we can send more fleets if we have ships left
            
    return moves
