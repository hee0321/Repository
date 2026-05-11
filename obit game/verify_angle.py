import math

def standard_angle(dx, dy):
    return math.atan2(dy, dx)

def inverted_angle(dx, dy):
    return math.atan2(dx, dy)

def simulate_move(angle, speed):
    # Based on orbit_wars.py:
    # x += cos(angle) * speed
    # y += sin(angle) * speed
    vx = math.cos(angle) * speed
    vy = math.sin(angle) * speed
    return vx, vy

if __name__ == "__main__":
    speed = 6.0
    # Goal: Move RIGHT (dx=1, dy=0)
    dx, dy = 1, 0
    
    a_std = standard_angle(dx, dy)
    vx_std, vy_std = simulate_move(a_std, speed)
    print(f"Standard: Angle={a_std:.2f}, VX={vx_std:.2f}, VY={vy_std:.2f}")
    
    a_inv = inverted_angle(dx, dy)
    vx_inv, vy_inv = simulate_move(a_inv, speed)
    print(f"Inverted: Angle={a_inv:.2f}, VX={vx_inv:.2f}, VY={vy_inv:.2f}")
    
    # Goal: Move DOWN (dx=0, dy=1)
    dx, dy = 0, 1
    
    a_std = standard_angle(dx, dy)
    vx_std, vy_std = simulate_move(a_std, speed)
    print(f"Standard (Down): Angle={a_std:.2f}, VX={vx_std:.2f}, VY={vy_std:.2f}")
    
    a_inv = inverted_angle(dx, dy)
    vx_inv, vy_inv = simulate_move(a_inv, speed)
    print(f"Inverted (Down): Angle={a_inv:.2f}, VX={vx_inv:.2f}, VY={vy_inv:.2f}")
