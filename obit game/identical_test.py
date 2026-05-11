
import killer_agent
import test_agent
import random

def generate_mock_obs():
    obs = {
        "player": 0,
        "angular_velocity": 0.05,
        "planets": [],
        "fleets": [],
        "comet_planet_ids": [10, 11]
    }
    # Add some planets
    for i in range(20):
        owner = -1
        if i == 0: owner = 0
        if i == 1: owner = 1
        obs["planets"].append([
            i, owner, random.uniform(0, 100), random.uniform(0, 100),
            random.uniform(2, 6), random.randint(10, 100), random.randint(1, 5)
        ])
    return obs

print("Comparing KILLER vs TEST on 10 random observations...")
for i in range(10):
    obs = generate_mock_obs()
    # Deep copy obs for each to be super safe
    obs_k = {k: v for k, v in obs.items()}
    obs_t = {k: v for k, v in obs.items()}
    
    moves_k = killer_agent.killer_agent(obs_k)
    moves_t = test_agent.test_agent(obs_t)
    
    if moves_k != moves_t:
        print(f"!!! DISCREPANCY in Trial {i} !!!")
        print(f"Killer: {moves_k}")
        print(f"Test:   {moves_t}")
    else:
        print(f"Trial {i}: Identical")

print("\nChecking source code locations:")
print(f"Killer: {killer_agent.__file__}")
print(f"Test:   {test_agent.__file__}")
