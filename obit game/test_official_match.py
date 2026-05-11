import orbit_wars
import submission
import math
from types import SimpleNamespace

def reset_agent(agent_func):
    if hasattr(agent_func, 's'): agent_func.s = {}
    if hasattr(agent_func, 'state'): agent_func.state = {}
    if hasattr(agent_func, '_s'): agent_func._s = {}
    try:
        import submission
        submission.GLOBAL_STATE = {
            'planet_history': {}, 'prev_planets': {}, 'planet_velocities': {}, 'step': 0, 'clusters': []
        }
    except: pass

def run_game(agent_a, agent_b):
    reset_agent(agent_a)
    reset_agent(agent_b)
    
    class MockConfig:
        def __init__(self):
            self.episodeSteps = 500
            self.shipSpeed = 6.0
            self.cometSpeed = 4.0
            self.actTimeout = 1.0
            self.runTimeout = 1200
            
    env = SimpleNamespace(configuration=MockConfig(), done=False)
    state = [SimpleNamespace(observation=SimpleNamespace(), status="ACTIVE", action=[], reward=0) for _ in range(2)]
    
    orbit_wars.interpreter(state, env)
    agents = [agent_a, agent_b]
    
    for step in range(500):
        for i in range(2):
            if state[i].status == "ACTIVE":
                obs = state[i].observation
                # The agent expects an object with .get() OR a SimpleNamespace
                # submission.py uses .get(), so we must provide a dict or dict-like object
                obs_dict = {
                    "step": step,
                    "player": i,
                    "planets": [list(p) for p in obs.planets],
                    "fleets": [list(f) for f in obs.fleets],
                    "angular_velocity": obs.angular_velocity,
                    "comet_planet_ids": list(obs.comet_planet_ids) if hasattr(obs, 'comet_planet_ids') else []
                }
                class ObsWrapper:
                    def __init__(self, d): self.__dict__ = d
                    def get(self, k, default=None): return self.__dict__.get(k, default)
                    def __getitem__(self, k): return self.__dict__[k]
                
                try:
                    state[i].action = agents[i](ObsWrapper(obs_dict))
                except Exception:
                    state[i].action = []
            else:
                state[i].action = []
        
        orbit_wars.interpreter(state, env)
        if env.done: break
        
    scores = [0, 0]
    for p in state[0].observation.planets:
        if p[1] != -1: scores[p[1]] += p[5]
    for f in state[0].observation.fleets:
        if f[1] != -1: scores[f[1]] += f[6]
    return scores

if __name__ == "__main__":
    # Test against the BUILT-IN agents in orbit_wars.py
    v48 = submission.agent
    opponents = [
        ("RANDOM", orbit_wars.random_agent),
        ("STARTER", orbit_wars.starter_agent)
    ]
    
    for name, opp in opponents:
        print(f"\n--- SUBMISSION vs {name} ---")
        for i in range(3):
            s = run_game(v48, opp)
            print(f"  [G{i+1}] Score: {s[0]} vs {s[1]}")
