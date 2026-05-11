import orbit_wars
import submission
import mastermind_v12
import mastermind_v13
import mastermind_v14
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
    
    # Official-style setup
    class MockConfig:
        def __init__(self):
            self.episodeSteps = 500
            self.shipSpeed = 6.0
            self.cometSpeed = 4.0
            self.actTimeout = 1.0
            self.runTimeout = 1200
            
    env = SimpleNamespace(configuration=MockConfig(), done=False)
    state = [SimpleNamespace(observation=SimpleNamespace(), status="ACTIVE", action=[], reward=0) for _ in range(2)]
    
    # Init
    orbit_wars.interpreter(state, env)
    agents = [agent_a, agent_b]
    
    for step in range(500):
        for i in range(2):
            if state[i].status == "ACTIVE":
                obs = state[i].observation
                # Build the EXACT dict Kaggle passes
                obs_dict = {
                    "step": step,
                    "player": i,
                    "planets": [list(p) for p in obs.planets],
                    "fleets": [list(f) for f in obs.fleets],
                    "angular_velocity": obs.angular_velocity,
                    "comet_planet_ids": list(obs.comet_planet_ids) if hasattr(obs, 'comet_planet_ids') else []
                }
                try:
                    state[i].action = agents[i](SimpleNamespace(**obs_dict) if not hasattr(agents[i], 'expects_dict') else obs_dict)
                    # Support both SimpleNamespace (v48) and Dict (v12)
                    # Actually, submission.py uses .get(), so it NEEDS a dict or a custom object.
                    # I'll provide a dict-like object.
                    class ObsWrapper:
                        def __init__(self, d): self.__dict__ = d
                        def get(self, k, default=None): return self.__dict__.get(k, default)
                    state[i].action = agents[i](ObsWrapper(obs_dict))
                except Exception:
                    state[i].action = []
            else:
                state[i].action = []
        
        orbit_wars.interpreter(state, env)
        if env.done: break
        
    # Tally final scores from the interpreter's own reward logic or raw planets
    scores = [0, 0]
    for p in state[0].observation.planets:
        if p[1] != -1: scores[p[1]] += p[5]
    for f in state[0].observation.fleets:
        if f[1] != -1: scores[f[1]] += f[6]
    return scores

def run_matchup(name_a, agent_a, name_b, agent_b, num_games=5):
    print(f"\n{'='*60}\n  {name_a} vs {name_b}\n{'='*60}")
    wins_a, wins_b = 0, 0
    for i in range(num_games):
        if i % 2 == 0:
            s = run_game(agent_a, agent_b)
            sa, sb = s[0], s[1]
        else:
            s = run_game(agent_b, agent_a)
            sa, sb = s[1], s[0]
        
        if sa > sb: wins_a += 1
        elif sb > sa: wins_b += 1
        print(f"  [G{i+1}] {name_a}: {sa} | {name_b}: {sb} -> {'A' if sa > sb else 'B'} WIN")
    
    print(f"\nFINAL: {name_a} {wins_a}W - {wins_b}W {name_b}")

if __name__ == "__main__":
    v48 = submission.agent
    opponents = [
        ("v12", mastermind_v12.agent),
        ("v13", mastermind_v13.agent),
        ("v14", mastermind_v14.agent)
    ]
    for name, opp in opponents:
        run_matchup("SUBMISSION (v48.11)", v48, name, opp, 5)
