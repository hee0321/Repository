import orbit_wars
import mastermind_v47
import mastermind_v48
import math

class ObjDict(dict):
    def __getattr__(self, name):
        if name in self: return self[name]
        raise AttributeError(f"No such attribute: {name}")
    def __setattr__(self, name, value):
        self[name] = value

class MockEnv:
    def __init__(self):
        self.configuration = ObjDict({"episodeSteps": 500, "shipSpeed": 6.0, "cometSpeed": 4.0})
        self.done = False

class MockState:
    def __init__(self):
        self.observation = ObjDict({})
        self.action = []
        self.status = "ACTIVE"
        self.reward = 0

def run_simulation(agent_a, agent_b):
    env = MockEnv()
    state = [MockState() for _ in range(2)]
    orbit_wars.interpreter(state, env)
    agents = [agent_a, agent_b]
    for step in range(500):
        for i in range(2):
            if state[i].status == "ACTIVE":
                state[i].observation.step = step
                try:
                    state[i].action = agents[i](state[i].observation)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    state[i].action = []
            else: state[i].action = []
        orbit_wars.interpreter(state, env)
        if state[0].status == "DONE": break
    scores = [0, 0]
    for p in state[0].observation.planets:
        if p[1] >= 0: scores[p[1]] += p[5]
    for f in state[0].observation.fleets:
        if f[1] >= 0: scores[f[1]] += f[6]
    return 1 if scores[0] > scores[1] else 0

if __name__ == "__main__":
    N = 10
    print(f"=== V48 (Ghost Path) vs V47 (Dominator) [{N} games] ===")
    wins = 0
    for i in range(N):
        mastermind_v47.GLOBAL_STATE = {'planet_history': {}, 'prev_planets': {}, 'planet_velocities': {}, 'step': 0, 'clusters': []}
        mastermind_v48.GLOBAL_STATE = {'planet_history': {}, 'prev_planets': {}, 'planet_velocities': {}, 'step': 0, 'clusters': []}
        res = run_simulation(mastermind_v48.agent, mastermind_v47.agent)
        wins += res
        print(f"Game {i+1}: {'V48' if res else 'V47'} WON")
    
    print(f"\nFINAL RESULT: V48 wins {wins}/{N} ({wins/N*100:.1f}%)")
