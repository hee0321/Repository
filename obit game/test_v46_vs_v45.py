import orbit_wars
import mastermind_v45
import mastermind_v46
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

def run_simulation(agent_a, agent_b, name_a, name_b):
    env = MockEnv()
    state = [MockState() for _ in range(2)]
    orbit_wars.interpreter(state, env)
    agents = [agent_a, agent_b]
    for step in range(500):
        for i in range(2):
            if state[i].status == "ACTIVE":
                state[i].observation.step = step
                try: state[i].action = agents[i](state[i].observation)
                except Exception as e: 
                    # print(f"Error in {i}: {e}")
                    state[i].action = []
            else: state[i].action = []
        orbit_wars.interpreter(state, env)
        if state[0].status == "DONE": break
    scores = [0, 0]
    for p in state[0].observation.planets:
        if p[1] != -1: scores[p[1]] += p[5]
    for f in state[0].observation.fleets:
        if f[1] != -1: scores[f[1]] += f[6]
    return 1 if scores[0] > scores[1] else 0

if __name__ == "__main__":
    N = 10
    print(f"=== V46 vs V45 [{N} games] ===")
    wins = 0
    for i in range(N):
        mastermind_v45.GLOBAL_STATE = {'planet_history': {}, 'prev_planets': {}, 'planet_velocities': {}, 'swarm_tick': 0}
        mastermind_v46.GLOBAL_STATE = {'planet_history': {}, 'prev_planets': {}, 'planet_velocities': {}, 'swarm_tick': 0, 'adversaries': {0: {}, 1: {}, 2: {}, 3: {}}}
        res = run_simulation(mastermind_v46.agent, mastermind_v45.agent, "V46", "V45")
        wins += res
        print(f"Game {i+1}: {'V46' if res else 'V45'} WON")
    
    print(f"\nFINAL RESULT: V46 wins {wins}/{N} ({wins/N*100:.1f}%)")
