import orbit_wars
import mastermind_v49
import mastermind_v47
import mastermind_v30

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
                state[i].observation.player = i
                try: state[i].action = agents[i](state[i].observation)
                except Exception as e: 
                    # print(f"Error in {i}: {e}")
                    state[i].action = []
            else: state[i].action = []
        orbit_wars.interpreter(state, env)
        if state[0].status == "DONE": break
    
    scores = [0, 0]
    for p in state[0].observation.planets:
        if p[1] != -1: scores[p[1]] += p[5] + p[6]*10
    for f in state[0].observation.fleets:
        scores[f[1]] += f[6]
    
    return 1 if scores[0] > scores[1] else 0

if __name__ == "__main__":
    N = 20
    print(f"=== BENCHMARKING V3 (v49) [{N} games each] ===")

    # Test vs V30 (Nebula)
    w_v30 = 0
    for i in range(N):
        mastermind_v49.GLOBAL_STATE = {'planet_history': {}, 'prev_planets': {}, 'planet_velocities': {}, 'step': 0, 'clusters': []}
        mastermind_v30.GLOBAL_STATE = {'planet_history': {}}
        win = run_simulation(mastermind_v49.agent, mastermind_v30.agent, "V49", "V30")
        w_v30 += win
        print(f"Game {i+1}: {'V49 won' if win else 'V30 won'}")
    print(f"V49 vs V30 Score: {w_v30}/{N}\n")

    # Test vs V47 (Dominance)
    w_v47 = 0
    for i in range(N):
        mastermind_v49.GLOBAL_STATE = {'planet_history': {}, 'prev_planets': {}, 'planet_velocities': {}, 'step': 0, 'clusters': []}
        mastermind_v47.GLOBAL_STATE = {'planet_history': {}, 'prev_planets': {}, 'planet_velocities': {}, 'step': 0, 'clusters': []}
        win = run_simulation(mastermind_v49.agent, mastermind_v47.agent, "V49", "V47")
        w_v47 += win
        print(f"Game {i+1}: {'V49 won' if win else 'V47 won'}")
    print(f"V49 vs V47 Score: {w_v47}/{N}\n")

    print("=" * 50)
    print(f"FINAL SUMMARY: V49 vs V30: {w_v30}/{N} | V49 vs V47: {w_v47}/{N}")
