import orbit_wars
import mastermind_v36
import mastermind_v37

class ObjDict(dict):
    def __getattr__(self, name):
        if name in self: return self[name]
        raise AttributeError(f"No such attribute: {name}")
    def __setattr__(self, name, value): self[name] = value

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
                except: state[i].action = []
            else: state[i].action = []
        state[0].observation.step = step
        orbit_wars.interpreter(state, env)
        if state[0].status == "DONE": break
    scores = [0, 0]
    for p in state[0].observation.planets:
        if p[1] != -1: scores[p[1]] += p[5]
    for f in state[0].observation.fleets:
        if f[1] != -1: scores[f[1]] += f[6]
    return 1 if scores[0] > scores[1] else 0

if __name__ == "__main__":
    N = 20
    print(f"=== V37 COSMIC OVERLORD vs V36 PHANTOM HARASS [{N} games] ===")
    w = 0
    for i in range(N):
        mastermind_v36.GLOBAL_STATE = {'planet_history': {}}
        mastermind_v37.GLOBAL_STATE = {'planet_history': {}}
        win = run_simulation(mastermind_v37.agent, mastermind_v36.agent, "V37", "V36")
        w += win
        print(f"Game {i+1}: {'V37' if win else 'V36'} won", flush=True)
    
    print(f"\nFinal Score: V37 wins {w}/{N} ({w/N*100:.1f}%)")
