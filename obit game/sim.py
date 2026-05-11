import orbit_wars
import mastermind_v18
import mastermind_v30
import mastermind_v31

class ObjDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"No such attribute: {name}")
    def __setattr__(self, name, value):
        self[name] = value

class MockEnv:
    def __init__(self):
        self.configuration = ObjDict({
            "episodeSteps": 500,
            "shipSpeed": 6.0,
            "cometSpeed": 4.0
        })
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
    planets = [0, 0]
    for p in state[0].observation.planets:
        if p[1] != -1:
            scores[p[1]] += p[5]
            planets[p[1]] += 1
    for f in state[0].observation.fleets:
        scores[f[1]] += f[6]
    winner = "A" if scores[0] > scores[1] else "B" if scores[1] > scores[0] else "DRAW"
    print(f"  {name_a}: {scores[0]} ({planets[0]}p) | {name_b}: {scores[1]} ({planets[1]}p) -> {winner} WON")
    return 1 if scores[0] > scores[1] else 0

if __name__ == "__main__":
    N = 30

    print(f"=== V31 APEX vs V30 NEBULA [{N} games] ===")
    w1 = 0
    for i in range(N):
        # Reset global state between games
        mastermind_v31.GLOBAL_STATE = {'planet_history': {}, 'game_mode': 0}
        mastermind_v30.GLOBAL_STATE = {'planet_history': {}}
        w1 += run_simulation(mastermind_v31.agent, mastermind_v30.agent, "V31", "V30")
    print(f"V31 wins: {w1}/{N}\n")

    print(f"=== V31 APEX vs V18 OVERSEER [{N} games] ===")
    w2 = 0
    for i in range(N):
        mastermind_v31.GLOBAL_STATE = {'planet_history': {}, 'game_mode': 0}
        w2 += run_simulation(mastermind_v31.agent, mastermind_v18.agent, "V31", "V18")
    print(f"V31 wins: {w2}/{N}\n")

    print("=" * 50)
    print(f"SUMMARY: V31 vs V30: {w1}/{N} | V31 vs V18: {w2}/{N}")
    if w1 >= 6 and w2 >= 6:
        print("V31 APPROVED - Ready for submission!")
    else:
        print("V31 needs improvement - Review parameters")
