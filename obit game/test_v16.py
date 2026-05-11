import sys
import orbit_wars
import mastermind_v16
import mastermind_v13
import test_agent

class ObjDict(dict):
    def __getattr__(self, name):
        if name in self: return self[name]
        raise AttributeError(f"No such attribute: {name}")
    def __setattr__(self, name, value): self[name] = value

class MockEnv:
    def __init__(self, n=2):
        self.configuration = ObjDict({"episodeSteps": 500, "shipSpeed": 6.0, "cometSpeed": 4.0})
        self.done = False

class MockState:
    def __init__(self):
        self.observation = ObjDict({})
        self.action = []
        self.status = "ACTIVE"
        self.reward = 0

def run_one_game(agent_a, agent_b):
    env = MockEnv(2)
    state = [MockState() for _ in range(2)]
    # IMPORTANT: First interpreter call initializes the board
    orbit_wars.interpreter(state, env)
    
    agents = [agent_a, agent_b]
    for step in range(500):
        for i in range(2):
            if state[i].status == "ACTIVE":
                state[i].observation.step = step
                try:
                    state[i].action = agents[i](state[i].observation)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    state[i].action = []
            else:
                state[i].action = []
        state[0].observation.step = step
        orbit_wars.interpreter(state, env)
        if state[0].status == "DONE": break
    
    scores = [0, 0]
    for p in state[0].observation.planets:
        if p[1] != -1: scores[p[1]] += p[5]
    for f in state[0].observation.fleets:
        scores[f[1]] += f[6]
    winner = 0 if scores[0] > scores[1] else (1 if scores[1] > scores[0] else -1)
    return scores, winner

def run_matchup(name_a, agent_a, name_b, agent_b, num_games=30):
    print(f"\n{'='*60}\n  {name_a} vs {name_b} - {num_games} games\n{'='*60}")
    wins_a, wins_b = 0, 0
    for i in range(num_games):
        if i % 2 == 0:
            s, w = run_one_game(agent_a, agent_b)
            if w == 0: wins_a += 1
            elif w == 1: wins_b += 1
        else:
            s, w = run_one_game(agent_b, agent_a)
            if w == 1: wins_a += 1
            elif w == 0: wins_b += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1:>2}/{num_games}] {name_a}: {wins_a}W | {name_b}: {wins_b}W")
    print(f"  FINAL: {name_a} {wins_a}W / {name_b} {wins_b}W")

if __name__ == "__main__":
    run_matchup("LEGEND v16", mastermind_v16.agent, "OVERLORD v13", mastermind_v13.agent, 30)
    run_matchup("LEGEND v16", mastermind_v16.agent, "TEST_BOT", test_agent.test_agent, 20)
