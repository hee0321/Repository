"""
100게임 대규모 대결: killer(v5) vs test_agent + killer vs farmer_agent
"""
import sys
import random
import orbit_wars
import killer_agent
import test_agent
import farmer_agent

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

def run_one_game(agent_a, agent_b):
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
                except:
                    state[i].action = []
            else:
                state[i].action = []
        state[0].observation.step = step
        orbit_wars.interpreter(state, env)
        if state[0].status == "DONE":
            break
    scores = [0, 0]
    for p in state[0].observation.planets:
        if p[1] != -1:
            scores[p[1]] += p[5]
    for f in state[0].observation.fleets:
        scores[f[1]] += f[6]
    winner = 0 if scores[0] > scores[1] else (1 if scores[1] > scores[0] else -1)
    return scores[0], scores[1], winner

def run_matchup(name_a, agent_a, name_b, agent_b, num_games=100):
    print(f"\n{'='*50}")
    print(f"  {name_a} vs {name_b} ({num_games} games)")
    print(f"{'='*50}")
    
    wins_a, wins_b, draws = 0, 0, 0
    total_a, total_b = 0, 0
    
    for i in range(num_games):
        if i % 2 == 0:
            s0, s1, w = run_one_game(agent_a, agent_b)
            if w == 0: wins_a += 1
            elif w == 1: wins_b += 1
            else: draws += 1
            total_a += s0
            total_b += s1
        else:
            s0, s1, w = run_one_game(agent_b, agent_a)
            if w == 1: wins_a += 1
            elif w == 0: wins_b += 1
            else: draws += 1
            total_a += s1
            total_b += s0
        
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{num_games}] {name_a}: {wins_a}W ({wins_a/(i+1)*100:.0f}%)")
    
    print(f"\n  RESULT: {name_a} {wins_a}W / {name_b} {wins_b}W / Draw {draws}")
    print(f"  AVG SCORE: {name_a}={total_a//num_games} / {name_b}={total_b//num_games}")
    return wins_a, wins_b

if __name__ == "__main__":
    import importlib
    import inspect
    importlib.reload(killer_agent)
    importlib.reload(test_agent)
    
    print(f"DEBUG: killer_agent file: {killer_agent.__file__}")
    print(f"DEBUG: test_agent file: {test_agent.__file__}")
    
    print("\n--- KILLER AGENT SOURCE ---")
    print(inspect.getsource(killer_agent.killer_agent))
    
    print("\n--- TEST AGENT SOURCE ---")
    print(inspect.getsource(test_agent.test_agent))

    run_matchup("KILLER v9.5", killer_agent.killer_agent,
                "TEST", test_agent.test_agent, 100)
