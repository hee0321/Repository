"""Losing game analyzer: find WHY killer loses"""
import random, orbit_wars, killer_agent, test_agent

class ObjDict(dict):
    def __getattr__(self, name):
        return self[name] if name in self else (_ for _ in ()).throw(AttributeError(name))
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

def analyze_loss():
    """Run games until we find one killer loses, then print details"""
    for game_num in range(200):
        random.seed(game_num)
        env = MockEnv()
        state = [MockState(), MockState()]
        orbit_wars.interpreter(state, env)
        agents = [killer_agent.killer_agent, test_agent.test_agent]
        
        history = []
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
            
            # Record state before step
            if step % 25 == 0 or step < 5:
                planets = state[0].observation.planets
                p0 = sum(1 for p in planets if p[1] == 0)
                p1 = sum(1 for p in planets if p[1] == 1)
                pn = sum(1 for p in planets if p[1] == -1)
                s0 = sum(p[5] for p in planets if p[1] == 0)
                s1 = sum(p[5] for p in planets if p[1] == 1)
                prod0 = sum(p[6] for p in planets if p[1] == 0)
                prod1 = sum(p[6] for p in planets if p[1] == 1)
                f0 = sum(f[6] for f in state[0].observation.fleets if f[1] == 0)
                f1 = sum(f[6] for f in state[0].observation.fleets if f[1] == 1)
                history.append(f"  T{step:3d}: P0={p0}planets/{s0+f0:4d}ships/prod{prod0} | P1={p1}planets/{s1+f1:4d}ships/prod{prod1} | neutral={pn}")
            
            state[0].observation.step = step
            orbit_wars.interpreter(state, env)
            if state[0].status == "DONE":
                break
        
        scores = [0, 0]
        for p in state[0].observation.planets:
            if p[1] != -1: scores[p[1]] += p[5]
        for f in state[0].observation.fleets:
            scores[f[1]] += f[6]
        
        if scores[1] > scores[0]:  # Killer LOST
            print(f"\n=== GAME {game_num} (seed={game_num}): KILLER LOST {scores[0]} vs {scores[1]} ===")
            for h in history:
                print(h)
            print()
            # Only show first 3 losses
            if game_num > 20:
                break

if __name__ == "__main__":
    analyze_loss()
