import sys
import orbit_wars
import advanced_agent
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

def run_simulation():
    env = MockEnv()
    num_agents = 2
    state = [MockState() for _ in range(num_agents)]
    
    # Initialize
    orbit_wars.interpreter(state, env)
    
    agents = [farmer_agent.farmer_agent, advanced_agent.advanced_agent]
    
    for step in range(500):
        # Get actions
        for i in range(num_agents):
            if state[i].status == "ACTIVE":
                state[i].observation.step = step
                try:
                    state[i].action = agents[i](state[i].observation)
                except Exception as e:
                    print(f"Agent {i} error: {e}")
                    state[i].action = []
            else:
                state[i].action = []
                
        # Step env
        state[0].observation.step = step
        orbit_wars.interpreter(state, env)
        
        # Check termination
        if state[0].status == "DONE":
            print(f"Game Over at step {step}")
            break

    # Calculate final scores
    scores = [0, 0]
    planets_owned = [0, 0]
    for p in state[0].observation.planets:
        if p[1] != -1:
            scores[p[1]] += p[5]
            planets_owned[p[1]] += 1
    for f in state[0].observation.fleets:
        scores[f[1]] += f[6]
        
    print(f"\n===== SIMULATION RESULT =====")
    print(f"Player 0 (Farmer Agent)   : Score = {scores[0]}, Planets = {planets_owned[0]}")
    print(f"Player 1 (V1 Original)    : Score = {scores[1]}, Planets = {planets_owned[1]}")
    
    if scores[1] > scores[0]:
        print("V1 WON!")
    elif scores[0] > scores[1]:
        print("FARMER WON!")
    else:
        print("🤝 It's a draw!")

if __name__ == "__main__":
    run_simulation()
