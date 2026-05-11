import orbit_wars
import mastermind_v30

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

env = MockEnv()
state = [MockState() for _ in range(2)]
orbit_wars.interpreter(state, env)

# Run 5 steps and print V30's moves
for step in range(5):
    state[0].observation.step = step
    state[1].observation.step = step
    
    obs0 = dict(state[0].observation)
    obs0['player'] = 0
    obs0['step'] = step
    
    try:
        moves = mastermind_v30.agent(obs0)
        print(f"Step {step}: V30 moves = {moves}")
    except Exception as e:
        print(f"Step {step}: ERROR = {e}")
        import traceback
        traceback.print_exc()
    
    # Dummy actions
    state[0].action = []
    state[1].action = []
    state[0].observation.step = step
    orbit_wars.interpreter(state, env)
