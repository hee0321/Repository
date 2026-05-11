import sys
import orbit_wars
import mastermind_v13
import test_agent

class ObjDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"No such attribute: {name}")
    def __setattr__(self, name, value):
        self[name] = value

class MockEnv:
    def __init__(self, num_agents=4):
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

def run_ffa_game(agents_list):
    """Run a 4-player FFA game. agents_list is list of 4 agent functions."""
    n = len(agents_list)
    env = MockEnv(n)
    state = [MockState() for _ in range(n)]
    orbit_wars.interpreter(state, env)
    
    for step in range(500):
        for i in range(n):
            if state[i].status == "ACTIVE":
                state[i].observation.step = step
                try:
                    state[i].action = agents_list[i](state[i].observation)
                except Exception as e:
                    print(f"Error in agent {i} at step {step}: {e}")
                    import traceback
                    traceback.print_exc()
                    state[i].action = []
            else:
                state[i].action = []
        state[0].observation.step = step
        orbit_wars.interpreter(state, env)
        if state[0].status == "DONE":
            break
    
    scores = [0] * n
    for p in state[0].observation.planets:
        if p[1] != -1:
            scores[p[1]] += p[5]
    for f in state[0].observation.fleets:
        scores[f[1]] += f[6]
    
    max_score = max(scores)
    winner = scores.index(max_score)
    return scores, winner

def run_ffa_tournament(our_agent, other_agents, our_name="OVERLORD", num_games=20):
    """Run FFA tournament: our agent at position 0, others fill remaining slots."""
    print(f"\n=== 4P FFA TOURNAMENT ({num_games} games) ===")
    print(f"Player 0: {our_name}")
    for i, (name, _) in enumerate(other_agents):
        print(f"Player {i+1}: {name}")
    
    wins = [0] * 4
    total_scores = [0] * 4
    our_placements = {1: 0, 2: 0, 3: 0, 4: 0}
    
    for g in range(num_games):
        # Rotate positions every game for fairness
        agents = [our_agent]
        for _, a in other_agents:
            agents.append(a)
        
        # Shuffle starting positions
        import random
        our_pos = g % 4
        agent_order = [None] * 4
        agent_order[our_pos] = our_agent
        other_idx = 0
        for i in range(4):
            if i != our_pos:
                agent_order[i] = other_agents[other_idx % len(other_agents)][1]
                other_idx += 1
        
        scores, winner = run_ffa_game(agent_order)
        
        # Track our performance
        our_score = scores[our_pos]
        sorted_scores = sorted(scores, reverse=True)
        our_placement = sorted_scores.index(our_score) + 1
        our_placements[our_placement] += 1
        total_scores[0] += our_score
        
        if winner == our_pos:
            wins[0] += 1
        
        if (g + 1) % 5 == 0:
            print(f"  [{g+1}/{num_games}] Wins: {wins[0]} | Avg Score: {total_scores[0]/(g+1):.0f}")
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"  {our_name}: {wins[0]}W / {num_games} games ({wins[0]/num_games*100:.0f}% win rate)")
    print(f"  Avg Score: {total_scores[0]/num_games:.0f}")
    print(f"  Placements: 1st={our_placements[1]} | 2nd={our_placements[2]} | 3rd={our_placements[3]} | 4th={our_placements[4]}")
    return wins[0]

if __name__ == "__main__":
    # First: 1v1 against test_bot (30 games)
    from test_v13 import run_matchup
    run_matchup("OVERLORD v13", mastermind_v13.agent, "TEST_BOT", test_agent.test_agent, 30)
    
    # Then: 4P FFA
    other_agents = [
        ("TEST_BOT_1", test_agent.test_agent),
        ("TEST_BOT_2", test_agent.test_agent),
        ("TEST_BOT_3", test_agent.test_agent),
    ]
    run_ffa_tournament(mastermind_v13.agent, other_agents, "OVERLORD v13", 20)
