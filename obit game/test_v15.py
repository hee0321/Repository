import sys
import orbit_wars
import mastermind_v15
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

def run_one_game(agent_a, agent_b, num_agents=2):
    env = MockEnv(num_agents)
    state = [MockState() for _ in range(num_agents)]
    orbit_wars.interpreter(state, env)
    agents = [agent_a, agent_b] if num_agents == 2 else [agent_a, agent_b, agent_b, agent_b]
    for step in range(500):
        for i in range(num_agents):
            if state[i].status == "ACTIVE":
                state[i].observation.step = step
                try: state[i].action = agents[i](state[i].observation)
                except Exception as e: 
                    import traceback
                    traceback.print_exc()
                    state[i].action = []
            else: state[i].action = []
        state[0].observation.step = step
        orbit_wars.interpreter(state, env)
        if state[0].status == "DONE": break
    scores = [0] * num_agents
    for p in state[0].observation.planets:
        if p[1] != -1: scores[p[1]] += p[5]
    for f in state[0].observation.fleets:
        scores[f[1]] += f[6]
    winner = scores.index(max(scores)) if max(scores) > 0 else -1
    return scores, winner

def run_matchup(name_a, agent_a, name_b, agent_b, num_games=30):
    print(f"\n{'='*60}")
    print(f"  {name_a} vs {name_b} - {num_games} games")
    print(f"{'='*60}")
    wins_a, wins_b = 0, 0
    for i in range(num_games):
        if i % 2 == 0:
            scores, w = run_one_game(agent_a, agent_b)
            if w == 0: wins_a += 1
            elif w == 1: wins_b += 1
        else:
            scores, w = run_one_game(agent_b, agent_a)
            if w == 1: wins_a += 1
            elif w == 0: wins_b += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1:>2}/{num_games}] {name_a}: {wins_a}W | {name_b}: {wins_b}W")
    print(f"  FINAL: {name_a} {wins_a}W / {name_b} {wins_b}W")
    return wins_a

def run_ffa(name, our_agent, opp_name, opp_agent, num_games=20):
    print(f"\n{'='*60}")
    print(f"  4P FFA: {name} vs 3x {opp_name} - {num_games} games")
    print(f"{'='*60}")
    wins = 0
    placements = {1:0, 2:0, 3:0, 4:0}
    for g in range(num_games):
        pos = g % 4
        agents = [opp_agent] * 4
        agents[pos] = our_agent
        scores, winner = run_one_game(agents[0], agents[1], num_agents=4) # This doesn't actually work right for 4 agents in original logic
        # Correct 4p logic
        env = MockEnv(4); state = [MockState() for _ in range(4)]; orbit_wars.interpreter(state, env)
        for step in range(500):
            for i in range(4):
                if state[i].status == "ACTIVE":
                    state[i].observation.step = step
                    try: state[i].action = agents[i](state[i].observation)
                    except Exception: state[i].action = []
            state[0].observation.step = step
            orbit_wars.interpreter(state, env)
            if state[0].status == "DONE": break
        res_scores = [0] * 4
        for p in state[0].observation.planets:
            if p[1] != -1: res_scores[p[1]] += p[5]
        for f in state[0].observation.fleets:
            res_scores[f[1]] += f[6]
        
        our_s = res_scores[pos]
        sorted_s = sorted(res_scores, reverse=True)
        place = sorted_s.index(our_s) + 1
        placements[place] += 1
        if res_scores.index(max(res_scores)) == pos: wins += 1
        
        if (g + 1) % 5 == 0:
            print(f"  [{g+1:>2}/{num_games}] Wins: {wins} | 1st={placements[1]} 2nd={placements[2]}")
    
    print(f"  FINAL: {name} {wins}W / {num_games}G")
    print(f"  Placements: 1st={placements[1]} | 2nd={placements[2]} | 3rd={placements[3]} | 4th={placements[4]}")
    return wins

if __name__ == "__main__":
    run_matchup("EMPEROR v15", mastermind_v15.agent, "TEST_BOT", test_agent.test_agent, 30)
    run_matchup("EMPEROR v15", mastermind_v15.agent, "OVERLORD v13", mastermind_v13.agent, 30)
    run_ffa("EMPEROR v15", mastermind_v15.agent, "TEST_BOT", test_agent.test_agent, 20)
