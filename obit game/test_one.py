import sim
import mastermind_v31
import mastermind_v30

mastermind_v31.GLOBAL_STATE = {'planet_history': {}}
mastermind_v30.GLOBAL_STATE = {'planet_history': {}}

print("Starting single game test...")
sim.run_simulation(mastermind_v31.agent, mastermind_v30.agent, "V31", "V30")
print("Done.")
