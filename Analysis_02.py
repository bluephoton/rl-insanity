import matplotlib.pyplot as pp
import pandas as pd
import time as t
import numpy as np
from tqdm import tqdm
from BanditMachine import BanditMachine
from Experiment import Experiment
from Insanity.Initializers import OptimisticInitializer, ZeroInitializer

# Here, for each ε, we create multiple expriments. Each with different bandit machine
def averaged_runs(ε, num_runs, num_pulls_per_run, initializer):
    exs = [Experiment(ε, 10, initializer) for _ in range(num_runs)]
    _ = [ex.run(num_pulls_per_run) for ex in tqdm(exs)]
    rewards = np.average(np.stack([ex.rewards for ex in exs], axis=1), axis=1)
    percent_optimal_actions = 100 * np.average(np.stack([ex.optimality for ex in exs], axis=1), axis=1)
    return rewards, percent_optimal_actions

def run_experiment(num_runs, num_pulls_per_run, εs, initializers):
    results = [averaged_runs(εs[i], num_runs, num_pulls_per_run, initializers[i]) for i in range(len(εs))]
    rewards = [r[0] for r in results]
    percent_optimals = [r[1] for r in results]
    return rewards, percent_optimals

#───────────────────────────────────────────────────────────────────────
#                        Experiment Start
#───────────────────────────────────────────────────────────────────────

# Run our experiment
εs = [0, 0.1]
initializers = [OptimisticInitializer(5, 0.1), ZeroInitializer()]
num_runs = 2000
num_pulls_per_run = 1000
before = t.perf_counter()
rewards, percent_optimals = run_experiment(num_runs, num_pulls_per_run, εs, initializers)
print("time (seconds): {0}".format(t.perf_counter() - before))

# plotting
labels = ["ε={0}{1}".format(ε, (" (greedy, optimistic, Q₁=5), " if ε == 0 else " realistic, Q₁=0")) for ε in εs]

pp.figure(figsize=(10,4))
[pp.plot(percent_optimals[i], label=labels[i]) for i in range(len(εs))]
pp.legend(bbox_to_anchor=(1.2, 0.5)) 
pp.xlabel("Steps") 
pp.ylabel("% Optimal Action") 
pp.title("% times optical action selected averaged over " + str(num_runs) + " Runs") 
pp.show()

