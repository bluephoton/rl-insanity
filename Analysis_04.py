import matplotlib.pyplot as pp
import pandas as pd
import time as t
import numpy as np
from tqdm import tqdm
from BanditMachine import BanditMachine
from Experiment import run_experiment
from Insanity.Initializers import OptimisticInitializer, ZeroInitializer
from ActionValuesProvider import ActionValuesProvider
from Context import ctx, use_context
from munch import munchify
%matplotlib inline

#───────────────────────────────────────────────────────────────────────
#                           Experiment Start
#───────────────────────────────────────────────────────────────────────
use_context(4)

before = t.perf_counter()
rewards, percent_optimals = run_experiment()
print("time (seconds): {0}".format(t.perf_counter() - before))

#───────────────────────────────────────────────────────────────────────
#                              Plotting
#───────────────────────────────────────────────────────────────────────
labels = ["α={0}{1}".format(α, " (no baseline)" if i > 1 else "") for i,α in enumerate([0.1, 0.4, 0.1, 0.4])]

pp.figure(figsize=(10,4))
[pp.plot(percent_optimals[i], label=labels[i]) for i in range(len(ctx().εs))]
pp.legend(bbox_to_anchor=(1.2, 0.5)) 
pp.xlabel("Steps") 
pp.ylabel("% Optimal Action") 
pp.title("Gradient Bandit (averaged over " + str(ctx().num_runs) + " Runs)") 
pp.show()