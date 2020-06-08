from BanditEnvironment import BanditEnvironment
from BanditAgent import BanditAgent
from Insanity.Selectors import EpsilonGreedyActionSelector
from Insanity.Initializers import OptimisticInitializer
from Insanity.Utils import RunningAverage
import numpy as np
from tqdm import tqdm
from Context import ctx

""" 
Expriment will perform a running average on rewards it receive from environment
set dont_average_reward to True to capture the rewards as they are received and
skip the averaging process.
Sutton and Barton sometimes do this, hence its curves are usually noisy.
"""
class Experiment:
    def __init__(self, µ, σ, ε, num_arms,
                       action_selector, action_values_provider_factory, action_value_initializer,
                       dont_average_reward = False):
        # Crearte and setup environment
        self.__num_arms = num_arms
        self.__env = BanditEnvironment(num_arms, µ, σ)
        self.__env.on_update(self.__handle_environment_update)
        self.__dont_average_reward = dont_average_reward
        action_count = self.__env.action_count

        # We are interested in observing how average reward change as we learn. Using 
        # running average is natural thing here as we are progressing step by step
        # This average is for the whole machine, ie; across all arms
        self.__average_reward = RunningAverage()
        self.__rewards = []
        #
        self.__optimality = []

        # Create agent
        # ActionValuesProvider carries state, hence we need to create one per agent
        provider = action_values_provider_factory(num_arms, action_value_initializer)
        self.__agent = BanditAgent(action_count, provider, action_selector)

    # this runs a full episode with given number of time steps
    def run(self, steps):
        for _ in range(steps):
            action = self.__agent.step()
            is_optimal = self.__env.execute_action(action)
            self.__optimality.append(1 if is_optimal else 0)

    @property
    def rewards(self):
        return np.array(self.__rewards)

    @property
    def optimality(self):
        return np.array(self.__optimality)

    def __handle_environment_update(self, new_state, reward):
        self.__average_reward.addValue(reward)
        used_reward = reward if self.__dont_average_reward else self.__average_reward.value
        self.__rewards.append(used_reward)
        self.__agent.step_outcome(None, reward) # state is not used in this experiment, hence None

# Here, for each ε, we create multiple expriments. Each with different bandit machine
def averaged_runs(µ, σ, ε, selector, provider_factory, initializer, dont_average_reward):
    exs = [Experiment(µ, σ, ε, ctx().num_arms, selector, provider_factory, initializer, dont_average_reward)
                 for _ in range(ctx().num_runs)]
    _ = [ex.run(ctx().num_pulls_per_run) for ex in tqdm(exs)]
    rewards = np.average(np.stack([ex.rewards for ex in exs], axis=1), axis=1)
    percent_optimal_actions = 100 * np.average(np.stack([ex.optimality for ex in exs], axis=1), axis=1)
    return rewards, percent_optimal_actions

def run_experiment():
    results = [averaged_runs(
                ctx().ms[i],
                ctx().σs[i],
                ctx().εs[i],
                ctx().selectors[i](ctx().εs[i], ctx().num_arms),
                ctx().providers[i],
                ctx().initializers[i](),
                ctx().dont_average_reward
               ) for i in range(len(ctx().εs))]
    rewards = [r[0] for r in results]
    percent_optimals = [r[1] for r in results]
    return rewards, percent_optimals