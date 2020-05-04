from BanditEnvironment import BanditEnvironment
from BanditAgent import BanditAgent
from Insanity.Selectors import EpsilonGreedyActionSelector
from Insanity.Initializers import ZeroActionInitializer
from Insanity.Utils import RunningAverage
import numpy as np

class Experiment:
    def __init__(self, ε, num_arms):
        # Crearte and setup environment
        self.__num_arms = num_arms
        self.__env = BanditEnvironment(num_arms)
        self.__env.on_update(self.__handle_environment_update)
        action_count = self.__env.action_count

        # We are interested in observing how average reward change as we learn. Using 
        # running average is natural thing here as we are progressing step by step
        # This average is for the whole machine, ie; across all arms
        self.__average_reward = RunningAverage()
        self.__rewards = []
        #
        self.__optimality = []

        # Create agent
        action_value_initializer = ZeroActionInitializer()
        action_selector = EpsilonGreedyActionSelector(ε, action_count)
        self.__agent = BanditAgent(action_count, action_value_initializer, action_selector)

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
        self.__rewards.append(self.__average_reward.value)
        self.__agent.step_outcome(None, reward) # state is not used in this experiment, hence None

