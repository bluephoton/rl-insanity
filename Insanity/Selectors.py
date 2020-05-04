import numpy as np
from interface import implements, Interface
import Insanity.Core as rl

class EpsilonGreedyActionSelector(implements(rl.IActionSelector)):
    def __init__(self, ε, action_count):
        self.ε = ε
        self.__step = 0
        self.__action_count = action_count

    def select_action(self, action_values_provider):
        self.__step += 1

        if self.__step == 1:
            return self.__explore()

        if np.random.rand() < self.ε:
            # Explore
            return self.__explore()

        # Exploit
        return self.__exploit(action_values_provider)

    def __explore(self):
        return np.random.choice(self.__action_count)

    def __exploit(self, action_values_provider):
        # https://stackoverflow.com/questions/42071597/numpy-argmax-random-tie-breaking
        values = action_values_provider.action_values
        return np.random.choice(np.flatnonzero(np.isclose(values, values.max())))