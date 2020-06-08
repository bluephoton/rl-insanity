import numpy as np
from interface import implements, Interface
from Insanity.Core import IActionSelector

class EpsilonGreedyActionSelector(implements(IActionSelector)):
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

class UCBActionSelector(implements(IActionSelector)):
    def __init__(self):
        self.__step = 0

    def select_action(self, action_values_provider):
        exploit, explore = action_values_provider.action_values
        values = exploit + explore
        return np.random.choice(np.flatnonzero(np.isclose(values, values.max())))

class GradientSelector(implements(IActionSelector)):
    def __init__(self):
        self.__step = 0

    def select_action(self, action_values_provider):
        values = action_values_provider.action_values
        # Book is not explicit on this, but we need to sample from this probability mass function (discrete)
        return np.random.choice(len(values), p=values)