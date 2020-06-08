import numpy as np
from interface import implements, Interface
from Insanity.Core import IActionValuesProvider

class GradientActionValuesProvider(implements(IActionValuesProvider)):
    def __init__(self, action_count, action_value_initializer, α, no_baseline = False):
        self.__action_count = action_count
        self.__action_value_initializer = action_value_initializer
        self.__α = α
        self.__H = [0 for i in range(action_count)]
        self.__Rbar = [self.__get_initial_action_value(i) for i in range(action_count)]
        # start with all actions have equal propability
        self.__π = [1/action_count for i in range(action_count)]
        self.__no_baseline = no_baseline

    @property
    def action_count(self):
        return self.__action_count

    @property
    def action_values(self):
        return np.array(self.__π)

    def update_action_value(self, action, new_value):
        self.__update_π(action, new_value)
        if(not self.__no_baseline):
            self.__Rbar[action].addValue(new_value)

    def __get_initial_action_value(self, action):
        return self.__action_value_initializer.initialize_action(action, self.__action_count)

    def __update_π(self, action, R):
        for i in range(self.__action_count):
            if i == action:
                self.__H[i] += self.__α * (R - self.__Rbar[i].value) * (1 - self.__π[i])
            else:
                self.__H[i] -= self.__α * (R - self.__Rbar[i].value) * (self.__π[i])

        sum = np.sum(np.exp(np.array(self.__H)))
        for i in range(self.__action_count):
            self.__π[i] = np.exp(self.__H[i]) / sum
