import numpy as np
from interface import implements, Interface
from Insanity.Core import IActionValuesProvider

class UCBActionValuesProvider(implements(IActionValuesProvider)):
    def __init__(self, action_count, action_value_initializer, c):
        self.__c = c
        self.__t = 0
        self.__n = [0 for _ in range(action_count)]
        self.__action_count = action_count
        self.__action_value_initializer = action_value_initializer
        self._action_values = [self.__get_initial_action_value(i) for i in range(action_count)]

    @property
    def action_count(self):
        return self.__action_count

    # this is called at time t to select an action
    @property
    def action_values(self):
        exploit = np.array([av.value for av in self._action_values])
        explore = np.array([self.__calc_explore_part(a) for a in range(self.__action_count)])
        return  exploit, explore

    # this is called at time t+1 as it takes time for environment to produce reward
    def update_action_value(self, action, new_value):
        self.__t += 1
        # increment the number of times action is taken upto now (t+1 not included), so it
        # is lagged by one as expected
        self.__n[action] += 1
        # as usual update the value of Q
        self._action_values[action].addValue(new_value)

    def __get_initial_action_value(self, action):
        return self.__action_value_initializer.initialize_action(action, self.__action_count)

    def __calc_explore_part(self, a):
        # c = 0 means we don't want to promote any exploration
        if self.__c == 0:
            return 0

        # if we never tried an action, we give it very large value so we can select it, accoriding to book:
        # "If Nt (a) = 0, then a is considered to be a maximizing action"
        if self.__n[a] == 0:
            return float("inf")

        # self.__t will never be 0 as it would've been caught up
        return self.__c * np.sqrt(np.log(self.__t) / self.__n[a])
