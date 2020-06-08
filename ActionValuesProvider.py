import numpy as np
from interface import implements, Interface
from Insanity.Core import IActionValuesProvider

class ActionValuesProvider(implements(IActionValuesProvider)):
    def __init__(self, action_count, action_value_initializer):
        self.__action_count = action_count
        self.__action_value_initializer = action_value_initializer
        self._action_values = [self.__get_initial_action_value(i) for i in range(action_count)]

    @property
    def action_count(self):
        return self.__action_count

    @property
    def action_values(self):
        return np.array([av.value for av in self._action_values])

    def update_action_value(self, action, new_value):
        self._action_values[action].addValue(new_value)

    def __get_initial_action_value(self, action):
        return self.__action_value_initializer.initialize_action(action, self.__action_count)
