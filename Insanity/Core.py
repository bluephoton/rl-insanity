import numpy as np
from interface import implements, Interface

#──────────────────────────────────────────────────────────────────────────────
#                                   Environment
#──────────────────────────────────────────────────────────────────────────────
class IEnvironment(Interface):
    def setup(self):
        pass

    def cleanup(self):
        pass

    @property
    def action_count(self):
        pass

    def execute_action(self, action):
        pass

    # update_handler is a function of lambds that accept two 
    # parameters, state and reward
    def on_update(self, update_handler):
        pass

#──────────────────────────────────────────────────────────────────────────────
#                                 Action Related
#──────────────────────────────────────────────────────────────────────────────
class IActionSelector(Interface):
    def select_action(self, action_values_provider):
        pass

class IActionValueInitializer(Interface):
    def initialize_action(self, action, action_count):
        pass

class IActionValuesProvider(Interface):
    @property
    def action_count(self):
        pass

    @property
    def action_values(self):
        pass

    def update_action_value(self, action, new_value):
        pass

class ActionValuesProvider(implements(IActionValuesProvider)):
    def __init__(self, action_count, action_value_initializer):
        self.__action_count = action_count
        self.__action_value_initializer = action_value_initializer
        self._action_values = np.array([ self.__get_initial_action_value(i) for i in range(action_count)])

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

#──────────────────────────────────────────────────────────────────────────────
#                                     Agent
#──────────────────────────────────────────────────────────────────────────────

class IAgent(Interface):
    def step(self):
        pass

    def step_outcome(self, new_state, reward):
        pass

class Agent(implements(IAgent)):
    def __init__(self, action_count, action_value_initializer):
        self._action_value_provider = ActionValuesProvider(action_count, action_value_initializer)

    def step(self):
        pass

    def step_outcome(self, new_state, reward):
        pass

    