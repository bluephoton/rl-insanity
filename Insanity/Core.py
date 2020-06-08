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

#──────────────────────────────────────────────────────────────────────────────
#                                     Agent
#──────────────────────────────────────────────────────────────────────────────

class IAgent(Interface):
    def step(self):
        pass

    def step_outcome(self, new_state, reward):
        pass

class Agent(implements(IAgent)):
    def __init__(self, action_count, action_values_provider):
        self._action_value_provider = action_values_provider

    def step(self):
        pass

    def step_outcome(self, new_state, reward):
        pass

    