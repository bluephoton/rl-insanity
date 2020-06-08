from interface import implements, Interface
from Insanity.Core import Agent
from Insanity.Utils import RunningAverage
from ActionValuesProvider import ActionValuesProvider
import numpy as np

class BanditAgent(Agent):
    def __init__(self, action_count, action_values_provider, action_selector):
        Agent.__init__(self, action_count, action_values_provider)
        self.__action_count = action_count
        self.__action_selector = action_selector
        self.__recent_action = None

    def step(self):  
        self.__recent_action =  self.__action_selector.select_action(self._action_value_provider)
        return self.__recent_action

    def step_outcome(self, new_state, reward):
        self._action_value_provider.update_action_value(self.__recent_action, reward)
        self.__recent_action = None # forget the action so if called twice bad things happen!
