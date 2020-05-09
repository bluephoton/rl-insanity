from interface import implements, Interface
from Insanity.Core import IActionValueInitializer
from Insanity.Utils import RunningAverage

class ZeroInitializer(implements(IActionValueInitializer)):
    def initialize_action(self, action, action_count):
        return RunningAverage()

class OptimisticInitializer(implements(IActionValueInitializer)):
    def __init__(self, initial_value):
        self.__initial_value = initial_value

    def initialize_action(self, action, action_count):
        return RunningAverage(self.__initial_value, 0.1)