from interface import implements, Interface
from BanditMachine import BanditMachine
from Insanity.Core import IEnvironment

class BanditEnvironment(implements(IEnvironment)):
    def __init__(self, num_arms):
        self.__bandit = BanditMachine(num_arms)
        self.__update_handler = None

    def setup(self):
        pass

    def cleanup(self):
        pass

    @property
    def action_count(self):
        return self.__bandit.number_of_arms

    def on_update(self, update_handler):
        self.__update_handler = update_handler

    def execute_action(self, action):
        # returns two values: reward, isOptimum
        reward, is_optimal = self.__bandit.pull_arm(action)
        self.__trigger_update(reward)
        return is_optimal

    def __trigger_update(self, reward):
        if(self.__update_handler != None):
            self.__update_handler(None, reward)