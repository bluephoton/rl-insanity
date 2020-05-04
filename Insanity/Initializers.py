from interface import implements, Interface
import Insanity.Core as rl

class ZeroActionInitializer(implements(rl.IActionValueInitializer)):
    def initialize_action(self, action, action_count):
        return 0

class OptimisticActionInitialized(implements(rl.IActionValueInitializer)):
    def initialize_action(self, action, action_count):
        return 2