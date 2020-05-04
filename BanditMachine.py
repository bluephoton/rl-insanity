import numpy as np

class Arm:
    def __init__(self, µ, σ):
        self.reward_µ = µ               # mean of the actual q* (unknown) reward PDF for this arm
        self.reward_σ = σ               # sigma of the actual q* (unknown) reward PDF for this arm

    def pull(self):
        reward =  np.random.normal(self.reward_µ, self.reward_σ)
        return reward

class BanditMachine:
    def __init__(self, number_of_arms):
        # We use random sampled from N(0,1) distribution to initialize q* for our simulation
        self.__arms = [Arm(np.random.normal(0, 1), 1) for _ in range(number_of_arms)]
        # cache the index of arm with maximum expected reward
        arm_values = [a.reward_µ for a in self.__arms]
        self.__optimum_arm = np.argmax(arm_values)

    # Returns two values:
    #  reward
    #  optimality: true if arm was optimal false otherwise
    def pull_arm(self, arm):
        reward = self.__arms[arm].pull()
        return reward, (self.__optimum_arm == arm)

    @property
    def number_of_arms(self):
        return len(self.__arms)

    @property
    def optimal_reward(self):
        return self.__arms[self.__optimum_arm].reward_µ
