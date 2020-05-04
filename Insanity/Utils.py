class RunningAverage:
    def __init__(self, initial_value = 0):
        self.__count = 0
        self.__average = initial_value

    def addValue(self, value):
        self.__count += 1
        self.__average += (value - self.__average) / self.__count

    @property
    def value(self):
        return self.__average