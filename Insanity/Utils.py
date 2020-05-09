class RunningAverage:
    def __init__(self, initial_value = 0, α = None):
        self.__count = 0
        self.__average = initial_value
        self.__α = α

    def addValue(self, value):
        self.__count += 1
        α = 1 / self.__count if self.__α == None else self.__α 
        self.__average += α * (value - self.__average)

    @property
    def value(self):
        return self.__average