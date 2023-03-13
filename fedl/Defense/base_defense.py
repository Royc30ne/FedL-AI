from abc import ABC, abstractmethod

class BaseDefense(ABC):
    @abstractmethod
    def __init__(self, config):
        pass

    def defense(self, x):
        return x
