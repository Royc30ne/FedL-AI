from abc import ABC, abstractmethod

class BaseDefense(ABC):
    @abstractmethod
    def __init__(self, config):
        pass

    def defend(self, x):
        return x
