from abc import ABC, abstractmethod

class BaseDefense(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def defend(self, x):
        return x
