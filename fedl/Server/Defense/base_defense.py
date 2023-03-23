from fedl.Server.Aggregator.aggregator import Aggregator

class BaseDefense(Aggregator):
    @abstractmethod
    def __init__(self):
        pass

    def defend(self, x):
        return x
