import numpy as np
from abc import ABC, abstractmethod

class Aggregator(ABC):
    def __init__(self):
        super().__init__()

    def aggregate(self, updates):
        return updates
    