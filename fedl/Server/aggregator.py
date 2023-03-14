import numpy as np
from abc import ABC, abstractmethod

class Aggregator(ABC):
    def __init__(self, updates):
        self.updates = updates

    @abstractmethod
    def aggregate(self, updates):
        return self.defense(self.defense_aggregate(updates))
    
    

class FedAvgAggregator(Aggregator):
    def __init__(self, updates):
        super().__init__(updates)

    def aggregate(self):
        # FedAvg with weight
        total_weight = 0.
        base = [0] * len(self.updates[0][1])
        for (client_samples, client_model, id) in self.updates:
            total_weight += client_samples
            for i, v in enumerate(client_model):
                base[i] += (client_samples * v.astype(np.float64))
        averaged_model = [v / total_weight for v in base]

        # Update the model
        return averaged_model
    