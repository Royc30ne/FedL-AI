import numpy as np
from abc import ABC, abstractmethod

class Aggregator(ABC):
    def __init__(self, updates):
        self.weights = [weight for (num_samples, weight, id) in updates]
        self.num_samples = [num_samples for (num_samples, weight, id) in updates]
        self.ids = [id for (num_samples, weight, id) in updates]

    @abstractmethod
    def aggregate(self):
        return NotImplemented
    
    

class FedAvgAggregator(Aggregator):
    def __init__(self, updates):
        super().__init__(updates)

    def aggregate(self):
        # FedAvg with weight
        total_samples = 0.
        base = [0] * len(self.updates[0][1])
        for i, client_weight in enumerate(self.weights):
            total_samples += self.num_samples[i]
            for j, v in enumerate(client_weight):
                base[j] += (self.num_samples[i] * v.astype(np.float64))
        averaged_model = [v / total_samples for v in base]

        # Update the model
        return averaged_model
    