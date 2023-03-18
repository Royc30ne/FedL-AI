import numpy as np
from fedl.Server.Aggregator.aggregator import Aggregator


class FedAvgAggregator(Aggregator):
    def __init__(self):
        super().__init__()

    def aggregate(self, updates):
        weights = [weight for (num_samples, weight, id) in updates]
        num_samples = [num_samples for (num_samples, weight, id) in updates]
        ids = [id for (num_samples, weight, id) in updates]

        # FedAvg with weight
        total_samples = sum(num_samples)
        base = [0] * len(weights[0])
        for i, client_weight in enumerate(weights):
            total_samples += num_samples[i]
            for j, v in enumerate(client_weight):
                base[j] += (num_samples[i] / total_samples * v.astype(np.float64))

        # Update the model
        return base
    