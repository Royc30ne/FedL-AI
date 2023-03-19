import numpy as np

from base_defense import BaseDefense
from scipy import stats


def trimmed_mean(users_weights, client_count, corrupted_count=None):
    base = [0] * len(users_weights[0])
    for layer in range(len(users_weights[0])):
        curr_layer_per_client = [0] * client_count
        
        for n, client_model in enumerate(users_weights):
            curr_layer_per_client[n] = (1 * client_model[layer].astype(np.float64))

        trimmed_per = float(corrupted_count) / float(client_count)
        base[layer] += (stats.trim_mean(curr_layer_per_client, trimmed_per))

    return base

class TrimmedMeanDefense(BaseDefense):
    # Pass configuration
    def __init__(self, corrupted_count):
        self.corrupted_count = corrupted_count

    def defend(self, updates):
        """
        Krum defense
        :param updates: list of updates
        :return: the defense update
        """
        user_weights = [u[1] for u in updates]
        client_count = len(user_weights)
        return self.trimmed_mean(user_weights, client_count, self.corrupted_count)
