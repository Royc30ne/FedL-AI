import numpy as np

from base_defense import BaseDefense
from ..utils import create_euclidean_dict

class KrumDefense(BaseDefense):
    # Pass configuration
    def __init__(self, config):
        self.config = config
        self.corrupted_count = config['defense']['krum']['corrupted_count']

    def defend(self, updates):
        """
        Krum defense
        :param updates: list of updates
        :return: the defense update
        """
        return self.krum_main(updates)

    def krum_main(self, updates, users_weights=None, user_count=None, distances=None, return_index=False):
        if users_weights is None:
            clients_weights = [u[1] for u in updates]
        else:
            clients_weights = users_weights

        if user_count is None:
            client_count = len(clients_weights)
        else:
            client_count = user_count

        if not return_index:
            assert client_count >= 2*self.corrupted_count + 1,('users_count>=2*corrupted_count + 3', client_count, self.corrupted_count)
        
        non_malicious_count = client_count - self.corrupted_count
        
        if distances is None:
            scores = self._krum_scores(clients_weights, non_malicious_count)
        else:
            scores = [sum(sorted(distances[user].values())[:non_malicious_count]) for user in distances.keys()]
                
        minimal_error_index = np.argmin(scores)
            
        if return_index:
            return minimal_error_index # return the index of the client with the minimal error
        else:
            return clients_weights[minimal_error_index] # return the weights of the client with the minimal error
    
    def _krum_scores(self, weight_list, non_malicious_count):
        distances = create_euclidean_dict(weight_list)
        scores = [sum(sorted(distances[user].values())[:non_malicious_count]) for user in distances.keys()]
        return scores