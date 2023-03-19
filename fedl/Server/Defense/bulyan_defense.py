import numpy as np

from base_defense import BaseDefense
from fedl.utils.norm_distances import create_euclidean_dict
from krum_defense import KrumDefense
from trimmed_mean_defense import trimmed_mean

class BulyanDefense(BaseDefense):
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
        return self.bulyan_main(user_weights)
    
    def bulyan_main(self,user_weights):
        users_count = len(user_weights) 

        assert users_count >= 4*self.corrupted_count + 3
        set_size = users_count - 2*self.corrupted_count
        selection_set = []

        distances = create_euclidean_dict(user_weights)
        while len(selection_set) < set_size:
            currently_selected = KrumDefense(self.corrupted_count).krum_main(None, user_weights, users_count, distances, True)
            currently_selected = self.update_model_krum(self.corrupted_count, user_weights, (users_count-len(selection_set)), distances, True)
            selection_set.append(user_weights[currently_selected])

            # remove the selected from next iterations:
            distances.pop(currently_selected)
            for remaining_user in distances.keys():
                distances[remaining_user].pop(currently_selected)

        return trimmed_mean(selection_set, set_size, self.corrupted_count)