import numpy as np

from abc import ABC, abstractmethod

class BaseServer(ABC):
    def __init__(self, client_model):
        self.client_model = client_model
        if client_model is not None:
            self.model = client_model.get_params()
        self.selected_clients = []
        self.adversaries = []
        self.updates = []

    def set_clients(self, clients, num_clients, cur_round):
        self.clients = clients
        num_clients = min(num_clients, len(clients))
        np.random.seed(cur_round + np.random.randint(10000)) 
        self.selected_clients = np.random.choice(clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples, c.id) for c in self.selected_clients]

    def normal_client_train():
        return

    def clean_client_train():
        return