import numpy as np

from abc import ABC, abstractmethod
from fedl.Server.Aggregator.aggregator import Aggregator
from fedl.utils.model_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY
class BaseServer(ABC):
    def __init__(self, client_model, aggregator:Aggregator, model_params=None):
        self.client_model = client_model
        self.aggregator = aggregator
        if model_params is not None:
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

    def client_train(self, single_center, num_epochs=1, batch_size=10, minibatch=None, clients=None, apply_prox=False):
        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}
        for c in clients:
            if single_center is not None:
                c.model.set_params(single_center)
            else:
                c.model.set_params(self.model)
            comp, num_samples, update = c.train(num_epochs, batch_size, minibatch, apply_prox)

            sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
            sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp

            self.updates.append((num_samples, update, c.id))

        return sys_metrics, self.updates

    def aggregate(self):
        self.model = self.aggregator.aggregate(self.updates)
        return self.model
    
    def __model__(self):
        return self.model