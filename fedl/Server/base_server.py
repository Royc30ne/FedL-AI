import numpy as np

from abc import ABC, abstractmethod
from fedl.Server.Aggregator.aggregator import Aggregator
from fedl.utils.model_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY
class BaseServer(ABC):
    def __init__(self, client_model, aggregator:Aggregator):
        self.client_model = client_model
        self.aggregator = aggregator
        if client_model is not None:
            self.model = client_model.get_params()
        self.selected_clients = []
        self.adversaries = []
        self.updates = []

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
        self.updates = []
        return self.model
    
    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.
        Returns info about self.selected_clients if clients=None;
        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples
    
    def test_model(self, clients_to_test, set_to_use='test'):
        """
        Test model for different comparison of metrics
        and save them to report file for plotting

        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for client in clients_to_test:
            client.model.set_params(self.model)
            c_metrics = client.test(set_to_use)
            metrics[client.id] = c_metrics

        return metrics
    
    def select_clients(self, possible_clients, num_clients=20, my_round=100):
        """Selects num_clients clients randomly from possible_clients.
        

        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).
            
        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round + np.random.randint(10000)) 
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]
    
    
    def __model__(self):
        return self.model