import importlib
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v1 as tf
import fedl.utils.metrics_utils as metrics_utils

from fedl.Server.Aggregator.aggregator import Aggregator
from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from fedl.Client.client_container import Client
from fedl.Client.client_model.base_client_model import BaseClientModel
from ..Server.base_server import BaseServer, MDLpoisonServer, MDLpoisonServerNew
from sklearn.cluster import KMeans
STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'


def online(clients):
    """We assume all users are always online."""
    return clients


def save_model(server_model, dataset, model):
    """Saves the given server model on checkpoints/dataset/model.ckpt."""
    # Save server model
    ckpt_path = os.path.join('checkpoints', dataset)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server_model.save(os.path.join(ckpt_path, '%s.ckpt' % model))
    print('Model saved in path: %s' % save_path)

def print_metrics(metrics, weights):
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_utils.get_metrics_names(metrics)

    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print('%s: %g, 10th percentile: %g, 90th percentile %g' \
              % (metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 90)))
        
    micros = [metrics[c]['microf1'] for c in sorted(metrics)]
    final_micro = np.average(micros, weights=ordered_weights)
    loss = [metrics[c]['loss'] for c in sorted(metrics)]
    final_loss = np.average(loss, weights=ordered_weights)
    macro = [metrics[c]['macrof1'] for c in sorted(metrics)]
    final_macro = np.average(macro, weights=ordered_weights)
    return final_loss, final_micro, final_macro

class Base_Tainer:
    def __init__(self, users, groups, train_data, test_data, num_class, log_path=None):
        self.users = users
        self.train_data = train_data
        self.test_data = test_data
        self.num_class = num_class
        self.log_path = log_path
        self.server = None
        self.clients = []
        self.client_model = None

    def model_config(self, client_model: BaseClientModel, clients_per_round, poison, poison_rate, aggregator:Aggregator):
        print('############################################################')
        tf.reset_default_graph()
        self.client_model = client_model

        # Create clients
        _users = self.users
        groups = [[] for _ in _users]
        self.clients = [Client(u, g, self.train_data[u], self.test_data[u], client_model) \
                   for u, g in zip(_users, groups)]
        print('%d Clients in Total' % len(self.clients))

        # Create server
        # if poison == True:
        #     num_workers = int(poison_rate * clients_per_round)
        #     self.server = MDLpoisonServerNew(client_model, self.clients, num_workers, self.num_class, clients_per_round)
        # else:
        self.server = BaseServer(client_model, aggregator=aggregator)
        return self.clients, self.server, self.client_model

    def begins(self, num_rounds, eval_every, epochs_per_round, batch_size, clients_per_round):

        # Test untrained model on all clients
        stat_metrics = self.server.test_model(self.clients)
        all_ids, all_groups, all_num_samples = self.server.get_clients_info(self.clients)

        # Simulate training
        micro_acc = 0.
        for i in range(num_rounds):
            print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

            self.server.select_clients(online(self.clients), num_clients=clients_per_round)
            c_ids, c_groups, c_num_samples = self.server.get_clients_info(None)

            sys_metics = self.server.client_train(single_center=None, num_epochs=epochs_per_round, batch_size=batch_size,
                                            minibatch=None)
            
            self.server.aggregate()

            # Test model on all clients
            if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
                stat_metrics = self.server.test_model(self.clients)
                loss, micro_acc, macro_acc = print_metrics(stat_metrics, all_num_samples)
                if self.log_path is not None:
                    log_history(i + 1, loss, micro_acc, macro_acc, c_ids, self.log_path)

        self.client_model.close()
        return micro_acc

    def ends(self, save_model=False):
        print("-" * 3, "End of exerpiment.", "-" * 3)
        if save_model:
            print("Saving model...")
            save_model(self.server.__model__)
        return


def log_history(my_rounds, loss, micro_acc, macro_acc, client_list,filename):
    df = pd.DataFrame({'round': my_rounds, 'loss':loss ,'micro': micro_acc, 'macro': macro_acc, 'clients': client_list}, index=[0])
    if my_rounds == 1:
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)