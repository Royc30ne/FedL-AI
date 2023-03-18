import importlib
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v1 as tf
import metrics.writer as metrics_writer

from fedl.Server.Aggregator.aggregator import Aggregator
from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from ..Server.base_server import BaseServer, MDLpoisonServer, MDLpoisonServerNew
from model import ServerModel
from utils.constants import DATASETS
from sklearn.cluster import KMeans
from mlhead_utilfuncs import save_historyfile, save_expr_file

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'

krum_rounds = []
his_acc = []
his_loss = []
his_mcoacc = []
his_assignment =[]

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

def print_custom_metrics(metrics, weights):
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)

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
    def __init__(self, users, groups, train_data, test_data, log_path=None):
        self.users = users
        self.train_data = train_data
        self.test_data = test_data
        self.log_path = log_path
        self.server = None

    def model_config(self, model_path, lr, seed, clients_per_round, poison, poison_rate, aggregator:Aggregator):
        if not os.path.exists(model_path):
            print('Please specify a valid dataset and a valid model.')

        print('############################## %s ##############################' % model_path)
        mod = importlib.import_module(model_path)
        ClientModel = getattr(mod, 'ClientModel')
        # Suppress tf warnings
        tf.logging.set_verbosity(tf.logging.WARN)

        # Create 2 models
        model_params = MODEL_PARAMS[model_path]
        model_params_list = list(model_params)
        model_params_list.insert(0, seed)
        model_params_list[1] = lr
        model_params = tuple(model_params_list)
        tf.reset_default_graph()
        client_model = ClientModel(*model_params)

        # Create clients
        _users = self.users
        groups = [[] for _ in _users]
        clients = [Client(u, g, self.train_data[u], self.test_data[u], client_model) \
                   for u, g in zip(_users, groups)]
        print('%d Clients in Total' % len(clients))

        # Create server
        if poison == True:
            num_workers = int(poison_rate * clients_per_round)
            server = MDLpoisonServerNew(client_model, clients, num_workers, model_params_list[2], clients_per_round)
        else:
            server = BaseServer(client_model, aggregator=aggregator)
        return clients, server, client_model

    def begins(self, config, args):
        clients, self.server, client_model = self.model_config(config, args.dataset, 'cnn',)

        num_rounds = config["num-rounds"]
        eval_every = config["eval-every"]
        epochs_per_round = config['epochs']
        batch_size = config['batch-size']
        clients_per_round = config["clients-per-round"]

        # Test untrained model on all clients
        stat_metrics = self.server.test_model(clients)
        all_ids, all_groups, all_num_samples = self.server.get_clients_info(clients)

        # Simulate training
        micro_acc = 0.
        for i in range(num_rounds):
            print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

            self.server.select_clients(online(clients), num_clients=clients_per_round)
            c_ids, c_groups, c_num_samples = self.server.get_clients_info(None)

            sys_metics = self.server.client_train(single_center=None, num_epochs=epochs_per_round, batch_size=batch_size,
                                            minibatch=None)
            
            self.server.aggregate()

            # Test model on all clients
            if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
                stat_metrics = self.server.test_model(clients)
                loss, micro_acc, macro_acc = print_custom_metrics(stat_metrics, all_num_samples)
                if self.log_path is not None:
                    log_history(i + 1, loss, micro_acc, macro_acc, c_ids, self.log_path)

        client_model.close()
        return micro_acc

    def ends(self, save_model=False):
        print("-" * 3, "End of Krum exerpiment.", "-" * 3)
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