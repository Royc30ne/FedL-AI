import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))
 
from fedl.utils.dataset_utils import load_ez_dataset, read_data
from fedl.Client.client_model.cnn_client_model import CNNClientModel
from fedl.Client.client_container import Client
from fedl.Server.Aggregator.fedavg import FedAvgAggregator
from fedl.Server.base_server import BaseServer
from fedl.Trainer.base_trainer import BaseTrainer

# Load dataset
# train_data_path = 'data/20news/train'
# test_data_path = 'data/20news/test'
# client_ids, group_ids, train_data, test_data = load_data(train_data_path, test_data_path)
train_data_path = '/home/royc30ne/projects/age-based-multi-center-FL/data/femnist/data/train'
test_data_path = '/home/royc30ne/projects/age-based-multi-center-FL/data/femnist/data/test'
ez_data_path = '/home/royc30ne/projects/FedL-AI/datasets/ez-dataset/femnist_niid_full_dataset.npz'
client_ids, group_ids, train_data, test_data = read_data(train_data_path, test_data_path)
# client_ids, group_ids, train_data, test_data = load_ez_dataset(ez_data_path)
print("client_ids shape: ", len(client_ids))
print("group_ids shape: ", len(group_ids))
print("train_data shape: ", len(train_data))
print("test_data shape: ", len(test_data))
# Create clients
# Parameters: seed, lr, num_classes
seed = 123
lr = 0.01
num_classes = 20
client_model = CNNClientModel(seed=seed, lr=lr, num_classes=num_classes)

_users = client_ids
print("user 1 id: ", _users[0])
print("user 1 train data: ", train_data[_users[0]])
groups = [[] for _ in _users]
clients = [Client(u, g, train_data[u], test_data[u], client_model) for u, g in zip(_users, groups)]
print("clients shape: ", len(clients))

# Create Server
aggregator = FedAvgAggregator()
# Parameters: client_model, aggregator, clients_per_round, (optional: model_params)
clients_per_round = 10
server = BaseServer(client_model, aggregator)

# # Load Trainer
rounds = 100
eval_every = 10
epochs_per_round = 1
batch_size = 6
clients_per_round = 5
trainer = BaseTrainer(clients, num_classes, server, log_path=None)
trainer.begins(rounds, eval_every, epochs_per_round, batch_size, clients_per_round)