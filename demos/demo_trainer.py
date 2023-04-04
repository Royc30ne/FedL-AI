from fedl.utils.dataset_utils import load_data
from fedl.Client.client_model.base_client_model import BaseClientModel
from fedl.Client.client_container import Client
from fedl.Server.Aggregator.fedavg import FedAvgAggregator
from fedl.Server.base_server import BaseServer
from fedl.Trainer.base_trainer import BaseTrainer

# Load dataset
train_data_path = 'data/20news/train'
test_data_path = 'data/20news/test'
client_ids, group_ids, train_data, test_data = load_data(train_data_path, test_data_path)

# Create clients
# Parameters: seed, lr, num_classes
seed = 123
lr = 0.01
num_classes = 20
client_model = BaseClientModel()

groups = [[] for _ in group_ids]
clients = [Client(c_id, g_id, train_data[c_id], test_data[c_id], client_model) for c_id, g_id in zip(client_ids, group_ids)]

# Create Server
aggregator = FedAvgAggregator()
# Parameters: client_model, aggregator, clients_per_round, (optional: model_params)
server = BaseServer(client_model, aggregator)

# Load Trainer
rounds = 100
trainer = BaseTrainer(server, clients, num_classes, rounds)
trainer.begins()