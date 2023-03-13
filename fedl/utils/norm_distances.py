import numpy as np

from collections import defaultdict

# Create Euclidean distance dictonary for each client's weights
def create_euclidean_dict(users_weights):
    distances = defaultdict(dict)
    flatten_weights = np.array([np.concatenate([l.flatten().astype(np.float64) for l in w]) for w in users_weights], dtype=np.float64)
    for i in range(len(users_weights)):
        for j in range(i):
            distances[i][j] = distances[j][i] = np.linalg.norm(flatten_weights[i] - flatten_weights[j])
    return distances

# Create Manhattan distance dictonary for each client's weights
def create_manhattan_dict(users_weights):
    distances = defaultdict(dict)
    flatten_weights = np.array([np.concatenate([l.flatten().astype(np.float64) for l in w]) for w in users_weights], dtype=np.float64)
    for i in range(len(users_weights)):
        for j in range(i):
            distances[i][j] = distances[j][i] = np.linalg.norm(flatten_weights[i] - flatten_weights[j], ord=0)
    return distances