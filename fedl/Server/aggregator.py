import numpy as np

def fedavg(updates):
    return np.mean(updates, axis=0)