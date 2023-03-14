from abc import ABC, abstractmethod

class BasePoison(ABC):
    def __init__(self, model, dataset, device):
        self.model = model
        self.dataset = dataset
        self.device = device

    def poison(self, poisoned_client_id, poison_data):
        pass