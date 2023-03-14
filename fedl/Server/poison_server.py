import numpy as np

from base_server import BaseServer
from aggregator import Aggregator

class PoisonServer(BaseServer):
    def __init__(self, client_model, aggregator : Aggregator, poisoner , model_params=None):
        super(PoisonServer, self).__init__(client_model, aggregator, model_params)
        self.poisoned_client_id = poisoned_client_id