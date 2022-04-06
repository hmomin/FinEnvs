import torch.nn as nn
from ...device_utils import set_device
from torch.optim import Adam


class GenericNetwork(nn.Module):
    def __init__(self, device_id: int = 0):
        super().__init__()
        self.device = set_device(device_id)

    def create_optimizer(self, learning_rate):
        self.optimizer = Adam(self.parameters(), learning_rate)
