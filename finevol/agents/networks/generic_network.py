import torch.nn as nn
from ...device_utils import set_device
from torch.optim import Adam


class GenericNetwork(nn.Module):
    def __init__(self, device_id: int = 0):
        super().__init__()
        self.device = set_device(device_id)

    def __new__(cls, *args, **kwargs):
        if cls is GenericNetwork:
            raise TypeError(
                f"'{cls.__name__}' should not be directly instantiated. "
                + f"Try 'MLPNetwork' or 'LSTMNetwork' instead."
            )
        return object.__new__(cls)

    def create_optimizer(self, learning_rate):
        self.optimizer = Adam(self.parameters(), learning_rate)
