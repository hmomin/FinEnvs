import torch
import torch.nn as nn
from .generic_network import GenericNetwork


class LSTMNetwork(GenericNetwork):
    def __init__(
        self,
        shape: tuple,
        output_activation=nn.Identity,
        device_id: int = 0,
    ):
        super().__init__(device_id=device_id)
        self.create_layers(shape, output_activation)

    def create_layers(self, shape: tuple, output_activation) -> None:
        assert (
            len(shape) == 3
        ), f"LSTM shape expected have length 3, but shape with length {len(shape)} was given"
        (self.input_size, self.hidden_size, self.output_size) = shape
        self.lstm = nn.LSTM(
            self.input_size, self.hidden_size, 1, batch_first=True, device=self.device
        )
        self.last_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size, device=self.device),
            output_activation(),
        )

    def forward(self, lstm_inputs: torch.Tensor) -> torch.Tensor:
        # if this is a single batch, we need an extra dimension...
        if len(lstm_inputs.shape) == 1:
            lstm_inputs = torch.unsqueeze(lstm_inputs, dim=0)
        # now, account for the sequence length
        # FIXME: the sequence length is going to change depending on the environment!
        lstm_inputs = torch.unsqueeze(lstm_inputs, dim=1)
        lstm_outputs, _ = self.lstm.forward(lstm_inputs)
        # lstm_outputs has shape: (batch_size, sequence_length, hidden_size)
        final_hidden_states = lstm_outputs[:, -1, :]
        final_outputs = self.last_layer(final_hidden_states)
        return final_outputs
