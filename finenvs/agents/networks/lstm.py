import torch
import torch.nn as nn
from .generic_network import GenericNetwork


class LSTMNetwork(GenericNetwork):
    def __init__(
        self,
        shape: tuple,
        sequence_length: int,
        output_activation=nn.Identity,
        device_id: int = 0,
    ):
        super().__init__(device_id=device_id)
        self.create_layers(shape, sequence_length, output_activation)

    def create_layers(
        self, shape: tuple, sequence_length: int, output_activation
    ) -> None:
        self.sequence_length = sequence_length
        if len(shape) == 2:
            # just duplicate the last size as the output size
            shape = tuple([*shape, shape[-1]])
        assert (
            len(shape) == 3
        ), f"LSTM shape expected to have length 3, but shape with length {len(shape)} was given"
        (self.input_size, self.hidden_size, self.output_size) = shape
        # NOTE: can experiment here with bidirectional and dropout settings
        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            num_layers=1,
            batch_first=True,
            device=self.device,
        )
        self.last_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size, device=self.device),
            output_activation(),
        )

    def forward(self, lstm_inputs: torch.Tensor) -> torch.Tensor:
        input_shape = lstm_inputs.shape
        assert (
            len(input_shape) == 3 or len(input_shape) == 2
        ), f"LSTM inputs expected to have shape [batch_size (optional), seq_length, input_size]"
        assert (
            input_shape[-1] == self.input_size
        ), f"Last dimension of LSTM inputs expected to be input size ({self.input_size}) - received ({input_shape[-1]})"
        assert (
            input_shape[-2] == self.sequence_length
        ), f"Middle dimension of LSTM inputs expected to be sequence length ({self.sequence_length}) - received ({input_shape[-2]})"

        # FIXME: commenting this out for now
        """
        # if this is a single batch, we need an extra dimension...
        if len(input_shape) == 1:
            lstm_inputs = lstm_inputs.unsqueeze(0)
        # the sequence length is going to change depending on the environment!
        lstm_inputs = lstm_inputs.unsqueeze(1)
        """
        # NOTE: lstm_inputs has shape: (batch_size, sequence length, input_size)
        # in the case of TimeSeriesEnv, it would be something like (num_envs, 390, 4)
        lstm_outputs, _ = self.lstm.forward(lstm_inputs)
        # lstm_outputs has shape: (batch_size, sequence_length, hidden_size)
        final_hidden_states = lstm_outputs[:, -1, :]
        final_outputs = self.last_layer(final_hidden_states)
        return final_outputs
