import torch
from typing import Tuple


def match_actions_dim_with_states(
    states: torch.Tensor, actions: torch.Tensor
) -> Tuple[torch.Tensor, int]:
    if len(actions.shape) < len(states.shape):
        sequence_length = states.shape[1]
        actions = actions.unsqueeze(1).repeat(1, sequence_length, 1)
        concat_dim = 2
    else:
        concat_dim = 1
    return (actions, concat_dim)
