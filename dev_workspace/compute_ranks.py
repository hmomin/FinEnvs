import numpy as np
import torch

torch.manual_seed(0)


def compute_ranks(x: torch.Tensor):
    ranks = torch.empty(x.shape[0])
    sort_indices = x.argsort()
    linear_ranks = torch.arange(0, x.shape[0], dtype=torch.float32)
    ranks[sort_indices] = linear_ranks
    return ranks


x = torch.randn(10)

print(x)

test_ranks = x.argsort()

print(test_ranks)

true_ranks = compute_ranks(x)

print(true_ranks)
