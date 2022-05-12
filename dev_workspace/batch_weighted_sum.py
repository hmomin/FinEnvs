"""
CONCLUSION: batched_weighted_sum() in the OpenAI ES implementation (
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
)
is not doing anything special...
"""

import numpy as np
from numpy.linalg import norm
from typing import Tuple


def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)


def batched_weighted_sum(
    weights: np.ndarray, vecs: np.ndarray, batch_size: int
) -> Tuple[np.ndarray, int]:
    total = 0.0
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(
        itergroups(weights, batch_size), itergroups(vecs, batch_size)
    ):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(
            np.asarray(batch_weights, dtype=np.float32),
            np.asarray(batch_vecs, dtype=np.float32),
        )
        num_items_summed += len(batch_weights)
    return total, num_items_summed


if __name__ == "__main__":
    weights = np.random.randn(10000)
    vecs = np.random.randn(10000, 400)

    print(norm(weights))

    g1, count = batched_weighted_sum(
        weights,
        vecs,
        batch_size=500,
    )

    g2 = weights.dot(vecs)

    print(norm(g2 - g1))
    print(g1.shape)
    print(count)
