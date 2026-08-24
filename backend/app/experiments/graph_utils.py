"""Small graph utilities shared by the RCEP experiments."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def edge_metrics(pred: np.ndarray, true: np.ndarray) -> Dict:
    pred_binary = (np.abs(pred) > 1e-10).astype(int)
    true_binary = (np.abs(true) > 1e-10).astype(int)
    np.fill_diagonal(pred_binary, 0)
    np.fill_diagonal(true_binary, 0)
    tp = int(np.sum((pred_binary == 1) & (true_binary == 1)))
    fp = int(np.sum((pred_binary == 1) & (true_binary == 0)))
    fn = int(np.sum((pred_binary == 0) & (true_binary == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "shd": int(np.sum(pred_binary != true_binary)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def sample_sparse_dag(
    n_variables: int,
    edge_probability: float,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Sample a weighted DAG from a random total order."""
    names = [f"V{i}" for i in range(n_variables)]
    order = list(rng.permutation(n_variables))
    adjacency = np.zeros((n_variables, n_variables), dtype=float)
    for source_rank, source in enumerate(order[:-1]):
        for target in order[source_rank + 1:]:
            if rng.rand() < edge_probability:
                adjacency[source, target] = rng.uniform(0.18, 0.45)
    if not np.any(adjacency):
        adjacency[order[0], order[1]] = 0.3
    return adjacency, names, order
