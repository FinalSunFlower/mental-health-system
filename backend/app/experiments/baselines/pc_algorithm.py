import numpy as np
from typing import Optional


def run_pc(X: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    from causallearn.search.ConstraintBased.PC import pc

    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    cg = pc(X_std, alpha=alpha)
    G = cg.G
    p = X.shape[1]
    adj = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if i != j:
                if G.graph[j, i] == 1 and G.graph[i, j] == -1:
                    adj[i, j] = 1.0
    return adj
