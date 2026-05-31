import numpy as np


def run_ges(X: np.ndarray) -> np.ndarray:
    from causallearn.search.ScoreBased.GES import ges

    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    Record = ges(X_std)
    G = Record["G"]
    p = X.shape[1]
    adj = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if i != j:
                if G.graph[j, i] == 1 and G.graph[i, j] == -1:
                    adj[i, j] = 1.0
    return adj
