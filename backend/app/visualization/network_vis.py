import numpy as np
from typing import Dict, List, Optional


def plot_centrality_heatmap(
    adjacency,
    centrality_ranking: List[float],
    variable_names: Optional[List[str]] = None,
) -> Dict:
    labels = variable_names if variable_names else [f"V{i}" for i in range(len(centrality_ranking))]

    if hasattr(adjacency, "tolist"):
        adj_np = np.array(adjacency)
    else:
        adj_np = np.array(adjacency)

    p = len(labels)
    heatmap_matrix = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if i == j:
                heatmap_matrix[i, j] = centrality_ranking[i] if i < len(centrality_ranking) else 0.0
            else:
                heatmap_matrix[i, j] = abs(adj_np[i, j]) if i < adj_np.shape[0] and j < adj_np.shape[1] else 0.0

    max_val = np.max(heatmap_matrix) if np.max(heatmap_matrix) > 1e-10 else 1.0
    heatmap_normalized = (heatmap_matrix / max_val).tolist()

    centrality_arr = np.array(centrality_ranking[:p])
    cent_max = np.max(centrality_arr) if np.max(centrality_arr) > 1e-10 else 1.0
    centrality_normalized = (centrality_arr / cent_max).tolist()

    return {
        "matrix": adj_np.tolist(),
        "heatmap_normalized": heatmap_normalized,
        "labels": labels,
        "centrality_values": centrality_ranking,
        "centrality_normalized": centrality_normalized,
    }
