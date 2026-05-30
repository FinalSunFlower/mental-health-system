import numpy as np
from typing import Dict, List, Optional


def plot_causal_dag(
    adjacency,
    variable_names: List[str],
    centrality_ranking: List[float],
    bridge_symptoms: Optional[List[str]] = None,
) -> Dict:
    bridge_set = set(bridge_symptoms) if bridge_symptoms else set()

    if hasattr(centrality_ranking, "tolist"):
        centrality_ranking = centrality_ranking.tolist()

    cent_arr = np.array(centrality_ranking[:len(variable_names)])
    cent_max = np.max(cent_arr) if np.max(cent_arr) > 1e-10 else 1.0
    cent_normalized = cent_arr / cent_max

    min_node_size = 10
    max_node_size = 40

    nodes = [
        {
            "id": i,
            "label": variable_names[i],
            "centrality": centrality_ranking[i] if i < len(centrality_ranking) else 0.0,
            "centrality_normalized": float(cent_normalized[i]),
            "size": min_node_size + (max_node_size - min_node_size) * float(cent_normalized[i]),
            "is_bridge": variable_names[i] in bridge_set,
            "color": "#e74c3c" if variable_names[i] in bridge_set else "#3498db",
        }
        for i in range(len(variable_names))
    ]

    edges = []
    p = len(variable_names)
    if hasattr(adjacency, "tolist"):
        adj_np = np.array(adjacency)
    else:
        adj_np = np.array(adjacency)

    edge_weights = []
    for i in range(min(p, adj_np.shape[0])):
        for j in range(min(p, adj_np.shape[1])):
            if abs(adj_np[i, j]) > 1e-10:
                edge_weights.append(abs(adj_np[i, j]))

    if edge_weights:
        ew_max = max(edge_weights)
        ew_min = min(edge_weights)
        ew_range = ew_max - ew_min if ew_max > ew_min else 1.0
    else:
        ew_range = 1.0
        ew_min = 0.0

    for i in range(min(p, adj_np.shape[0])):
        for j in range(min(p, adj_np.shape[1])):
            if abs(adj_np[i, j]) > 1e-10:
                w = abs(adj_np[i, j])
                normalized_w = (w - ew_min) / ew_range
                edges.append({
                    "source": i,
                    "target": j,
                    "weight": w,
                    "weight_normalized": float(normalized_w),
                    "sign": "positive" if adj_np[i, j] > 0 else "negative",
                })

    return {"nodes": nodes, "edges": edges}
