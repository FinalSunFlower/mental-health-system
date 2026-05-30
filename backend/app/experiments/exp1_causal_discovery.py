import os
import numpy as np
from typing import Dict, Optional, List
from app.cuspnet.layer1_causal import CausalDiscoveryLayer
from app.data.loaders import SachsLoader, NHANESLoader
from app.experiments.baselines.ml_baselines import run_pc, run_ges, run_notears


def _compute_edge_metrics(pred: np.ndarray, true: np.ndarray) -> Dict:
    p = pred.shape[0]
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    tp = np.sum((pred_flat != 0) & (true_flat != 0))
    fp = np.sum((pred_flat != 0) & (true_flat == 0))
    fn = np.sum((pred_flat == 0) & (true_flat != 0))
    tn = np.sum((pred_flat == 0) & (true_flat == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    shd = int(np.sum(pred_flat != true_flat))
    sid = 0
    for i in range(p):
        pred_parents = set(j for j in range(p) if pred[j, i] != 0)
        true_parents = set(j for j in range(p) if true[j, i] != 0)
        sid += len(pred_parents.symmetric_difference(true_parents))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "shd": shd,
        "sid": sid,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def run_exp1(
    dataset: str = "sachs",
    ebic_gamma: float = 0.5,
    score_threshold: float = 0.1,
    pc_alpha: float = 0.01,
    notears_lambda: float = 0.01,
) -> Dict:
    results = {}

    if dataset.lower() == "sachs":
        loader = SachsLoader()
        X, adj_true, var_names = loader.load()
        has_ground_truth = True
    elif dataset.lower() == "nhanes":
        loader = NHANESLoader()
        X, var_names = loader.load()
        adj_true = None
        has_ground_truth = False
    elif dataset.lower() == "osf_borsboom":
        import pandas as pd
        raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
        osf_file = os.path.join(raw_dir, "osf_borsboom_phq_gad.csv")
        if not os.path.exists(osf_file):
            alt_file = os.path.join(raw_dir, "osf_phq_gad.csv")
            if os.path.exists(alt_file):
                osf_file = alt_file
            else:
                raise FileNotFoundError(
                    f"OSF Borsboom dataset not found at {osf_file}. "
                    "Please download from https://osf.io/ (search Eiko Fried / network psychometrics) "
                    "and place as osf_borsboom_phq_gad.csv in app/data/raw/"
                )
        df = pd.read_csv(osf_file)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(thresh=max(1, len(df.columns) // 2))
        df = df.fillna(df.mean())
        X = df.values.astype(np.float64)
        var_names = list(df.columns)
        adj_true = None
        has_ground_truth = False
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'sachs', 'nhanes', or 'osf_borsboom'.")

    cuspnet_layer = CausalDiscoveryLayer(
        ebic_gamma=ebic_gamma, score_threshold=score_threshold
    )
    cuspnet_result = cuspnet_layer.fit(X, var_names)
    results["cuspnet"] = {
        "adjacency": cuspnet_result["causal_adjacency"],
        "partial_correlation": cuspnet_result["partial_correlation"],
        "topological_order": cuspnet_result["topological_order"],
        "centrality_ranking": cuspnet_result["centrality_ranking"],
        "bridge_symptoms": cuspnet_result["bridge_symptoms"],
        "positive_feedback_loops": cuspnet_result["positive_feedback_loops"],
    }
    if has_ground_truth:
        results["cuspnet"]["metrics"] = _compute_edge_metrics(
            cuspnet_result["causal_adjacency"], adj_true
        )

    baseline_methods = {
        "pc": lambda: run_pc(X, alpha=pc_alpha),
        "ges": lambda: run_ges(X),
        "notears": lambda: run_notears(X, lambda1=notears_lambda),
    }
    for method_name, method_fn in baseline_methods.items():
        try:
            adj_pred = method_fn()
            entry = {"adjacency": adj_pred}
            if has_ground_truth:
                entry["metrics"] = _compute_edge_metrics(adj_pred, adj_true)
            results[method_name] = entry
        except Exception as e:
            results[method_name] = {"error": str(e)}

    ebic_only_adj = np.zeros((X.shape[1], X.shape[1]))
    ebic_partial_corr = cuspnet_result["partial_correlation"]
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            if i != j and abs(ebic_partial_corr[i, j]) > 0.1:
                ebic_only_adj[i, j] = ebic_partial_corr[i, j]
    results["ebicglasso_only"] = {
        "adjacency": ebic_only_adj,
        "partial_correlation": ebic_partial_corr,
    }
    if has_ground_truth:
        results["ebicglasso_only"]["metrics"] = _compute_edge_metrics(
            ebic_only_adj, adj_true
        )

    score_only_adj = np.zeros((X.shape[1], X.shape[1]))
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    try:
        from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
        result = bic_exact_search(X_std, search_method="astar")
        p = X.shape[1]
        if isinstance(result, tuple) and len(result) >= 1:
            bic_adj = np.array(result[0])
            for i in range(p):
                for j in range(p):
                    if i != j and bic_adj[i, j] != 0:
                        score_only_adj[i, j] = 1.0
        elif hasattr(result, "G"):
            G = result.G
            for i in range(p):
                for j in range(p):
                    if i != j and G.graph[j, i] == 1 and G.graph[i, j] == -1:
                        score_only_adj[i, j] = 1.0
        partial_corr_full = np.corrcoef(X_std.T)
        np.fill_diagonal(partial_corr_full, 0)
        for i in range(p):
            for j in range(p):
                if score_only_adj[i, j] != 0 and abs(partial_corr_full[i, j]) > score_threshold:
                    score_only_adj[i, j] = partial_corr_full[i, j]
                elif score_only_adj[i, j] != 0:
                    score_only_adj[i, j] = 0.1
        results["score_only"] = {"adjacency": score_only_adj}
        if has_ground_truth:
            results["score_only"]["metrics"] = _compute_edge_metrics(score_only_adj, adj_true)
    except Exception as e:
        try:
            from causallearn.search.ScoreBased.GES import ges
            Record = ges(X_std)
            G = Record["G"]
            p = X.shape[1]
            for i in range(p):
                for j in range(p):
                    if i != j and G.graph[j, i] == 1 and G.graph[i, j] == -1:
                        score_only_adj[i, j] = 1.0
            results["score_only"] = {"adjacency": score_only_adj, "note": "GES fallback used"}
            if has_ground_truth:
                results["score_only"]["metrics"] = _compute_edge_metrics(score_only_adj, adj_true)
        except Exception as e2:
            results["score_only"] = {"error": str(e), "fallback_error": str(e2)}

    if has_ground_truth:
        comparison = {}
        for method_name in ["cuspnet", "pc", "ges", "notears", "ebicglasso_only", "score_only"]:
            if method_name in results and "metrics" in results[method_name]:
                comparison[method_name] = results[method_name]["metrics"]
        results["comparison"] = comparison

    results["dataset_info"] = {
        "name": dataset,
        "n_samples": X.shape[0],
        "n_variables": X.shape[1],
        "has_ground_truth": has_ground_truth,
        "variable_names": var_names,
    }

    return results
