"""
Ablation Study: Component Contribution Analysis.
Systematically removes CUSPNet components (theory constraints, CUSP dynamics,
LLM appraisal) to quantify their individual contributions to performance.
"""
import numpy as np
from typing import Dict, List, Optional
from app.cuspnet.layer1_causal import CausalDiscoveryLayer
from app.cuspnet.utils import compute_resilience_reserve, compute_potential, find_fixed_points, classify_fixed_points, detect_positive_feedback_loops, compute_loop_strength
from app.cuspnet.layer3_llm import LazarusAppraisalChain
from app.data.loaders import SachsLoader, NHANESLoader, KossakowskiLoader, DAICWOZLoader
from app.experiments.baselines.ml_baselines import train_rf, train_xgb


ABLATION_CONFIGS = {
    "full": {
        "use_ebicglasso": True,
        "use_score": True,
        "use_theory_constraints": True,
        "use_cusp_dynamics": True,
        "use_a_embedding": True,
        "use_llm": True,
        "use_lazarus_reflection": True,
    },
    "no_score": {
        "use_ebicglasso": True,
        "use_score": False,
        "use_theory_constraints": True,
        "use_cusp_dynamics": True,
        "use_a_embedding": True,
        "use_llm": True,
        "use_lazarus_reflection": True,
    },
    "no_theory": {
        "use_ebicglasso": True,
        "use_score": True,
        "use_theory_constraints": False,
        "use_cusp_dynamics": True,
        "use_a_embedding": True,
        "use_llm": True,
        "use_lazarus_reflection": True,
    },
    "no_cusp": {
        "use_ebicglasso": True,
        "use_score": True,
        "use_theory_constraints": True,
        "use_cusp_dynamics": False,
        "use_a_embedding": True,
        "use_llm": True,
        "use_lazarus_reflection": True,
    },
    "no_a_embedding": {
        "use_ebicglasso": True,
        "use_score": True,
        "use_theory_constraints": True,
        "use_cusp_dynamics": True,
        "use_a_embedding": False,
        "use_llm": True,
        "use_lazarus_reflection": True,
    },
    "no_llm": {
        "use_ebicglasso": True,
        "use_score": True,
        "use_theory_constraints": True,
        "use_cusp_dynamics": True,
        "use_a_embedding": True,
        "use_llm": False,
        "use_lazarus_reflection": False,
    },
    "no_lazarus": {
        "use_ebicglasso": True,
        "use_score": True,
        "use_theory_constraints": True,
        "use_cusp_dynamics": True,
        "use_a_embedding": True,
        "use_llm": True,
        "use_lazarus_reflection": False,
    },
}


def _run_ablation_causal(
    X: np.ndarray,
    var_names: List[str],
    config: Dict,
    adj_true: Optional[np.ndarray] = None,
) -> Dict:
    result = {}
    cached_layer_result = None

    if not config["use_score"]:
        p = X.shape[1]
        topo_order = list(range(p))
        result["topological_order"] = topo_order
    else:
        layer = CausalDiscoveryLayer(ebic_gamma=0.5, score_threshold=0.01)
        cached_layer_result = layer.fit(X, var_names)
        result["topological_order"] = cached_layer_result["topological_order"]

    if not config["use_theory_constraints"]:
        p = X.shape[1]
        adj = np.zeros((p, p))
        partial_corr = np.corrcoef(X.T)
        np.fill_diagonal(partial_corr, 0)
        for i in range(p):
            for j in range(p):
                if i != j and abs(partial_corr[i, j]) > 0.1:
                    if result["topological_order"].index(i) < result["topological_order"].index(j):
                        adj[i, j] = partial_corr[i, j]
        result["causal_adjacency"] = adj
        result["partial_correlation"] = partial_corr
    else:
        if cached_layer_result is None:
            layer = CausalDiscoveryLayer(ebic_gamma=0.5, score_threshold=0.01)
            cached_layer_result = layer.fit(X, var_names)
        result["causal_adjacency"] = cached_layer_result["causal_adjacency"]
        result["partial_correlation"] = cached_layer_result["partial_correlation"]
        result["centrality_ranking"] = cached_layer_result["centrality_ranking"]
        result["bridge_symptoms"] = cached_layer_result["bridge_symptoms"]
        result["positive_feedback_loops"] = cached_layer_result["positive_feedback_loops"]

    if adj_true is not None:
        pred = result.get("causal_adjacency", np.zeros_like(adj_true))
        pred_flat = pred.flatten()
        true_flat = adj_true.flatten()
        tp = np.sum((pred_flat != 0) & (true_flat != 0))
        fp = np.sum((pred_flat != 0) & (true_flat == 0))
        fn = np.sum((pred_flat == 0) & (true_flat != 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        shd = int(np.sum(pred_flat != true_flat))
        result["metrics"] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "shd": shd,
        }

    return result


def _run_ablation_cusp(
    X: np.ndarray,
    config: Dict,
    causal_adjacency: Optional[np.ndarray] = None,
    window_size: int = 7,
    theta_bifurcation: float = 0.5,
) -> Dict:
    n = X.shape[0]
    result = {}

    if not config["use_cusp_dynamics"]:
        delta_v_series = np.ones(n) * 0.5
        result["delta_v_series"] = delta_v_series.tolist()
        result["delta_v_mean"] = 0.5
        result["delta_v_std"] = 0.0
        result["n_critical"] = 0
        result["model_type"] = "linear"
    else:
        delta_v_list = []
        for i in range(n):
            start = max(0, i - window_size + 1)
            window = X[start : i + 1]
            if window.shape[0] < 2:
                delta_v_list.append(0.5)
                continue
            stress_idx = 0
            resilience_idx = min(1, X.shape[1] - 1)
            social_idx = min(2, X.shape[1] - 1)
            pss_norm = np.mean(window[:, stress_idx])
            cdrisc_norm = np.mean(window[:, resilience_idx])
            mspss_norm = np.mean(window[:, social_idx])
            a = pss_norm - cdrisc_norm
            b = cdrisc_norm * mspss_norm - theta_bifurcation
            c = mspss_norm * 0.5
            if config["use_a_embedding"] and causal_adjacency is not None:
                coupling_effect = np.mean(np.abs(causal_adjacency))
                loop_coupling = 0.0
                loops = detect_positive_feedback_loops(causal_adjacency)
                if loops:
                    loop_coupling = np.mean([compute_loop_strength(causal_adjacency, lp) for lp in loops])
                a = a + 0.1 * coupling_effect
                b = b - 0.05 * loop_coupling
            dv = compute_resilience_reserve(a, b, c)
            delta_v_list.append(dv if np.isfinite(dv) else 0.0)
        delta_v_arr = np.array(delta_v_list)
        result["delta_v_series"] = delta_v_arr.tolist()
        result["delta_v_mean"] = float(np.mean(delta_v_arr))
        result["delta_v_std"] = float(np.std(delta_v_arr))
        result["n_critical"] = int(np.sum(delta_v_arr < 0.1))
        result["model_type"] = "cusp_with_embedding" if config["use_a_embedding"] else "cusp_no_embedding"

    return result


def _run_ablation_llm(
    interviews: List,
    phq8_scores: List[int],
    config: Dict,
    n_reflection_steps: int = 3,
    model_name: str = r"D:\Models\huggingface\Qwen3.5-2B",
) -> Dict:
    result = {}

    if not config["use_llm"]:
        result["method"] = "no_llm"
        result["n_appraisals"] = len(interviews)
        if phq8_scores:
            result["accuracy"] = 0.0
            result["n_samples"] = len(phq8_scores)
        return result

    llm = LazarusAppraisalChain(model_name=model_name)

    predicted_levels = []
    for interview in interviews:
        try:
            if not config["use_lazarus_reflection"]:
                primary = llm.primary_appraisal(interview)
                appraisal = {
                    "primary_appraisal": primary,
                    "secondary_appraisal": {"secondary_appraisal_score": 5},
                    "reappraisal": {},
                    "cognitive_distortions": [],
                }
            else:
                appraisal = llm.full_chain(interview)

            primary_score = appraisal.get("primary_appraisal", {}).get("primary_appraisal_score", 5)
            secondary_score = appraisal.get("secondary_appraisal", {}).get("secondary_appraisal_score", 5)
            distortions = appraisal.get("cognitive_distortions", [])
            n_distortions = len(distortions) if isinstance(distortions, list) else 0
            composite = (primary_score * 0.4 + (10 - secondary_score) * 0.3 + min(n_distortions, 5) * 0.6)
            if composite <= 3:
                predicted_levels.append("minimal")
            elif composite <= 5:
                predicted_levels.append("mild")
            elif composite <= 7:
                predicted_levels.append("moderate")
            elif composite <= 9:
                predicted_levels.append("moderately_severe")
            else:
                predicted_levels.append("severe")
        except Exception:
            predicted_levels.append("moderate")

    if phq8_scores:
        def _phq8_to_level(score: int) -> str:
            if score <= 4:
                return "minimal"
            elif score <= 9:
                return "mild"
            elif score <= 14:
                return "moderate"
            elif score <= 19:
                return "moderately_severe"
            else:
                return "severe"

        true_levels = [_phq8_to_level(s) for s in phq8_scores]
        n = min(len(predicted_levels), len(true_levels))
        correct = sum(1 for p, t in zip(predicted_levels[:n], true_levels[:n]) if p == t)
        accuracy = correct / n if n > 0 else 0.0
        result["accuracy"] = float(accuracy)
        result["n_samples"] = n
        result["method"] = "lazarus_reflection" if config["use_lazarus_reflection"] else "single_prompt"
    else:
        result["n_appraisals"] = len(predicted_levels)

    return result


def run_ablation(
    causal_dataset: str = "sachs",
    cusp_dataset: str = "studentlife",
    llm_dataset: str = "daic_woz",
    window_size: int = 7,
    n_reflection_steps: int = 3,
) -> Dict:
    results = {}

    if causal_dataset.lower() == "sachs":
        causal_loader = SachsLoader()
        X_causal, adj_true, var_names_causal = causal_loader.load()
        has_gt = True
    elif causal_dataset.lower() == "nhanes":
        causal_loader = NHANESLoader()
        X_causal, var_names_causal = causal_loader.load()
        adj_true = None
        has_gt = False
    else:
        raise ValueError(f"Unknown causal dataset: {causal_dataset}")

    if cusp_dataset.lower() == "kossakowski":
        raise FileNotFoundError(
            "Kossakowski dataset has been replaced with official StudentLife data. "
            "Please use cusp_dataset='studentlife' instead."
        )
    elif cusp_dataset.lower() == "studentlife":
        from app.data.loaders import StudentLifeLoader
        sl_loader = StudentLifeLoader()
        sl_data = sl_loader.load()
        phq9 = sl_data.get("phq9_series", {})
        stress = sl_data.get("stress_series", {})
        all_series = []
        for pid in phq9:
            phq_arr = np.array(phq9[pid])
            stress_arr = np.array(stress.get(pid, [[0.5]] * len(phq_arr)))
            if phq_arr.ndim == 1:
                phq_arr = phq_arr.reshape(-1, 1)
            if stress_arr.ndim == 1:
                stress_arr = stress_arr.reshape(-1, 1)
            phq_mean = np.mean(phq_arr, axis=1) if phq_arr.ndim > 1 else phq_arr
            stress_mean = np.mean(stress_arr, axis=1) if stress_arr.ndim > 1 else stress_arr
            n_rows = min(len(phq_mean), len(stress_mean))
            combined_pid = np.column_stack([
                stress_mean[:n_rows],
                1.0 - phq_mean[:n_rows] / (np.max(np.abs(phq_mean[:n_rows])) + 1e-10),
                np.ones(n_rows) * 0.5,
                (phq_mean[:n_rows] > np.median(phq_mean[:n_rows])).astype(float),
            ])
            all_series.append(combined_pid)
        if all_series:
            X_cusp = np.vstack(all_series)
        else:
            X_cusp = np.random.RandomState(42).randn(100, 4)
    else:
        X_cusp = np.random.RandomState(42).randn(100, 4)

    if llm_dataset.lower() == "daic_woz":
        llm_loader = DAICWOZLoader()
        llm_data = llm_loader.load()
        transcripts = llm_data.get("transcripts", {})
        labels = llm_data.get("labels", {})
        interviews = list(transcripts.values())
        phq8_scores = [labels.get(pid, 0) for pid in transcripts]
    else:
        interviews = []
        phq8_scores = []

    for config_name, config in ABLATION_CONFIGS.items():
        config_result = {}

        try:
            causal_result = _run_ablation_causal(
                X_causal, var_names_causal, config, adj_true if has_gt else None
            )
            config_result["causal"] = causal_result
        except Exception as e:
            config_result["causal"] = {"error": str(e)}

        try:
            causal_adj = config_result.get("causal", {}).get("causal_adjacency", None)
            cusp_result = _run_ablation_cusp(
                X_cusp, config,
                causal_adjacency=causal_adj if config["use_a_embedding"] else None,
                window_size=window_size,
            )
            config_result["cusp"] = cusp_result
        except Exception as e:
            config_result["cusp"] = {"error": str(e)}

        if interviews:
            try:
                llm_result = _run_ablation_llm(
                    interviews, phq8_scores, config, n_reflection_steps=n_reflection_steps
                )
                config_result["llm"] = llm_result
            except Exception as e:
                config_result["llm"] = {"error": str(e)}

        results[config_name] = config_result

    if "full" in results and "metrics" in results["full"].get("causal", {}):
        full_metrics = results["full"]["causal"]["metrics"]
        contributions = {}
        for config_name in ABLATION_CONFIGS:
            if config_name == "full":
                continue
            if "metrics" in results.get(config_name, {}).get("causal", {}):
                ablated_metrics = results[config_name]["causal"]["metrics"]
                contributions[config_name] = {
                    "f1_drop": full_metrics["f1"] - ablated_metrics["f1"],
                    "shd_increase": ablated_metrics["shd"] - full_metrics["shd"],
                    "precision_drop": full_metrics["precision"] - ablated_metrics["precision"],
                    "recall_drop": full_metrics["recall"] - ablated_metrics["recall"],
                }
        results["component_contributions"] = contributions

    if "full" in results and "cusp" in results["full"]:
        full_cusp = results["full"]["cusp"]
        cusp_contributions = {}
        for config_name in ABLATION_CONFIGS:
            if config_name == "full":
                continue
            if "cusp" in results.get(config_name, {}):
                ablated_cusp = results[config_name]["cusp"]
                if "error" not in ablated_cusp and "error" not in full_cusp:
                    cusp_contributions[config_name] = {
                        "delta_v_mean_change": ablated_cusp.get("delta_v_mean", 0) - full_cusp.get("delta_v_mean", 0),
                        "n_critical_change": ablated_cusp.get("n_critical", 0) - full_cusp.get("n_critical", 0),
                    }
        results["cusp_component_contributions"] = cusp_contributions

    results["ablation_configs"] = ABLATION_CONFIGS
    results["datasets"] = {
        "causal": causal_dataset,
        "cusp": cusp_dataset,
        "llm": llm_dataset,
    }

    return results
