"""Risk-controlled, support-preserving auxiliary evidence benchmark.

The benchmark separates three objects that are often conflated: data support,
edge orientation, and external directional knowledge. A held-out calibration
sample certifies an acceptance threshold for expert proposals. Deployment
priors may then be correct, reversed, or entirely off-skeleton. All methods see
the same fitted data evidence in each replicate.
"""
from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Sequence, Tuple
from time import perf_counter

import networkx as nx
import numpy as np

from app.cuspnet.contracts import RiskControlConfig, RiskControlledEvidenceGate
from app.cuspnet.projection import (
    CausalDiscoveryLayer,
    TheoryConstraint,
    TheoryConstraintEngine,
)
from app.cuspnet.statistics import bootstrap_ci, paired_significance_test
from app.experiments.graph_utils import edge_metrics, sample_sparse_dag


def _generate_sem_data(
    adjacency: np.ndarray,
    order: Sequence[int],
    n_samples: int,
    family: str,
    noise_std: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generate observational data from a DAG under two mechanism families."""
    if family not in {"linear_gaussian", "nonlinear_additive"}:
        raise ValueError(f"unknown SEM family: {family}")
    p = adjacency.shape[0]
    values = np.zeros((n_samples, p), dtype=float)
    for node in order:
        parents = np.flatnonzero(adjacency[:, node] != 0)
        signal = np.zeros(n_samples, dtype=float)
        for parent in parents:
            parent_values = values[:, parent]
            weight = adjacency[parent, node]
            if family == "linear_gaussian":
                signal += weight * parent_values
            else:
                signal += weight * (
                    np.tanh(1.5 * parent_values)
                    + 0.15 * (parent_values ** 2 - 1.0)
                )
        values[:, node] = signal + rng.normal(0.0, noise_std, size=n_samples)
    return values


def _confidence_for_label(
    correct: bool,
    rng: np.random.RandomState,
    confidence_regime: str,
) -> float:
    """Sample informative or deliberately uninformative self-confidence."""
    if confidence_regime == "informative":
        return float(rng.beta(7.0, 2.5) if correct else rng.beta(2.5, 7.0))
    if confidence_regime == "uninformative":
        return float(rng.beta(3.0, 3.0))
    raise ValueError(f"unknown confidence regime: {confidence_regime}")


def _fit_gate(
    source_accuracy: float,
    n_calibration: int,
    target_error: float,
    seed: int,
    confidence_regime: str,
) -> RiskControlledEvidenceGate:
    scores, labels = _calibration_sample(
        source_accuracy, n_calibration, seed, confidence_regime
    )
    return RiskControlledEvidenceGate(
        RiskControlConfig(
            target_error=target_error,
            failure_probability=0.05,
            candidate_thresholds=tuple(np.linspace(0.0, 0.95, 20)),
            min_accepted_calibration=max(30, n_calibration // 10),
        )
    ).fit(scores, labels)


def _calibration_sample(
    source_accuracy: float,
    n_calibration: int,
    seed: int,
    confidence_regime: str,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    labels = rng.rand(n_calibration) < source_accuracy
    scores = np.array([
        _confidence_for_label(bool(label), rng, confidence_regime) for label in labels
    ])
    return scores, labels


class _EmpiricalThresholdGate:
    """Calibration baseline that omits finite-sample uncertainty control."""

    def __init__(self, threshold: float, certificate: Dict):
        self.threshold_ = float(threshold)
        self.certificate_ = certificate

    def filter_mask(self, confidence: Sequence[float]) -> np.ndarray:
        return np.asarray(confidence, dtype=float) >= self.threshold_

    def accepts(self, confidence: float) -> bool:
        return bool(float(confidence) >= self.threshold_)


def _fit_empirical_gate(
    source_accuracy: float,
    n_calibration: int,
    target_error: float,
    seed: int,
    confidence_regime: str,
) -> _EmpiricalThresholdGate:
    scores, labels = _calibration_sample(
        source_accuracy, n_calibration, seed, confidence_regime
    )
    thresholds = tuple(np.linspace(0.0, 0.95, 20))
    minimum = max(30, n_calibration // 10)
    diagnostics = []
    feasible = []
    for threshold in thresholds:
        accepted = scores >= threshold
        total = int(np.sum(accepted))
        errors = int(np.sum(~labels[accepted])) if total else 0
        empirical_error = errors / total if total else 1.0
        row = {
            "threshold": float(threshold),
            "accepted": total,
            "errors": errors,
            "empirical_error": float(empirical_error),
            "feasible": bool(total >= minimum and empirical_error <= target_error),
        }
        diagnostics.append(row)
        if row["feasible"]:
            feasible.append(row)
    if feasible:
        selected = min(feasible, key=lambda row: row["threshold"])
        threshold = selected["threshold"]
        status = "empirically_selected"
    else:
        selected = None
        threshold = 1.0 + np.finfo(float).eps
        status = "abstain_no_empirical_threshold"
    return _EmpiricalThresholdGate(threshold, {
        "status": status,
        "selected": selected,
        "target_error": target_error,
        "n_calibration": n_calibration,
        "threshold_diagnostics": diagnostics,
        "warning": "No finite-sample upper confidence bound or familywise guarantee.",
    })


def _deployment_priors(
    adjacency: np.ndarray,
    names: Sequence[str],
    coverage: float,
    direction_accuracy: float,
    spurious_ratio: float,
    rng: np.random.RandomState,
    confidence_regime: str,
) -> Tuple[List[TheoryConstraint], List[Dict]]:
    true_edges = [(int(i), int(j)) for i, j in zip(*np.where(adjacency != 0))]
    selected = [edge for edge in true_edges if rng.rand() < coverage]
    proposals: List[Tuple[int, int, str, bool]] = []
    for source, target in selected:
        correct = bool(rng.rand() < direction_accuracy)
        oriented = (source, target) if correct else (target, source)
        proposals.append((*oriented, "supported_direction", correct))

    adjacent_pairs = {
        tuple(sorted((int(i), int(j))))
        for i, j in true_edges
    }
    nonadjacent = [
        (i, j)
        for i in range(adjacency.shape[0])
        for j in range(i + 1, adjacency.shape[1])
        if (i, j) not in adjacent_pairs
    ]
    n_spurious = min(int(round(len(selected) * spurious_ratio)), len(nonadjacent))
    if n_spurious:
        chosen = rng.choice(len(nonadjacent), size=n_spurious, replace=False)
        for index in chosen:
            i, j = nonadjacent[int(index)]
            source, target = (i, j) if rng.rand() < 0.5 else (j, i)
            proposals.append((source, target, "unsupported_adjacency", False))

    constraints = []
    records = []
    for source, target, kind, correct in proposals:
        confidence = _confidence_for_label(correct, rng, confidence_regime)
        constraints.append(TheoryConstraint(
            names[source], names[target],
            theory="calibrated_external_source",
            confidence=confidence,
            citation="Synthetic deployment evidence",
        ))
        records.append({
            "source": source,
            "target": target,
            "kind": kind,
            "correct": correct,
            "confidence": confidence,
        })
    return constraints, records


def _orient_shared_evidence(
    partial_corr: np.ndarray,
    topo_order: Sequence[int],
    score_adj: np.ndarray,
    score_skeleton: np.ndarray,
    names: Sequence[str],
    constraints: Sequence[TheoryConstraint],
    solver: str,
    score_threshold: float,
    theory_weight: float,
    risk_gate: RiskControlledEvidenceGate | None = None,
    require_risk_certificate: bool = False,
    orientation_weights: Dict[str, float] | None = None,
) -> Tuple[np.ndarray, Dict]:
    weights = {"theory_prior": theory_weight, **(orientation_weights or {})}
    layer = CausalDiscoveryLayer(
        constraint_engine=TheoryConstraintEngine(
            theories=[], constraints=list(constraints), min_confidence=0.0,
            risk_gate=risk_gate,
            require_risk_certificate=require_risk_certificate,
        ),
        score_threshold=score_threshold,
        orientation_solver=solver,
        orientation_weights=weights,
        fixed_glasso_alpha=0.08,
    )
    adjacency = layer._theory_constrained_orientation(
        partial_corr,
        list(topo_order),
        list(names),
        score_adj,
        score_skeleton,
    )
    return adjacency, layer._last_orientation_audit


def _pc_backbone(
    X: np.ndarray,
    names: Sequence[str],
    alpha: float,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Return PC's directed marks and skeleton without any external knowledge."""
    from causallearn.search.ConstraintBased.PC import pc

    graph = pc(X, alpha=alpha, node_names=list(names), show_progress=False).G.graph
    p = len(names)
    directed = np.zeros((p, p), dtype=float)
    skeleton = np.zeros((p, p), dtype=float)
    for i in range(p):
        for j in range(i + 1, p):
            if graph[i, j] == 0 and graph[j, i] == 0:
                continue
            skeleton[i, j] = skeleton[j, i] = 1.0
            if graph[j, i] == 1 and graph[i, j] == -1:
                directed[i, j] = 1.0
            elif graph[i, j] == 1 and graph[j, i] == -1:
                directed[j, i] = 1.0
    layer = CausalDiscoveryLayer(constraint_engine=TheoryConstraintEngine(theories=[]))
    return layer._extract_topo_from_adj(directed, p), directed, skeleton


def _naive_prior_union(
    data_adjacency: np.ndarray,
    records: Sequence[Dict],
) -> np.ndarray:
    """Unrestricted baseline that lets every prior overwrite graph support."""
    output = np.array(data_adjacency, copy=True)
    for row in records:
        source, target = row["source"], row["target"]
        output[target, source] = 0.0
        weight = abs(output[source, target])
        output[source, target] = weight if weight > 1e-10 else 0.1
    return output


def _mean_interval(values: Iterable[float]) -> Dict:
    return bootstrap_ci(
        np.asarray(list(values), dtype=float),
        statistic_fn=np.mean,
        n_bootstrap=3000,
        seed=20260811,
    )


def _summarize(rows: Sequence[Dict], methods: Sequence[str]) -> Dict:
    summary = {}
    for method in methods:
        method_rows = [row["methods"][method] for row in rows]
        summary[method] = {
            "f1": _mean_interval(item["f1"] for item in method_rows),
            "shd": _mean_interval(item["shd"] for item in method_rows),
            "support_violations_mean": float(np.mean([
                item["support_violations"] for item in method_rows
            ])),
            "dag_rate": float(np.mean([item["is_dag"] for item in method_rows])),
        }
        objectives = [
            item["orientation_audit"]["fused_objective"]
            for item in method_rows
            if item.get("orientation_audit")
        ]
        if objectives:
            summary[method]["orientation_evidence_objective"] = _mean_interval(objectives)
    base = np.array([row["methods"]["data_only"]["f1"] for row in rows])
    for method in methods:
        if method == "data_only":
            continue
        values = np.array([row["methods"][method]["f1"] for row in rows])
        summary[method]["paired_delta_f1_vs_data"] = _mean_interval(values - base)
        summary[method]["harm_probability_vs_data"] = float(np.mean(values < base))
    return summary


def _clustered_method_inference(conditions: Sequence[Dict], methods: Sequence[str]) -> Dict:
    """Treat graph seed, rather than condition row, as the independent unit."""
    repeats = sorted({
        row["repeat"]
        for condition in conditions
        for row in condition["replicates"]
    })
    method_scores = {method: [] for method in methods}
    for repeat in repeats:
        repeat_rows = [
            row
            for condition in conditions
            for row in condition["replicates"]
            if row["repeat"] == repeat
        ]
        for method in methods:
            method_scores[method].append(float(np.mean([
                row["methods"][method]["f1"] for row in repeat_rows
            ])))

    data = np.asarray(method_scores["data_only"], dtype=float)
    inference = {}
    for method in methods:
        values = np.asarray(method_scores[method], dtype=float)
        delta = values - data
        record = {
            "cluster_unit": "graph seed averaged across the 24 corruption conditions",
            "n_clusters": len(repeats),
            "paired_delta_f1": _mean_interval(delta),
            "win_fraction": float(np.mean(delta > 1e-12)),
            "tie_fraction": float(np.mean(np.isclose(delta, 0.0, atol=1e-12))),
            "loss_fraction": float(np.mean(delta < -1e-12)),
        }
        if method != "data_only":
            record["paired_test_vs_data_only"] = paired_significance_test(
                values, data, test_type="wilcoxon"
            )
        inference[method] = record
    return inference


def run_exp7(
    n_repeats: int = 24,
    n_samples: int = 600,
    n_variables: int = 10,
    edge_probability: float = 0.18,
    noise_std: float = 0.8,
    prior_coverage: float = 0.70,
    direction_accuracies: Sequence[float] = (0.55, 0.75, 0.90),
    spurious_ratios: Sequence[float] = (0.0, 0.5),
    sem_families: Sequence[str] = ("linear_gaussian", "nonlinear_additive"),
    confidence_regimes: Sequence[str] = ("informative", "uninformative"),
    n_calibration: int = 600,
    target_prior_error: float = 0.20,
    score_threshold: float = 0.08,
    theory_weight: float = 0.85,
    topology_weight: float = 0.15,
    score_direction_weight: float = 0.5,
    fixed_glasso_alpha: float = 0.08,
    score_backbone: str = "ges_auto",
    seed: int = 20260811,
) -> Dict:
    """Run the pre-specified calibration and corruption stress matrix."""
    if n_repeats < 4:
        raise ValueError("n_repeats must be at least 4")
    methods = (
        "data_only",
        "uncalibrated_global_contract",
        "empirical_threshold_global_contract",
        "risk_controlled_global_contract",
        "risk_controlled_local_projection",
        "naive_unrestricted_union",
    )
    conditions = []
    for family, confidence_regime, accuracy, spurious_ratio in product(
        sem_families, confidence_regimes, direction_accuracies, spurious_ratios
    ):
        gate = _fit_gate(
            source_accuracy=accuracy / (1.0 + spurious_ratio),
            n_calibration=n_calibration,
            target_error=target_prior_error,
            seed=seed + int(accuracy * 1000) + int(spurious_ratio * 100),
            confidence_regime=confidence_regime,
        )
        empirical_gate = _fit_empirical_gate(
            source_accuracy=accuracy / (1.0 + spurious_ratio),
            n_calibration=n_calibration,
            target_error=target_prior_error,
            seed=seed + int(accuracy * 1000) + int(spurious_ratio * 100),
            confidence_regime=confidence_regime,
        )
        rows = []
        for repeat in range(n_repeats):
            replicate_seed = seed + repeat * 1009
            graph_rng = np.random.RandomState(replicate_seed)
            prior_rng = np.random.RandomState(
                replicate_seed + int(accuracy * 100) + int(spurious_ratio * 10) + 17
            )
            data_rng = np.random.RandomState(replicate_seed + 2)
            true_adjacency, names, order = sample_sparse_dag(
                n_variables, edge_probability, graph_rng
            )
            X = _generate_sem_data(
                true_adjacency, order, n_samples, family, noise_std, data_rng
            )

            if score_backbone not in {"ges_auto", "pc"}:
                raise ValueError("score_backbone must be 'ges_auto' or 'pc'")
            evidence_layer = CausalDiscoveryLayer(
                constraint_engine=TheoryConstraintEngine(theories=[]),
                score_threshold=score_threshold,
                fixed_glasso_alpha=fixed_glasso_alpha,
                orientation_solver="exact_order",
            )
            X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
            _, partial_corr = evidence_layer._ebic_glasso(X_std, list(names))
            if score_backbone == "pc":
                topo, score_adj, score_skeleton = _pc_backbone(
                    X_std, list(names), score_threshold
                )
            else:
                topo, score_adj, score_skeleton = evidence_layer._score_algorithm(
                    X_std, list(names)
                )
            constraints, prior_records = _deployment_priors(
                true_adjacency, names, prior_coverage, accuracy,
                spurious_ratio, prior_rng, confidence_regime,
            )
            accepted_mask = gate.filter_mask([
                constraint.confidence for constraint in constraints
            ]) if constraints else np.zeros(0, dtype=bool)
            accepted_records = [
                row for row, accepted in zip(prior_records, accepted_mask)
                if accepted
            ]
            data_only, data_audit = _orient_shared_evidence(
                partial_corr, topo, score_adj, score_skeleton, names,
                [], "exact_order", score_threshold, theory_weight,
                orientation_weights={
                    "topological_compatibility": topology_weight,
                    "score_direction": score_direction_weight,
                },
            )
            uncalibrated, uncalibrated_audit = _orient_shared_evidence(
                partial_corr, topo, score_adj, score_skeleton, names,
                constraints, "exact_order", score_threshold, theory_weight,
                orientation_weights={
                    "topological_compatibility": topology_weight,
                    "score_direction": score_direction_weight,
                },
            )
            empirical, empirical_audit = _orient_shared_evidence(
                partial_corr, topo, score_adj, score_skeleton, names,
                constraints, "exact_order", score_threshold, theory_weight,
                orientation_weights={
                    "topological_compatibility": topology_weight,
                    "score_direction": score_direction_weight,
                },
                risk_gate=empirical_gate,
            )
            controlled, controlled_audit = _orient_shared_evidence(
                partial_corr, topo, score_adj, score_skeleton, names,
                constraints, "exact_order", score_threshold, theory_weight,
                orientation_weights={
                    "topological_compatibility": topology_weight,
                    "score_direction": score_direction_weight,
                },
                risk_gate=gate,
            )
            local, local_audit = _orient_shared_evidence(
                partial_corr, topo, score_adj, score_skeleton, names,
                constraints, "score_sort", score_threshold, theory_weight,
                orientation_weights={
                    "topological_compatibility": topology_weight,
                    "score_direction": score_direction_weight,
                },
                risk_gate=gate,
            )
            naive = _naive_prior_union(data_only, prior_records)

            candidate_support = np.maximum(
                score_skeleton,
                (np.abs(partial_corr) > score_threshold).astype(float),
            )
            np.fill_diagonal(candidate_support, 0)
            method_outputs = {
                "data_only": (data_only, data_audit),
                "uncalibrated_global_contract": (uncalibrated, uncalibrated_audit),
                "empirical_threshold_global_contract": (empirical, empirical_audit),
                "risk_controlled_global_contract": (controlled, controlled_audit),
                "risk_controlled_local_projection": (local, local_audit),
                "naive_unrestricted_union": (naive, None),
            }
            evaluated = {}
            for method, (prediction, audit) in method_outputs.items():
                metrics = edge_metrics(prediction, true_adjacency)
                binary = (np.abs(prediction) > 1e-10).astype(int)
                metrics["support_violations"] = int(
                    np.sum(binary * (candidate_support == 0))
                )
                metrics["is_dag"] = bool(nx.is_directed_acyclic_graph(nx.DiGraph(binary)))
                metrics["orientation_audit"] = audit
                evaluated[method] = metrics

            accepted_correct = [row["correct"] for row in accepted_records]
            rows.append({
                "repeat": repeat,
                "seed": replicate_seed,
                "methods": evaluated,
                "prior": {
                    "n_proposed": len(prior_records),
                    "n_accepted": len(accepted_records),
                    "n_accepted_errors": int(sum(
                        not row["correct"] for row in accepted_records
                    )),
                    "accepted_error_rate": (
                        float(1.0 - np.mean(accepted_correct))
                        if accepted_correct else None
                    ),
                    "accepted_off_support": int(sum(
                        row["kind"] == "unsupported_adjacency"
                        for row in accepted_records
                    )),
                },
            })

        total_proposed = int(sum(row["prior"]["n_proposed"] for row in rows))
        total_accepted = int(sum(row["prior"]["n_accepted"] for row in rows))
        total_accepted_errors = int(sum(
            row["prior"]["n_accepted_errors"] for row in rows
        ))
        conditions.append({
            "sem_family": family,
            "confidence_regime": confidence_regime,
            "direction_accuracy": accuracy,
            "spurious_ratio": spurious_ratio,
            "risk_certificate": gate.certificate_,
            "empirical_threshold_selection": empirical_gate.certificate_,
            "deployment_gate_diagnostics": {
                "n_proposed": total_proposed,
                "n_accepted": total_accepted,
                "acceptance_rate": (
                    total_accepted / total_proposed if total_proposed else 0.0
                ),
                "n_accepted_errors": total_accepted_errors,
                "pooled_accepted_error": (
                    total_accepted_errors / total_accepted if total_accepted else None
                ),
            },
            "summary": _summarize(rows, methods),
            "replicates": rows,
        })

    controlled_effects = [
        condition["summary"]["risk_controlled_global_contract"]
        ["paired_delta_f1_vs_data"]["estimate"]
        for condition in conditions
    ]
    controlled_harm = [
        condition["summary"]["risk_controlled_global_contract"]
        ["harm_probability_vs_data"]
        for condition in conditions
    ]
    uncalibrated_effects = [
        condition["summary"]["uncalibrated_global_contract"]
        ["paired_delta_f1_vs_data"]["estimate"]
        for condition in conditions
    ]
    uncalibrated_harm = [
        condition["summary"]["uncalibrated_global_contract"]
        ["harm_probability_vs_data"]
        for condition in conditions
    ]
    empirical_effects = [
        condition["summary"]["empirical_threshold_global_contract"]
        ["paired_delta_f1_vs_data"]["estimate"]
        for condition in conditions
    ]
    empirical_harm = [
        condition["summary"]["empirical_threshold_global_contract"]
        ["harm_probability_vs_data"]
        for condition in conditions
    ]
    naive_effects = [
        condition["summary"]["naive_unrestricted_union"]
        ["paired_delta_f1_vs_data"]["estimate"]
        for condition in conditions
    ]
    certified = [
        condition for condition in conditions
        if condition["risk_certificate"]["status"] == "certified"
    ]
    certified_deployment_errors = [
        condition["deployment_gate_diagnostics"]["pooled_accepted_error"]
        for condition in certified
        if condition["deployment_gate_diagnostics"]["pooled_accepted_error"] is not None
    ]
    return {
        "method": "risk-controlled support-preserving global evidence projection",
        "pre_specified_protocol": {
            "n_repeats_per_condition": n_repeats,
            "n_samples": n_samples,
            "n_variables": n_variables,
            "sem_families": list(sem_families),
            "confidence_regimes": list(confidence_regimes),
            "direction_accuracies": list(direction_accuracies),
            "spurious_ratios": list(spurious_ratios),
            "prior_coverage": prior_coverage,
            "n_calibration": n_calibration,
            "target_prior_error": target_prior_error,
            "theory_weight": theory_weight,
            "topology_weight": topology_weight,
            "score_direction_weight": score_direction_weight,
            "fixed_glasso_alpha": fixed_glasso_alpha,
            "score_backbone": score_backbone,
            "primary_metric": "paired directed structural F1 versus shared-evidence data-only projection",
            "safety_metrics": [
                "support violations", "DAG rate", "observed harm rate versus data only"
            ],
        },
        "aggregate": {
            "mean_condition_delta_f1": float(np.mean(controlled_effects)),
            "worst_condition_delta_f1": float(np.min(controlled_effects)),
            "mean_condition_harm_probability": float(np.mean(controlled_harm)),
            "certified_conditions": len(certified),
            "abstained_conditions": len(conditions) - len(certified),
            "max_certified_deployment_pooled_error": (
                float(max(certified_deployment_errors))
                if certified_deployment_errors else None
            ),
            "uncalibrated_worst_condition_delta_f1": float(np.min(uncalibrated_effects)),
            "uncalibrated_worst_condition_harm_probability": float(np.max(uncalibrated_harm)),
            "empirical_threshold_worst_condition_delta_f1": float(np.min(empirical_effects)),
            "empirical_threshold_worst_condition_harm_probability": float(np.max(empirical_harm)),
            "naive_union_worst_condition_delta_f1": float(np.min(naive_effects)),
            "all_contract_support_violations_zero": bool(all(
                condition["summary"]["risk_controlled_global_contract"]
                ["support_violations_mean"] == 0.0
                for condition in conditions
            )),
        },
        "clustered_inference": _clustered_method_inference(conditions, methods),
        "conditions": conditions,
    }


def run_exp7_sensitivity(
    sample_sizes: Sequence[int] = (200, 600, 1200),
    variable_counts: Sequence[int] = (8, 10, 12),
    n_repeats: int = 12,
    seed: int = 20260812,
) -> Dict:
    """Evaluate the primary certified regime across data and graph scales."""
    settings = []
    for n_samples, n_variables in product(sample_sizes, variable_counts):
        started = perf_counter()
        result = run_exp7(
            n_repeats=n_repeats,
            n_samples=n_samples,
            n_variables=n_variables,
            direction_accuracies=(0.75,),
            spurious_ratios=(0.5,),
            sem_families=("linear_gaussian",),
            confidence_regimes=("informative",),
            seed=seed + n_samples * 17 + n_variables * 101,
        )
        condition = result["conditions"][0]
        settings.append({
            "n_samples": n_samples,
            "n_variables": n_variables,
            "elapsed_seconds": float(perf_counter() - started),
            "risk_certificate": condition["risk_certificate"],
            "deployment_gate_diagnostics": condition["deployment_gate_diagnostics"],
            "summary": condition["summary"],
        })
    deltas = [
        setting["summary"]["risk_controlled_global_contract"]
        ["paired_delta_f1_vs_data"]["estimate"]
        for setting in settings
    ]
    return {
        "protocol": {
            "sample_sizes": list(sample_sizes),
            "variable_counts": list(variable_counts),
            "n_repeats": n_repeats,
            "fixed_condition": {
                "sem_family": "linear_gaussian",
                "confidence_regime": "informative",
                "direction_accuracy": 0.75,
                "spurious_ratio": 0.5,
            },
        },
        "aggregate": {
            "mean_delta_f1": float(np.mean(deltas)),
            "worst_delta_f1": float(np.min(deltas)),
        },
        "settings": settings,
    }


def run_gate_calibration_sensitivity(
    calibration_sizes: Sequence[int] = (100, 300, 600, 1200),
    source_accuracies: Sequence[float] = (0.55, 0.75, 0.90),
    confidence_regimes: Sequence[str] = ("informative", "uninformative"),
    n_repeats: int = 200,
    n_deployment: int = 3000,
    target_error: float = 0.20,
    seed: int = 20260813,
) -> Dict:
    """Measure certificate rate and deployment behavior of the learned gate."""
    settings = []
    for n_calibration, accuracy, regime in product(
        calibration_sizes, source_accuracies, confidence_regimes
    ):
        certificate_count = 0
        coverages = []
        deployment_errors = []
        for repeat in range(n_repeats):
            repeat_seed = seed + repeat * 1009 + n_calibration + int(accuracy * 100)
            gate = _fit_gate(
                source_accuracy=accuracy,
                n_calibration=n_calibration,
                target_error=target_error,
                seed=repeat_seed,
                confidence_regime=regime,
            )
            if gate.certificate_["status"] != "certified":
                continue
            certificate_count += 1
            rng = np.random.RandomState(repeat_seed + 1)
            labels = rng.rand(n_deployment) < accuracy
            scores = np.array([
                _confidence_for_label(bool(label), rng, regime) for label in labels
            ])
            accepted = gate.filter_mask(scores)
            coverage = float(np.mean(accepted))
            coverages.append(coverage)
            if np.any(accepted):
                deployment_errors.append(float(np.mean(~labels[accepted])))
        settings.append({
            "n_calibration": n_calibration,
            "source_accuracy": accuracy,
            "confidence_regime": regime,
            "certificate_rate": certificate_count / n_repeats,
            "mean_deployment_coverage_when_certified": (
                float(np.mean(coverages)) if coverages else 0.0
            ),
            "mean_deployment_error_when_certified": (
                float(np.mean(deployment_errors)) if deployment_errors else None
            ),
            "fraction_certified_runs_above_target_error": (
                float(np.mean(np.asarray(deployment_errors) > target_error))
                if deployment_errors else None
            ),
        })
    return {
        "protocol": {
            "calibration_sizes": list(calibration_sizes),
            "source_accuracies": list(source_accuracies),
            "confidence_regimes": list(confidence_regimes),
            "n_repeats": n_repeats,
            "n_deployment": n_deployment,
            "target_error": target_error,
        },
        "settings": settings,
    }
