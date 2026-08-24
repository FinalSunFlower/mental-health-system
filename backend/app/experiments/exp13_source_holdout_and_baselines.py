"""Independent source holdout and knowledge-guided baseline audit.

This module evaluates a frozen OmniPath source against the public Sachs
reference graph without using Sachs labels to select source edges.  Source
proposals are split before any data bootstrap: calibration labels are used only
by the RCEP gate and held-out proposals are used only at deployment.
"""
from __future__ import annotations

from typing import Dict, Sequence

import networkx as nx
import numpy as np

from app.cuspnet.contracts import RiskControlConfig, RiskControlledEvidenceGate
from app.cuspnet.projection import (
    CausalDiscoveryLayer,
    TheoryConstraint,
    TheoryConstraintEngine,
)
from app.data.external_knowledge import load_omnipath_sachs_prior
from app.data.sachs import SachsLoader
from app.experiments.exp7_risk_controlled_projection import _orient_shared_evidence
from app.experiments.graph_utils import edge_metrics


def _source_label(constraint: TheoryConstraint, names: Sequence[str], truth: np.ndarray) -> bool:
    index = {name: i for i, name in enumerate(names)}
    source, target = index[constraint.cause], index[constraint.effect]
    return bool(truth[source, target] != 0)


def _pc_background_projection(
    X: np.ndarray,
    names: Sequence[str],
    constraints: Sequence[TheoryConstraint],
    partial_corr: np.ndarray,
    threshold: float,
    shared_support: np.ndarray,
) -> tuple[np.ndarray, Dict]:
    """PC with required/forbidden source directions, then deterministic DAG completion."""
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    background = BackgroundKnowledge()
    for constraint in constraints:
        source = GraphNode(constraint.cause)
        target = GraphNode(constraint.effect)
        background.add_required_by_node(source, target)
        background.add_forbidden_by_node(target, source)
    result = pc(
        X,
        alpha=threshold,
        node_names=list(names),
        background_knowledge=background,
        show_progress=False,
    )
    graph = result.G.graph
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
    skeleton *= (shared_support > 0)
    directed *= (shared_support > 0)
    layer = CausalDiscoveryLayer(
        constraint_engine=TheoryConstraintEngine(theories=[]),
        score_threshold=threshold,
        orientation_solver="exact_order",
        orientation_weights={
            "partial_correlation": 1.0,
            "score_direction": 3.0,
            "theory_prior": 100.0,
            "topological_compatibility": 0.0,
        },
    )
    order = layer._extract_topo_from_adj(directed, p)
    output = layer._theory_constrained_orientation(
        partial_corr,
        order,
        list(names),
        directed,
        skeleton,
    )
    audit = dict(layer._last_orientation_audit)
    audit.update({
        "baseline": "pc_background_knowledge_shared_support_dag_completion",
        "pc_directed_edges": int(np.sum(directed != 0)),
        "pc_skeleton_edges": int(np.sum(np.triu(skeleton, 1))),
        "source_constraints_passed": len(constraints),
    })
    return output, audit


def _metrics(prediction: np.ndarray, truth: np.ndarray, support: np.ndarray) -> Dict:
    metrics = edge_metrics(prediction, truth)
    binary = (np.abs(prediction) > 1e-10).astype(int)
    metrics["support_violations"] = int(np.sum(binary * (support == 0)))
    metrics["is_dag"] = bool(nx.is_directed_acyclic_graph(nx.DiGraph(binary)))
    return metrics


def run_exp13(
    n_bootstrap: int = 60,
    calibration_fraction: float = 0.67,
    target_error: float = 0.50,
    score_threshold: float = 0.08,
    seed: int = 20260821,
) -> Dict:
    """Run source-held-out real-data baseline and safety evaluation."""
    if n_bootstrap < 8:
        raise ValueError("n_bootstrap must be at least 8")
    if not 0.5 <= calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in [0.5, 1)")
    X, truth, names = SachsLoader().load()
    constraints, metadata = load_omnipath_sachs_prior()
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(len(constraints))
    split = max(1, min(len(constraints) - 1, int(round(
        calibration_fraction * len(constraints)
    ))))
    calibration_indices = set(int(i) for i in permutation[:split])
    calibration = [c for i, c in enumerate(constraints) if i in calibration_indices]
    held_out = [c for i, c in enumerate(constraints) if i not in calibration_indices]
    calibration_labels = np.asarray([
        _source_label(c, names, truth) for c in calibration
    ], dtype=bool)
    gate = RiskControlledEvidenceGate(RiskControlConfig(
        target_error=target_error,
        failure_probability=0.05,
        candidate_thresholds=(0.0, 0.5, 0.65, 0.8, 0.9),
        min_accepted_calibration=max(3, len(calibration) // 3),
    )).fit([c.confidence for c in calibration], calibration_labels)

    rows = []
    for repeat in range(n_bootstrap):
        sample = X[rng.choice(len(X), size=len(X), replace=True)]
        standardized = (sample - sample.mean(axis=0)) / (sample.std(axis=0) + 1e-10)
        evidence = CausalDiscoveryLayer(
            constraint_engine=TheoryConstraintEngine(theories=[]),
            score_threshold=score_threshold,
            fixed_glasso_alpha=0.08,
            orientation_solver="exact_order",
        )
        _, partial_corr = evidence._ebic_glasso(standardized, names)
        topo, score_adj, score_skeleton = evidence._score_algorithm(standardized, names)
        support = np.maximum(
            score_skeleton,
            (np.abs(partial_corr) > score_threshold).astype(float),
        )
        np.fill_diagonal(support, 0)
        data_only, data_audit = _orient_shared_evidence(
            partial_corr, topo, score_adj, score_skeleton, names,
            [], "exact_order", score_threshold, 0.85,
        )
        soft_prior, soft_audit = _orient_shared_evidence(
            partial_corr, topo, score_adj, score_skeleton, names,
            constraints, "exact_order", score_threshold, 0.85,
        )
        pc_bk, pc_audit = _pc_background_projection(
            standardized, names, constraints, partial_corr, score_threshold, support,
        )
        rcep, rcep_audit = _orient_shared_evidence(
            partial_corr, topo, score_adj, score_skeleton, names,
            held_out, "exact_order", score_threshold, 0.85,
            risk_gate=gate, require_risk_certificate=True,
        )
        outputs = {
            "data_only": (data_only, data_audit),
            "soft_prior_uncalibrated": (soft_prior, soft_audit),
            "pc_background_knowledge": (pc_bk, pc_audit),
            "rcep_source_holdout": (rcep, rcep_audit),
        }
        evaluated = {
            name: {**_metrics(prediction, truth, support), "orientation_audit": audit}
            for name, (prediction, audit) in outputs.items()
        }
        accepted_test = [c for c in held_out if gate.accepts(c.confidence)]
        accepted_labels = [
            _source_label(c, names, truth) for c in accepted_test
        ]
        rows.append({
            "repeat": repeat,
            "methods": evaluated,
            "held_out_source": {
                "n_proposed": len(held_out),
                "n_accepted": len(accepted_test),
                "accepted_error_rate": (
                    float(1.0 - np.mean(accepted_labels)) if accepted_labels else None
                ),
            },
        })

    method_names = tuple(rows[0]["methods"])
    summary = {}
    for method in method_names:
        f1 = np.asarray([row["methods"][method]["f1"] for row in rows])
        base = np.asarray([row["methods"]["data_only"]["f1"] for row in rows])
        summary[method] = {
            "mean_f1": float(np.mean(f1)),
            "mean_delta_f1_vs_data": float(np.mean(f1 - base)),
            "percentile_interval_delta_f1": [
                float(np.percentile(f1 - base, 2.5)),
                float(np.percentile(f1 - base, 97.5)),
            ],
            "harm_fraction_vs_data": float(np.mean(f1 < base)),
            "dag_rate": float(np.mean([
                row["methods"][method]["is_dag"] for row in rows
            ])),
            "support_violations_max": int(max(
                row["methods"][method]["support_violations"] for row in rows
            )),
        }
    return {
        "protocol": {
            "dataset": "Sachs protein-signaling benchmark",
            "knowledge_source": "frozen OmniPath consensus-direction snapshot",
            "source_split": "fixed source-proposal split before data bootstrap",
            "calibration_fraction": calibration_fraction,
            "n_calibration_proposals": len(calibration),
            "n_held_out_proposals": len(held_out),
            "calibration_labels": "Sachs reference graph labels, never used to select source proposals",
            "deployment_labels": "held-out Sachs reference directions, used only for final evaluation",
            "target_error": target_error,
            "claim_boundary": (
                "This is a finite source-holdout safety audit. The small OmniPath/Sachs overlap "
                "cannot establish an i.i.d. source population; failure to certify must abstain."
            ),
        },
        "knowledge_metadata": metadata,
        "calibration_proposal_labels": [
            {"cause": c.cause, "effect": c.effect, "correct": bool(label)}
            for c, label in zip(calibration, calibration_labels)
        ],
        "gate_certificate": gate.certificate_,
        "summary": summary,
        "rows": rows,
    }
