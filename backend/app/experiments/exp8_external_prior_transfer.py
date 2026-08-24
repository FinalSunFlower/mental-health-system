"""External knowledge transfer diagnostic on Sachs protein signaling."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from app.cuspnet.projection import CausalDiscoveryLayer, TheoryConstraintEngine
from app.data.external_knowledge import load_omnipath_sachs_prior
from app.data.sachs import SachsLoader
from app.experiments.graph_utils import edge_metrics
from app.experiments.exp7_risk_controlled_projection import _orient_shared_evidence


def _prior_overlap(constraints, names: Sequence[str], truth: np.ndarray) -> Dict:
    index = {name: position for position, name in enumerate(names)}
    correct = reversed_direction = nonadjacent = 0
    for constraint in constraints:
        source, target = index[constraint.cause], index[constraint.effect]
        if truth[source, target] != 0:
            correct += 1
        elif truth[target, source] != 0:
            reversed_direction += 1
        else:
            nonadjacent += 1
    total = len(constraints)
    return {
        "n_constraints": total,
        "correct_reference_direction": correct,
        "reversed_reference_direction": reversed_direction,
        "nonadjacent_to_reference": nonadjacent,
        "reference_direction_precision": correct / total if total else 0.0,
        "note": "Computed only for post-hoc diagnostic reporting; not used to select constraints.",
    }


def run_exp8(
    n_bootstrap: int = 200,
    theory_weight: float = 0.85,
    score_threshold: float = 0.08,
    seed: int = 20260814,
) -> Dict:
    X, truth, names = SachsLoader().load()
    constraints, knowledge_metadata = load_omnipath_sachs_prior()
    rng = np.random.RandomState(seed)
    rows = []
    for repeat in range(n_bootstrap):
        indices = rng.choice(len(X), size=len(X), replace=True)
        sample = X[indices]
        X_std = (sample - sample.mean(axis=0)) / (sample.std(axis=0) + 1e-10)
        evidence = CausalDiscoveryLayer(
            constraint_engine=TheoryConstraintEngine(theories=[]),
            score_threshold=score_threshold,
            fixed_glasso_alpha=0.08,
            orientation_solver="exact_order",
        )
        _, partial_corr = evidence._ebic_glasso(X_std, names)
        topo, score_adj, score_skeleton = evidence._score_algorithm(X_std, names)
        data_only, _ = _orient_shared_evidence(
            partial_corr, topo, score_adj, score_skeleton, names,
            [], "exact_order", score_threshold, theory_weight,
        )
        external, audit = _orient_shared_evidence(
            partial_corr, topo, score_adj, score_skeleton, names,
            constraints, "exact_order", score_threshold, theory_weight,
        )
        fail_closed, fail_closed_audit = _orient_shared_evidence(
            partial_corr, topo, score_adj, score_skeleton, names,
            constraints, "exact_order", score_threshold, theory_weight,
            require_risk_certificate=True,
        )
        data_metrics = edge_metrics(data_only, truth)
        external_metrics = edge_metrics(external, truth)
        fail_closed_metrics = edge_metrics(fail_closed, truth)
        rows.append({
            "repeat": repeat,
            "data_only": data_metrics,
            "external_prior": external_metrics,
            "risk_controlled_no_certificate": fail_closed_metrics,
            "delta_f1": external_metrics["f1"] - data_metrics["f1"],
            "delta_shd": data_metrics["shd"] - external_metrics["shd"],
            "orientation_audit": audit,
            "fail_closed_orientation_audit": fail_closed_audit,
        })
    delta_f1 = np.asarray([row["delta_f1"] for row in rows])
    delta_shd = np.asarray([row["delta_shd"] for row in rows])
    return {
        "protocol": {
            "dataset": "Sachs protein signaling",
            "knowledge_source": "frozen OmniPath consensus-direction snapshot",
            "bootstrap_unit": "observational row; conditional uncertainty only",
            "n_bootstrap": n_bootstrap,
            "theory_weight": theory_weight,
            "claim_boundary": (
                "A real-source transfer diagnostic. OmniPath can contain overlapping "
                "literature, so this is not independent validation of causal knowledge."
            ),
            "risk_control_behavior": (
                "No source-specific calibration labels are available, so the "
                "risk-controlled method abstains and returns the data-only graph."
            ),
        },
        "knowledge_metadata": knowledge_metadata,
        "prior_reference_overlap": _prior_overlap(constraints, names, truth),
        "summary": {
            "mean_delta_f1": float(np.mean(delta_f1)),
            "delta_f1_percentile_interval": [
                float(np.percentile(delta_f1, 2.5)),
                float(np.percentile(delta_f1, 97.5)),
            ],
            "probability_delta_f1_positive": float(np.mean(delta_f1 > 0)),
            "mean_delta_shd": float(np.mean(delta_shd)),
            "fail_closed_delta_f1": float(np.mean([
                row["risk_controlled_no_certificate"]["f1"]
                - row["data_only"]["f1"] for row in rows
            ])),
            "support_violations": int(max(
                row["orientation_audit"]["support_violations"] for row in rows
            )),
        },
        "replicates": rows,
    }
