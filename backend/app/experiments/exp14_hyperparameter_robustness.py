"""Pre-specified factorial robustness audit for the RCEP core."""
from __future__ import annotations

from itertools import product
from typing import Dict, Sequence

import numpy as np

from app.experiments.exp7_risk_controlled_projection import run_exp7


def run_exp14(
    theory_weights: Sequence[float] = (0.35, 0.85, 1.35),
    support_thresholds: Sequence[float] = (0.05, 0.08, 0.12),
    glasso_alphas: Sequence[float] = (0.04, 0.12),
    topology_weights: Sequence[float] = (0.0, 0.15),
    backbones: Sequence[str] = ("ges_auto", "pc"),
    n_repeats: int = 6,
    seed: int = 20260822,
) -> Dict:
    """Sweep all declared choices under one calibrated synthetic regime.

    Each setting uses fresh, deterministic graph seeds. The output records both
    utility and the invariants needed to detect a superficially positive but
    unsafe configuration.
    """
    settings = []
    for index, (theory, threshold, alpha, topology, backbone) in enumerate(product(
        theory_weights,
        support_thresholds,
        glasso_alphas,
        topology_weights,
        backbones,
    )):
        result = run_exp7(
            n_repeats=n_repeats,
            n_samples=400,
            n_variables=8,
            direction_accuracies=(0.75,),
            spurious_ratios=(0.5,),
            sem_families=("linear_gaussian",),
            confidence_regimes=("informative",),
            n_calibration=600,
            score_threshold=threshold,
            theory_weight=theory,
            topology_weight=topology,
            fixed_glasso_alpha=alpha,
            score_backbone=backbone,
            seed=seed + index * 1009,
        )
        condition = result["conditions"][0]
        method = condition["summary"]["risk_controlled_global_contract"]
        settings.append({
            "theory_weight": theory,
            "support_threshold": threshold,
            "glasso_alpha": alpha,
            "topology_weight": topology,
            "backbone": backbone,
            "mean_delta_f1": method["paired_delta_f1_vs_data"]["estimate"],
            "harm_fraction": method["harm_probability_vs_data"],
            "support_violations_mean": method["support_violations_mean"],
            "dag_rate": method["dag_rate"],
            "certificate_status": condition["risk_certificate"]["status"],
        })
    deltas = np.asarray([row["mean_delta_f1"] for row in settings])
    return {
        "protocol": {
            "design": "full factorial over declared fixed choices",
            "n_settings": len(settings),
            "n_repeats_per_setting": n_repeats,
            "fixed_condition": {
                "n_samples": 400,
                "n_variables": 8,
                "source_accuracy": 0.75,
                "spurious_ratio": 0.5,
                "sem_family": "linear_gaussian",
                "confidence_regime": "informative",
            },
            "theory_weights": list(theory_weights),
            "support_thresholds": list(support_thresholds),
            "glasso_alphas": list(glasso_alphas),
            "topology_weights": list(topology_weights),
            "backbones": list(backbones),
        },
        "aggregate": {
            "mean_delta_f1": float(np.mean(deltas)),
            "worst_delta_f1": float(np.min(deltas)),
            "positive_settings_fraction": float(np.mean(deltas > 0)),
            "maximum_harm_fraction": float(max(row["harm_fraction"] for row in settings)),
            "all_support_violations_zero": bool(all(
                row["support_violations_mean"] == 0.0 for row in settings
            )),
            "all_outputs_dag": bool(all(row["dag_rate"] == 1.0 for row in settings)),
            "per_backbone_mean_delta_f1": {
                backbone: float(np.mean([
                    row["mean_delta_f1"] for row in settings if row["backbone"] == backbone
                ]))
                for backbone in backbones
            },
        },
        "settings": settings,
    }
