"""Independent deployment audit for finite-sample source-admission gates."""
from __future__ import annotations

from itertools import product
from typing import Dict, Sequence

import numpy as np

from app.experiments.exp7_risk_controlled_projection import (
    _calibration_sample,
    _fit_empirical_gate,
    _fit_gate,
)


def _deployment_error(
    gate,
    source_accuracy: float,
    n_deployment: int,
    seed: int,
    confidence_regime: str,
) -> Dict:
    confidence, correct = _calibration_sample(
        source_accuracy, n_deployment, seed, confidence_regime
    )
    accepted = gate.filter_mask(confidence)
    n_accepted = int(np.sum(accepted))
    error = float(np.mean(~correct[accepted])) if n_accepted else None
    return {
        "n_accepted": n_accepted,
        "acceptance_rate": float(np.mean(accepted)),
        "accepted_error": error,
    }


def run_exp11(
    calibration_sizes: Sequence[int] = (60, 100, 300),
    source_accuracies: Sequence[float] = (0.55, 0.75, 0.90),
    confidence_regimes: Sequence[str] = ("informative", "uninformative"),
    n_repeats: int = 300,
    n_deployment: int = 3000,
    target_error: float = 0.20,
    seed: int = 20260817,
) -> Dict:
    """Compare certified and empirical gates on independent deployments."""
    settings = []
    for n_calibration, accuracy, regime in product(
        calibration_sizes, source_accuracies, confidence_regimes
    ):
        rows = []
        for repeat in range(n_repeats):
            base_seed = (
                seed + repeat * 1009 + n_calibration * 7
                + int(accuracy * 100) + (17 if regime == "informative" else 31)
            )
            certified = _fit_gate(
                accuracy, n_calibration, target_error, base_seed, regime
            )
            empirical = _fit_empirical_gate(
                accuracy, n_calibration, target_error, base_seed, regime
            )
            deployment_seed = base_seed + 499_979
            certified_result = _deployment_error(
                certified, accuracy, n_deployment, deployment_seed, regime
            )
            empirical_result = _deployment_error(
                empirical, accuracy, n_deployment, deployment_seed, regime
            )
            rows.append({
                "repeat": repeat,
                "certified_status": certified.certificate_["status"],
                "empirical_status": empirical.certificate_["status"],
                "certified": certified_result,
                "empirical": empirical_result,
            })

        def summarize(method: str) -> Dict:
            deployed = [row[method] for row in rows if row[method]["n_accepted"] > 0]
            violations = [
                row for row in deployed if row["accepted_error"] > target_error
            ]
            return {
                "selection_rate": float(np.mean([
                    row[f"{method}_status"] not in {
                        "abstain_no_certified_threshold",
                        "abstain_no_empirical_threshold",
                    }
                    for row in rows
                ])),
                "deployment_nonempty_rate": float(len(deployed) / len(rows)),
                "mean_acceptance_rate": float(np.mean([
                    row[method]["acceptance_rate"] for row in rows
                ])),
                "mean_accepted_error": float(np.mean([
                    row["accepted_error"] for row in deployed
                ])) if deployed else None,
                "target_violation_rate": float(len(violations) / len(rows)),
                "target_violation_rate_conditional_on_deployment": (
                    float(len(violations) / len(deployed)) if deployed else 0.0
                ),
            }

        settings.append({
            "n_calibration": n_calibration,
            "source_accuracy": accuracy,
            "confidence_regime": regime,
            "risk_controlled": summarize("certified"),
            "empirical_threshold": summarize("empirical"),
            "replicates": rows,
        })

    certified_violations = [
        item["risk_controlled"]["target_violation_rate"] for item in settings
    ]
    empirical_violations = [
        item["empirical_threshold"]["target_violation_rate"] for item in settings
    ]
    return {
        "protocol": {
            "independent_deployment_proposals": n_deployment,
            "n_repeats": n_repeats,
            "target_error": target_error,
            "failure_probability": 0.05,
            "comparison": (
                "RCEP's simultaneous Clopper-Pearson certificate versus the same "
                "threshold family selected by empirical error alone"
            ),
            "assumption": (
                "calibration and deployment proposals are independent draws "
                "from the same source distribution"
            ),
        },
        "aggregate": {
            "max_risk_controlled_target_violation_rate": float(max(certified_violations)),
            "max_empirical_target_violation_rate": float(max(empirical_violations)),
            "mean_risk_controlled_target_violation_rate": float(np.mean(certified_violations)),
            "mean_empirical_target_violation_rate": float(np.mean(empirical_violations)),
        },
        "settings": settings,
    }
