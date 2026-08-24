"""Controlled validation for context-conditioned source admission."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from app.cuspnet.contracts import (
    ContextualRiskControlledEvidenceGate,
    RiskControlConfig,
    RiskControlledEvidenceGate,
)


def _draw_scores(rng: np.random.RandomState, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    correct = rng.rand(n) < 0.55
    source = rng.beta(5.0, 2.0, size=n)
    context = np.where(
        correct,
        rng.beta(8.0, 2.0, size=n),
        rng.beta(2.0, 8.0, size=n),
    )
    return source, context, correct


def run_exp22(
    repetitions: int = 200,
    n_calibration: int = 2_000,
    n_deployment: int = 3_000,
    target_error: float = 0.20,
    seed: int = 20260901,
    thresholds: Sequence[float] = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> dict[str, Any]:
    """Compare static and contextual gates under a known compatibility signal."""
    config = RiskControlConfig(
        target_error=target_error,
        failure_probability=0.05,
        candidate_thresholds=thresholds,
        min_accepted_calibration=max(50, n_calibration // 20),
    )
    methods = {
        "static_source": {"certified": 0, "coverage": [], "error": [], "violations": 0},
        "contextual": {"certified": 0, "coverage": [], "error": [], "violations": 0},
    }
    selected_families: dict[str, int] = {}
    for repeat in range(repetitions):
        rng = np.random.RandomState(seed + repeat * 1009)
        source_cal, context_cal, correct_cal = _draw_scores(rng, n_calibration)
        source_test, context_test, correct_test = _draw_scores(rng, n_deployment)

        static = RiskControlledEvidenceGate(config).fit(source_cal, correct_cal)
        contextual = ContextualRiskControlledEvidenceGate(config).fit(
            source_cal, context_cal, correct_cal
        )
        for name, gate, accepted in (
            ("static_source", static, static.filter_mask(source_test)),
            (
                "contextual",
                contextual,
                np.asarray([
                    contextual.accepts_context(source_score, context_score)
                    for source_score, context_score in zip(source_test, context_test)
                ])
                if contextual.certificate_["status"] == "certified"
                else np.zeros(n_deployment, dtype=bool),
            ),
        ):
            if gate.certificate_["status"] != "certified":
                continue
            methods[name]["certified"] += 1
            coverage = float(np.mean(accepted))
            error = float(np.mean(~correct_test[accepted])) if np.any(accepted) else 1.0
            methods[name]["coverage"].append(coverage)
            methods[name]["error"].append(error)
            methods[name]["violations"] += int(error > target_error)
        if contextual.score_family_ is not None:
            selected_families[contextual.score_family_] = (
                selected_families.get(contextual.score_family_, 0) + 1
            )

    summary = {}
    for name, values in methods.items():
        certified = int(values["certified"])
        summary[name] = {
            "certificate_rate": certified / repetitions,
            "mean_deployment_coverage_when_certified": (
                float(np.mean(values["coverage"])) if certified else 0.0
            ),
            "mean_deployment_error_when_certified": (
                float(np.mean(values["error"])) if certified else None
            ),
            "target_violation_fraction_when_certified": (
                values["violations"] / certified if certified else None
            ),
        }
    return {
        "protocol": {
            "repetitions": repetitions,
            "n_calibration": n_calibration,
            "n_deployment": n_deployment,
            "target_error": target_error,
            "source_score": "Beta(5,2), independent of correctness",
            "context_score": "Beta(8,2) if correct; Beta(2,8) otherwise",
            "score_families": list(ContextualRiskControlledEvidenceGate._DEFAULT_SCORE_FAMILIES),
        },
        "summary": summary,
        "contextual_selected_family_counts": selected_families,
    }

