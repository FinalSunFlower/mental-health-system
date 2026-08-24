"""Known-stratum mixture-shift audit for source admission."""
from __future__ import annotations

from typing import Any

import numpy as np

from app.cuspnet.contracts import (
    RiskControlConfig,
    RiskControlledEvidenceGate,
    StratifiedRiskControlledEvidenceGate,
)


class _EmpiricalGate:
    def __init__(self, config: RiskControlConfig):
        self.config = config
        self.threshold_: float = 1.0 + np.finfo(float).eps
        self.status_: str = "abstain_no_empirical_threshold"

    def fit(self, confidence: np.ndarray, correct: np.ndarray) -> "_EmpiricalGate":
        for threshold in self.config.candidate_thresholds:
            accepted = confidence >= threshold
            if int(np.sum(accepted)) < self.config.min_accepted_calibration:
                continue
            if float(np.mean(~correct[accepted])) <= self.config.target_error:
                self.threshold_ = float(threshold)
                self.status_ = "selected_without_certificate"
                break
        return self

    def filter_mask(self, confidence: np.ndarray) -> np.ndarray:
        return confidence >= self.threshold_


class _EmpiricalStratifiedGate(_EmpiricalGate):
    def fit(
        self,
        confidence: np.ndarray,
        correct: np.ndarray,
        strata: np.ndarray,
    ) -> "_EmpiricalStratifiedGate":
        groups = sorted(set(strata.tolist()))
        for threshold in self.config.candidate_thresholds:
            feasible = True
            for group in groups:
                accepted = (confidence >= threshold) & (strata == group)
                if int(np.sum(accepted)) < self.config.min_accepted_calibration:
                    feasible = False
                    break
                if float(np.mean(~correct[accepted])) > self.config.target_error:
                    feasible = False
                    break
            if feasible:
                self.threshold_ = float(threshold)
                self.status_ = "selected_without_certificate"
                break
        return self


def _error_probability(confidence: np.ndarray, stratum: np.ndarray) -> np.ndarray:
    clean = 0.03 + 0.06 * (1.0 - confidence)
    noisy = 0.05 + 0.45 * (1.0 - confidence)
    return np.where(stratum == "clean", clean, noisy)


def _interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def run_exp18(
    repetitions: int = 200,
    calibration_size: int = 20_000,
    deployment_size: int = 10_000,
    seed: int = 20260829,
) -> dict[str, Any]:
    """Compare pooled and worst-stratum admission under mixture shift."""
    if repetitions < 20:
        raise ValueError("repetitions must be at least 20")
    if calibration_size < 2_000 or deployment_size < 1_000:
        raise ValueError("sample sizes are too small for the declared audit")
    config = RiskControlConfig(
        target_error=0.20,
        failure_probability=0.05,
        candidate_thresholds=tuple(np.linspace(0.0, 0.9, 10)),
        min_accepted_calibration=max(100, calibration_size // 200),
    )
    methods = (
        "pooled_certificate",
        "empirical_pooled",
        "empirical_stratified",
        "stratified_certificate",
    )
    rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        rng = np.random.RandomState(seed + 104729 * repetition)
        calibration_stratum = np.where(
            rng.rand(calibration_size) < 0.90, "clean", "noisy"
        )
        calibration_confidence = rng.rand(calibration_size)
        calibration_error = _error_probability(
            calibration_confidence, calibration_stratum
        )
        calibration_correct = rng.rand(calibration_size) >= calibration_error

        pooled = RiskControlledEvidenceGate(config).fit(
            calibration_confidence, calibration_correct
        )
        stratified = StratifiedRiskControlledEvidenceGate(config).fit(
            calibration_confidence, calibration_correct, calibration_stratum
        )
        empirical = _EmpiricalGate(config).fit(
            calibration_confidence, calibration_correct
        )
        empirical_stratified = _EmpiricalStratifiedGate(config).fit(
            calibration_confidence, calibration_correct, calibration_stratum
        )
        gates = {
            "pooled_certificate": pooled,
            "stratified_certificate": stratified,
            "empirical_pooled": empirical,
            "empirical_stratified": empirical_stratified,
        }

        deployment_stratum = np.where(
            rng.rand(deployment_size) < 0.10, "clean", "noisy"
        )
        deployment_confidence = rng.rand(deployment_size)
        deployment_error = _error_probability(
            deployment_confidence, deployment_stratum
        )
        deployment_correct = rng.rand(deployment_size) >= deployment_error
        for name in methods:
            gate = gates[name]
            accepted = gate.filter_mask(deployment_confidence)
            if np.any(accepted):
                error = float(np.mean(~deployment_correct[accepted]))
                group_errors = {
                    group: float(np.mean(~deployment_correct[accepted & (deployment_stratum == group)]))
                    for group in ("clean", "noisy")
                    if np.any(accepted & (deployment_stratum == group))
                }
            else:
                error = 0.0
                group_errors = {}
            status = (
                gate.certificate_["status"]
                if isinstance(gate, RiskControlledEvidenceGate)
                else gate.status_
            )
            rows.append({
                "repetition": repetition,
                "method": name,
                "status": status,
                "threshold": float(gate.threshold_),
                "accepted_coverage": float(np.mean(accepted)),
                "deployment_error": error,
                "target_violation": bool(error > config.target_error),
                "stratum_error": group_errors,
            })

    summary: dict[str, Any] = {}
    for name in methods:
        selected = [row for row in rows if row["method"] == name]
        errors = np.asarray([row["deployment_error"] for row in selected])
        coverage = np.asarray([row["accepted_coverage"] for row in selected])
        thresholds = np.asarray([row["threshold"] for row in selected])
        summary[name] = {
            "certification_or_selection_rate": float(np.mean(thresholds <= 1.0)),
            "mean_selected_threshold": float(np.mean(thresholds[thresholds <= 1.0])) if np.any(thresholds <= 1.0) else None,
            "mean_deployment_error": float(np.mean(errors)),
            "deployment_error_95_interval": _interval(errors),
            "maximum_deployment_error": float(np.max(errors)),
            "target_violation_fraction": float(np.mean(errors > config.target_error)),
            "mean_accepted_coverage": float(np.mean(coverage)),
            "accepted_coverage_95_interval": _interval(coverage),
        }

    return {
        "protocol": {
            "question": "known-stratum robustness under calibration/deployment mixture shift",
            "calibration_mixture": {"clean": 0.90, "noisy": 0.10},
            "deployment_mixture": {"clean": 0.10, "noisy": 0.90},
            "conditional_error_model": {
                "clean": "0.03 + 0.06 * (1 - confidence)",
                "noisy": "0.05 + 0.45 * (1 - confidence)",
            },
            "calibration_size": calibration_size,
            "deployment_size": deployment_size,
            "repetitions": repetitions,
            "seed": seed,
            "target_error": config.target_error,
            "failure_probability": config.failure_probability,
            "thresholds": list(config.candidate_thresholds),
            "min_accepted_calibration_per_gate_or_stratum": config.min_accepted_calibration,
            "claim_boundary": (
                "the stratified guarantee covers mixture changes over declared strata only; "
                "it does not cover unseen strata or conditional shift within a stratum"
            ),
        },
        "summary": summary,
        "repetitions": rows,
    }
