"""Finite-sample admission control for auxiliary directional evidence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class RiskControlConfig:
    """Finite-sample error control for accepting auxiliary proposals.

    Candidate thresholds are fixed before calibration. A Bonferroni-adjusted
    one-sided Clopper-Pearson bound makes threshold selection simultaneous over
    this finite family when calibration and deployment proposals are independent
    draws from the same source distribution.
    """

    target_error: float = 0.20
    failure_probability: float = 0.05
    candidate_thresholds: Sequence[float] = tuple(np.linspace(0.0, 0.95, 20))
    min_accepted_calibration: int = 20

    def __post_init__(self) -> None:
        if not 0.0 < self.target_error < 1.0:
            raise ValueError("target_error must be in (0, 1)")
        if not 0.0 < self.failure_probability < 1.0:
            raise ValueError("failure_probability must be in (0, 1)")
        thresholds = tuple(sorted(set(float(value) for value in self.candidate_thresholds)))
        if not thresholds or thresholds[0] < 0.0 or thresholds[-1] > 1.0:
            raise ValueError("candidate thresholds must lie in [0, 1]")
        if self.min_accepted_calibration < 1:
            raise ValueError("min_accepted_calibration must be positive")
        object.__setattr__(self, "candidate_thresholds", thresholds)


class RiskControlledEvidenceGate:
    """Learn an abstention threshold with a finite-sample error certificate."""

    def __init__(self, config: RiskControlConfig | None = None):
        self.config = config or RiskControlConfig()
        self.threshold_: float = 1.0
        self.certificate_: Dict[str, Any] | None = None

    @staticmethod
    def _upper_binomial_bound(errors: int, total: int, alpha: float) -> float:
        if total <= 0:
            return 1.0
        if errors >= total:
            return 1.0
        return float(beta.ppf(1.0 - alpha, errors + 1, total - errors))

    def fit(
        self,
        confidence: Sequence[float],
        correct: Sequence[bool | int],
    ) -> "RiskControlledEvidenceGate":
        scores = np.asarray(confidence, dtype=float)
        labels = np.asarray(correct, dtype=bool)
        if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels):
            raise ValueError("confidence and correct must be aligned one-dimensional arrays")
        if len(scores) == 0 or not np.isfinite(scores).all():
            raise ValueError("calibration confidence must be finite and non-empty")
        if np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("calibration confidence must lie in [0, 1]")

        family_size = len(self.config.candidate_thresholds)
        adjusted_alpha = self.config.failure_probability / family_size
        rows = []
        feasible = []
        for threshold in self.config.candidate_thresholds:
            accepted = scores >= threshold
            total = int(np.sum(accepted))
            errors = int(np.sum(~labels[accepted])) if total else 0
            empirical_error = errors / total if total else 1.0
            upper = self._upper_binomial_bound(errors, total, adjusted_alpha)
            row = {
                "threshold": threshold,
                "accepted": total,
                "errors": errors,
                "empirical_error": float(empirical_error),
                "simultaneous_error_upper": upper,
                "feasible": bool(
                    total >= self.config.min_accepted_calibration
                    and upper <= self.config.target_error
                ),
            }
            rows.append(row)
            if row["feasible"]:
                feasible.append(row)

        if feasible:
            selected = min(feasible, key=lambda row: row["threshold"])
            status = "certified"
            self.threshold_ = float(selected["threshold"])
        else:
            selected = {
                "threshold": 1.0,
                "accepted": int(np.sum(scores >= 1.0)),
                "errors": int(np.sum(~labels[scores >= 1.0])),
                "empirical_error": 1.0,
                "simultaneous_error_upper": 1.0,
                "feasible": False,
            }
            status = "abstain_no_certified_threshold"
            self.threshold_ = 1.0 + np.finfo(float).eps

        self.certificate_ = {
            "status": status,
            "selected": selected,
            "target_error": self.config.target_error,
            "failure_probability": self.config.failure_probability,
            "multiplicity_correction": "bonferroni_over_fixed_threshold_family",
            "adjusted_alpha": adjusted_alpha,
            "n_calibration": int(len(scores)),
            "threshold_diagnostics": rows,
            "assumption": (
                "independent calibration and deployment proposals drawn from "
                "the same source distribution"
            ),
        }
        return self

    def accepts(self, confidence: float) -> bool:
        if self.certificate_ is None:
            raise RuntimeError("fit must be called before accepts")
        return bool(float(confidence) >= self.threshold_)

    def filter_mask(self, confidence: Sequence[float]) -> np.ndarray:
        if self.certificate_ is None:
            raise RuntimeError("fit must be called before filter_mask")
        scores = np.asarray(confidence, dtype=float)
        return scores >= self.threshold_


class ContextualRiskControlledEvidenceGate(RiskControlledEvidenceGate):
    """Risk-controlled admission using source and task-context compatibility.

    Static database confidence is not assumed to be task confidence. This gate
    evaluates a finite, predeclared family of monotone score combinations and
    includes both score-family and threshold selection in the Bonferroni
    correction. The selected family is frozen after calibration.
    """

    _DEFAULT_SCORE_FAMILIES = ("source", "product", "geomean", "min")

    def __init__(
        self,
        config: RiskControlConfig | None = None,
        score_families: Sequence[str] = _DEFAULT_SCORE_FAMILIES,
    ):
        super().__init__(config)
        families = tuple(dict.fromkeys(str(value) for value in score_families))
        unknown = set(families) - set(self._DEFAULT_SCORE_FAMILIES)
        if not families or unknown:
            raise ValueError(f"unsupported contextual score families: {sorted(unknown)}")
        self.score_families = families
        self.score_family_: str | None = None

    @staticmethod
    def _combine(source: np.ndarray, context: np.ndarray, family: str) -> np.ndarray:
        if family == "source":
            return source
        if family == "product":
            return source * context
        if family == "geomean":
            return np.sqrt(source * context)
        if family == "min":
            return np.minimum(source, context)
        raise ValueError(f"unsupported contextual score family: {family}")

    def fit(
        self,
        source_confidence: Sequence[float],
        context_compatibility: Sequence[float],
        correct: Sequence[bool | int],
    ) -> "ContextualRiskControlledEvidenceGate":
        source = np.asarray(source_confidence, dtype=float)
        context = np.asarray(context_compatibility, dtype=float)
        labels = np.asarray(correct, dtype=bool)
        if (
            source.ndim != 1
            or context.ndim != 1
            or labels.ndim != 1
            or len(source) != len(context)
            or len(source) != len(labels)
        ):
            raise ValueError("contextual gate inputs must be aligned one-dimensional arrays")
        if len(source) == 0 or not np.isfinite(source).all() or not np.isfinite(context).all():
            raise ValueError("contextual calibration scores must be finite and non-empty")
        if np.any((source < 0.0) | (source > 1.0)) or np.any((context < 0.0) | (context > 1.0)):
            raise ValueError("contextual scores must lie in [0, 1]")

        family_size = len(self.score_families) * len(self.config.candidate_thresholds)
        adjusted_alpha = self.config.failure_probability / family_size
        rows = []
        feasible = []
        for family_index, family in enumerate(self.score_families):
            scores = self._combine(source, context, family)
            for threshold in self.config.candidate_thresholds:
                accepted = scores >= threshold
                total = int(np.sum(accepted))
                errors = int(np.sum(~labels[accepted])) if total else 0
                empirical_error = errors / total if total else 1.0
                upper = self._upper_binomial_bound(errors, total, adjusted_alpha)
                row = {
                    "score_family": family,
                    "threshold": threshold,
                    "accepted": total,
                    "errors": errors,
                    "empirical_error": float(empirical_error),
                    "simultaneous_error_upper": upper,
                    "feasible": bool(
                        total >= self.config.min_accepted_calibration
                        and upper <= self.config.target_error
                    ),
                }
                rows.append(row)
                if row["feasible"]:
                    feasible.append((-total, threshold, family_index, row))

        if feasible:
            _, _, _, selected = min(feasible)
            self.score_family_ = str(selected["score_family"])
            self.threshold_ = float(selected["threshold"])
            status = "certified"
        else:
            selected = {
                "score_family": None,
                "threshold": 1.0,
                "accepted": 0,
                "errors": 0,
                "empirical_error": 1.0,
                "simultaneous_error_upper": 1.0,
                "feasible": False,
            }
            self.score_family_ = None
            self.threshold_ = 1.0 + np.finfo(float).eps
            status = "abstain_no_certified_threshold"

        self.certificate_ = {
            "status": status,
            "selected": selected,
            "target_error": self.config.target_error,
            "failure_probability": self.config.failure_probability,
            "multiplicity_correction": "bonferroni_over_fixed_score_family_and_threshold_family",
            "adjusted_alpha": adjusted_alpha,
            "n_calibration": int(len(source)),
            "score_families": list(self.score_families),
            "threshold_diagnostics": rows,
            "assumption": (
                "independent calibration and deployment proposals with exchangeable "
                "source/context score pairs"
            ),
            "estimand": "accepted contextual proposal error",
        }
        return self

    def score(self, source_confidence: float, context_compatibility: float) -> float:
        if self.score_family_ is None:
            raise RuntimeError("fit must certify a contextual score family before scoring")
        source = np.asarray([source_confidence], dtype=float)
        context = np.asarray([context_compatibility], dtype=float)
        if (
            not np.isfinite(source).all()
            or not np.isfinite(context).all()
            or np.any((source < 0.0) | (source > 1.0))
            or np.any((context < 0.0) | (context > 1.0))
        ):
            raise ValueError("contextual scores must lie in [0, 1]")
        return float(self._combine(source, context, self.score_family_)[0])

    def accepts_context(self, source_confidence: float, context_compatibility: float) -> bool:
        return self.score(source_confidence, context_compatibility) >= self.threshold_


class StratifiedRiskControlledEvidenceGate(RiskControlledEvidenceGate):
    """Certify one threshold simultaneously across declared source strata.

    The contract controls the accepted proposal error separately in every
    calibration stratum. It is intended for deployments whose mixture over
    known strata may change while the conditional proposal distribution inside
    each stratum remains exchangeable with calibration.
    """

    def fit(
        self,
        confidence: Sequence[float],
        correct: Sequence[bool | int],
        strata: Sequence[str | int],
    ) -> "StratifiedRiskControlledEvidenceGate":
        scores = np.asarray(confidence, dtype=float)
        labels = np.asarray(correct, dtype=bool)
        groups = np.asarray(strata)
        if (
            scores.ndim != 1
            or labels.ndim != 1
            or groups.ndim != 1
            or len(scores) != len(labels)
            or len(scores) != len(groups)
        ):
            raise ValueError("confidence, correct, and strata must be aligned one-dimensional arrays")
        if len(scores) == 0 or not np.isfinite(scores).all():
            raise ValueError("calibration confidence must be finite and non-empty")
        if np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("calibration confidence must lie in [0, 1]")
        group_names = sorted({str(value) for value in groups.tolist()})
        if not group_names or any(not value for value in group_names):
            raise ValueError("strata must contain non-empty group labels")
        normalized_groups = np.asarray([str(value) for value in groups.tolist()])

        family_size = len(self.config.candidate_thresholds) * len(group_names)
        adjusted_alpha = self.config.failure_probability / family_size
        rows = []
        feasible = []
        for threshold in self.config.candidate_thresholds:
            accepted = scores >= threshold
            group_rows = []
            for group in group_names:
                selected = accepted & (normalized_groups == group)
                total = int(np.sum(selected))
                errors = int(np.sum(~labels[selected])) if total else 0
                empirical_error = errors / total if total else 1.0
                upper = self._upper_binomial_bound(errors, total, adjusted_alpha)
                group_rows.append({
                    "stratum": group,
                    "accepted": total,
                    "errors": errors,
                    "empirical_error": float(empirical_error),
                    "simultaneous_error_upper": upper,
                    "feasible": bool(
                        total >= self.config.min_accepted_calibration
                        and upper <= self.config.target_error
                    ),
                })
            row = {
                "threshold": threshold,
                "accepted": int(np.sum(accepted)),
                "worst_stratum_error_upper": float(max(item["simultaneous_error_upper"] for item in group_rows)),
                "feasible": bool(all(item["feasible"] for item in group_rows)),
                "strata": group_rows,
            }
            rows.append(row)
            if row["feasible"]:
                feasible.append(row)

        if feasible:
            selected = min(feasible, key=lambda row: row["threshold"])
            status = "certified"
            self.threshold_ = float(selected["threshold"])
        else:
            selected = {
                "threshold": 1.0,
                "accepted": 0,
                "worst_stratum_error_upper": 1.0,
                "feasible": False,
                "strata": [],
            }
            status = "abstain_no_certified_threshold"
            self.threshold_ = 1.0 + np.finfo(float).eps

        self.certificate_ = {
            "status": status,
            "selected": selected,
            "target_error": self.config.target_error,
            "failure_probability": self.config.failure_probability,
            "multiplicity_correction": "bonferroni_over_fixed_threshold_by_stratum_family",
            "adjusted_alpha": adjusted_alpha,
            "n_calibration": int(len(scores)),
            "n_strata": len(group_names),
            "strata": group_names,
            "threshold_diagnostics": rows,
            "assumption": (
                "independent calibration proposals within each declared stratum; "
                "conditional calibration/deployment exchangeability within strata; "
                "deployment uses only declared strata"
            ),
            "estimand": "worst-stratum accepted proposal error",
        }
        return self
