"""JOBS II randomized-design validation for permissioned graph orientation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from app.cuspnet.contracts import RiskControlConfig, RiskControlledEvidenceGate
from app.cuspnet.projection import (
    CausalDiscoveryLayer,
    TheoryConstraint,
    TheoryConstraintEngine,
)
from app.data.paths import RAW_DIR
from app.experiments.exp7_risk_controlled_projection import _orient_shared_evidence


PRE_TREATMENT = (
    "econ_hard",
    "depress1",
    "sex",
    "age",
    "nonwhite_num",
    "treat",
)
CALIBRATION_OUTCOMES = ("job_seek", "employed", "job_dich", "job_disc")
DEPLOYMENT_OUTCOMES = ("depress2", "comply")
GRAPH_VARIABLES = PRE_TREATMENT + ("job_seek",) + DEPLOYMENT_OUTCOMES + ("employed",)


def default_jobs_path() -> Path:
    return Path(RAW_DIR) / "psychology" / "jobs_ii.csv"


def _load_jobs(path: str | Path | None = None) -> pd.DataFrame:
    source = Path(path) if path is not None else default_jobs_path()
    if not source.exists():
        raise FileNotFoundError(
            f"JOBS II data not found at {source}. Run backend/download_jobs_ii.py."
        )
    frame = pd.read_csv(source)
    required = {
        "treat", "econ_hard", "depress1", "sex", "age", "nonwhite",
        "job_seek", "depress2", "work1", "comply", "job_dich", "job_disc",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"JOBS II file is missing columns: {missing}")
    frame = frame.copy()
    frame["nonwhite_num"] = frame["nonwhite"].astype(str).str.startswith("non.").astype(int)
    frame["employed"] = frame["work1"].astype(str).eq("psyemp").astype(int)
    if frame[list(GRAPH_VARIABLES)].isna().any().any():
        raise ValueError("JOBS II graph variables must not contain missing values")
    return frame


def _fit_design_gate() -> RiskControlledEvidenceGate:
    n_calibration = len(PRE_TREATMENT) * len(CALIBRATION_OUTCOMES)
    return RiskControlledEvidenceGate(
        RiskControlConfig(
            target_error=0.20,
            failure_probability=0.05,
            candidate_thresholds=(0.50, 0.60, 0.70, 0.80, 0.90),
            min_accepted_calibration=20,
        )
    ).fit(np.full(n_calibration, 0.90), np.ones(n_calibration, dtype=bool))


def _fit_reversed_design_gate() -> RiskControlledEvidenceGate:
    """Calibrate a source that proposes the design-impossible reverse direction."""
    n_calibration = len(PRE_TREATMENT) * len(CALIBRATION_OUTCOMES)
    return RiskControlledEvidenceGate(
        RiskControlConfig(
            target_error=0.20,
            failure_probability=0.05,
            candidate_thresholds=(0.50, 0.60, 0.70, 0.80, 0.90),
            min_accepted_calibration=20,
        )
    ).fit(np.full(n_calibration, 0.90), np.zeros(n_calibration, dtype=bool))


def run_exp9(
    n_bootstrap: int = 200,
    theory_weight: float = 0.85,
    score_threshold: float = 0.08,
    seed: int = 20260815,
    data_path: str | Path | None = None,
) -> Dict:
    """Measure correction of design-impossible reverse-time orientations."""
    frame = _load_jobs(data_path)
    values = frame[list(GRAPH_VARIABLES)].to_numpy(float)
    names = list(GRAPH_VARIABLES)
    name_to_index = {name: index for index, name in enumerate(names)}
    gate = _fit_design_gate()
    reversed_gate = _fit_reversed_design_gate()
    constraints = [
        TheoryConstraint(
            cause,
            effect,
            theory="JOBS II randomized and temporal design",
            confidence=0.90,
            citation="Vinokur and Schul (1997); JOBS II study design",
        )
        for cause in PRE_TREATMENT
        for effect in DEPLOYMENT_OUTCOMES
    ]
    reversed_constraints = [
        TheoryConstraint(
            effect,
            cause,
            theory="reversed JOBS II design stress source",
            confidence=0.90,
            citation="deliberately invalid reverse-time stress test",
        )
        for cause in PRE_TREATMENT
        for effect in DEPLOYMENT_OUTCOMES
    ]

    rng = np.random.RandomState(seed)
    rows = []
    for repeat in range(n_bootstrap):
        indices = rng.choice(len(values), size=len(values), replace=True)
        sample = values[indices]
        standardized = (sample - sample.mean(axis=0)) / (sample.std(axis=0) + 1e-10)
        evidence = CausalDiscoveryLayer(
            constraint_engine=TheoryConstraintEngine(theories=[]),
            score_threshold=score_threshold,
            fixed_glasso_alpha=0.08,
            orientation_solver="exact_order",
        )
        _, partial_corr = evidence._ebic_glasso(standardized, names)
        topo, score_adj, score_skeleton = evidence._score_algorithm(standardized, names)
        data_only, data_audit = _orient_shared_evidence(
            partial_corr, topo, score_adj, score_skeleton, names,
            [], "exact_order", score_threshold, theory_weight,
        )
        controlled, controlled_audit = _orient_shared_evidence(
            partial_corr, topo, score_adj, score_skeleton, names,
            constraints, "exact_order", score_threshold, theory_weight,
            risk_gate=gate,
        )
        reversed_source, reversed_source_audit = _orient_shared_evidence(
            partial_corr, topo, score_adj, score_skeleton, names,
            reversed_constraints, "exact_order", score_threshold, theory_weight,
            risk_gate=reversed_gate,
        )
        candidate = np.maximum(
            score_skeleton,
            (np.abs(partial_corr) > score_threshold).astype(float),
        )
        np.fill_diagonal(candidate, 0)
        supported = data_consistent = controlled_consistent = 0
        data_reversed = controlled_reversed = 0
        for cause in PRE_TREATMENT:
            for effect in DEPLOYMENT_OUTCOMES:
                i, j = name_to_index[cause], name_to_index[effect]
                if candidate[i, j] == 0:
                    continue
                supported += 1
                data_consistent += int(data_only[i, j] != 0)
                controlled_consistent += int(controlled[i, j] != 0)
                data_reversed += int(data_only[j, i] != 0)
                controlled_reversed += int(controlled[j, i] != 0)
        data_coverage = data_consistent / supported if supported else 0.0
        controlled_coverage = controlled_consistent / supported if supported else 0.0
        rows.append({
            "repeat": repeat,
            "supported_deployment_pairs": supported,
            "data_only_design_consistent_coverage": data_coverage,
            "risk_controlled_design_consistent_coverage": controlled_coverage,
            "delta_design_consistent_coverage": controlled_coverage - data_coverage,
            "data_only_reverse_time_rate": data_reversed / supported if supported else 0.0,
            "risk_controlled_reverse_time_rate": (
                controlled_reversed / supported if supported else 0.0
            ),
            "data_only_orientation_audit": data_audit,
            "risk_controlled_orientation_audit": controlled_audit,
            "reversed_source_fail_closed_equal_data": bool(
                np.array_equal(reversed_source, data_only)
            ),
            "reversed_source_orientation_audit": reversed_source_audit,
        })

    deltas = np.asarray([row["delta_design_consistent_coverage"] for row in rows])
    total_supported = sum(row["supported_deployment_pairs"] for row in rows)
    if total_supported == 0:
        raise RuntimeError("JOBS II validation produced no supported deployment pairs")
    data_weighted = sum(
        row["data_only_design_consistent_coverage"] * row["supported_deployment_pairs"]
        for row in rows
    ) / total_supported
    controlled_weighted = sum(
        row["risk_controlled_design_consistent_coverage"] * row["supported_deployment_pairs"]
        for row in rows
    ) / total_supported
    delta_interval = [
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
    ]
    data_reverse = sum(
        row["data_only_reverse_time_rate"] * row["supported_deployment_pairs"]
        for row in rows
    ) / total_supported
    controlled_reverse = sum(
        row["risk_controlled_reverse_time_rate"] * row["supported_deployment_pairs"]
        for row in rows
    ) / total_supported
    return {
        "protocol": {
            "dataset": "JOBS II randomized job-search intervention",
            "n_participants": len(frame),
            "calibration_targets": list(CALIBRATION_OUTCOMES),
            "deployment_targets": list(DEPLOYMENT_OUTCOMES),
            "group_disjoint_calibration": True,
            "uncertainty_interval": (
                "participant-bootstrap percentile interval over paired deltas"
            ),
            "verified_label": "pre-treatment/random assignment precedes post-treatment measurement",
            "claim_boundary": (
                "The reference verifies temporal and randomized-design direction only; "
                "it is not a complete causal graph and the illustrative subset must not "
                "be used to claim program efficacy."
            ),
        },
        "risk_certificate": gate.certificate_,
        "reversed_source_certificate": reversed_gate.certificate_,
        "summary": {
            "data_only_design_consistent_coverage": float(data_weighted),
            "risk_controlled_design_consistent_coverage": float(controlled_weighted),
            "mean_delta_design_consistent_coverage": float(np.mean(deltas)),
            "delta_design_consistent_coverage_interval": [
                delta_interval[0], delta_interval[1]
            ],
            "data_only_reverse_time_rate": float(data_reverse),
            "risk_controlled_reverse_time_rate": float(controlled_reverse),
            "positive_bootstrap_fraction": float(np.mean(deltas > 0)),
            "harm_bootstrap_fraction": float(np.mean(deltas < 0)),
            "maximum_support_violations": int(max(
                row["risk_controlled_orientation_audit"]["support_violations"]
                for row in rows
            )),
            "all_outputs_dag": bool(all(
                row["risk_controlled_orientation_audit"]["is_acyclic"]
                for row in rows
            )),
            "reversed_source_all_fail_closed_equal_data": bool(all(
                row["reversed_source_fail_closed_equal_data"] for row in rows
            )),
            "reversed_source_maximum_support_violations": int(max(
                row["reversed_source_orientation_audit"]["support_violations"]
                for row in rows
            )),
        },
        "replicates": rows,
    }
