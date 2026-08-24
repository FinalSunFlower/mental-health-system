"""Cross-study transfer of a calibrated randomized/temporal design source."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from app.cuspnet.projection import (
    CausalDiscoveryLayer,
    TheoryConstraint,
    TheoryConstraintEngine,
)
from app.data.paths import RAW_DIR
from app.experiments.exp7_risk_controlled_projection import _orient_shared_evidence
from app.experiments.exp9_psychology_design_validation import (
    CALIBRATION_OUTCOMES,
    PRE_TREATMENT,
    _fit_design_gate,
)


STAR_SOURCE_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/AER/STAR.csv"
STAR_SOURCE_SHA256 = "0e8b179ea3d883730b25008ca1293c4c8f886aa8d5d299c03ab6ff05d0123ae6"
JOBS_SOURCE_URL = (
    "https://vincentarelbundock.github.io/Rdatasets/csv/mediation/jobs.csv"
)
JOBS_SOURCE_SHA256 = "88d5cce12930f3d111597cf15b7c0cd3e7fa7f316f718de7024e147b9756fb9f"
STAR_VARIABLES = (
    "small_class",
    "readk", "mathk",
    "read1", "math1",
    "read2", "math2",
    "read3", "math3",
)
STAR_DESIGN_RELATIONS = tuple(
    [("small_class", outcome) for outcome in STAR_VARIABLES[1:]]
    + [
        (f"{subject}{earlier}", f"{subject}{later}")
        for subject in ("read", "math")
        for earlier_index, earlier in enumerate(("k", "1", "2", "3"))
        for later in ("k", "1", "2", "3")[earlier_index + 1:]
    ]
)
JOBS_CALIBRATION_RELATIONS = tuple(
    (cause, effect)
    for cause in PRE_TREATMENT
    for effect in CALIBRATION_OUTCOMES
)


def default_star_path() -> Path:
    return Path(RAW_DIR) / "psychology" / "star.csv"


def _load_star(path: str | Path | None = None) -> pd.DataFrame:
    source = Path(path) if path is not None else default_star_path()
    if not source.exists():
        raise FileNotFoundError(
            f"Project STAR data not found at {source}. Run backend/download_star.py."
        )
    frame = pd.read_csv(source)
    required = {"stark", *STAR_VARIABLES[1:]}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Project STAR file is missing columns: {missing}")
    frame = frame[["stark", *STAR_VARIABLES[1:]]].dropna().copy()
    frame["small_class"] = frame["stark"].astype(str).eq("small").astype(float)
    for name in STAR_VARIABLES[1:]:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    return frame[list(STAR_VARIABLES)]


def run_exp12(
    n_bootstrap: int = 200,
    theory_weight: float = 0.85,
    score_threshold: float = 0.08,
    seed: int = 20260818,
    data_path: str | Path | None = None,
) -> Dict:
    """Calibrate on JOBS II design labels and deploy on Project STAR."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    frame = _load_star(data_path)
    values = frame.to_numpy(float)
    names = list(STAR_VARIABLES)
    name_to_index = {name: index for index, name in enumerate(names)}
    gate = _fit_design_gate()
    constraints = [
        TheoryConstraint(
            cause,
            effect,
            theory="cross-study randomized/temporal design metadata",
            confidence=0.90,
            citation="Project STAR design; Stock and Watson (2007)",
        )
        for cause, effect in STAR_DESIGN_RELATIONS
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
        candidate = np.maximum(
            score_skeleton,
            (np.abs(partial_corr) > score_threshold).astype(float),
        )
        np.fill_diagonal(candidate, 0)

        supported = data_consistent = controlled_consistent = 0
        data_reversed = controlled_reversed = 0
        for cause, effect in STAR_DESIGN_RELATIONS:
            i, j = name_to_index[cause], name_to_index[effect]
            if candidate[i, j] == 0:
                continue
            supported += 1
            data_consistent += int(data_only[i, j] != 0)
            controlled_consistent += int(controlled[i, j] != 0)
            data_reversed += int(data_only[j, i] != 0)
            controlled_reversed += int(controlled[j, i] != 0)
        if supported == 0:
            raise RuntimeError("Project STAR produced no supported design relations")
        rows.append({
            "repeat": repeat,
            "supported_deployment_pairs": supported,
            "data_only_design_consistent_coverage": data_consistent / supported,
            "risk_controlled_design_consistent_coverage": controlled_consistent / supported,
            "delta_design_consistent_coverage": (
                controlled_consistent - data_consistent
            ) / supported,
            "data_only_reverse_time_rate": data_reversed / supported,
            "risk_controlled_reverse_time_rate": controlled_reversed / supported,
            "data_only_orientation_audit": data_audit,
            "risk_controlled_orientation_audit": controlled_audit,
        })

    deltas = np.asarray([row["delta_design_consistent_coverage"] for row in rows])
    delta_interval = [
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
    ]
    total_supported = sum(row["supported_deployment_pairs"] for row in rows)

    def weighted(metric: str) -> float:
        return float(sum(
            row[metric] * row["supported_deployment_pairs"] for row in rows
        ) / total_supported)

    return {
        "protocol": {
            "protocol_status": "predeclared_before_participant_bootstrap",
            "source_class": (
                "documented randomized-assignment and measurement-time direction metadata"
            ),
            "calibration_study": "JOBS II randomized job-search intervention",
            "deployment_study": "Project STAR randomized class-size experiment",
            "independent_studies": True,
            "independent_source_calibration_labels": True,
            "calibration_source_url": JOBS_SOURCE_URL,
            "calibration_source_sha256": JOBS_SOURCE_SHA256,
            "n_calibration_relations": len(JOBS_CALIBRATION_RELATIONS),
            "calibration_relations": [
                [cause, effect, 0.90, True]
                for cause, effect in JOBS_CALIBRATION_RELATIONS
            ],
            "calibration_label_definition": (
                "verified true when a baseline/randomized variable precedes a "
                "post-assignment measurement in the documented JOBS II design"
            ),
            "n_star_complete_cases": len(frame),
            "n_bootstrap": n_bootstrap,
            "uncertainty_interval": (
                "participant-bootstrap percentile interval over paired deltas"
            ),
            "deployment_variables": names,
            "n_predeclared_deployment_relations": len(STAR_DESIGN_RELATIONS),
            "deployment_relations": [list(pair) for pair in STAR_DESIGN_RELATIONS],
            "knowledge_source": "verified randomized-assignment and measurement-time metadata",
            "deployment_source_url": STAR_SOURCE_URL,
            "deployment_source_sha256": STAR_SOURCE_SHA256,
            "deployment_label_definition": (
                "verified randomized assignment or strict measurement-time precedence "
                "in the documented Project STAR design"
            ),
            "selection_policy": (
                "all 20 relations were fixed before bootstrap; evaluation retains only "
                "pairs present in the data-derived candidate support"
            ),
            "claim_boundary": (
                "The source verifies randomized-assignment/measurement order only. "
                "Coverage is evaluated only on data-supported pairs and does not "
                "estimate class-size effects or a complete causal graph."
            ),
        },
        "calibration_certificate": gate.certificate_,
        "summary": {
            "mean_supported_deployment_pairs": float(np.mean([
                row["supported_deployment_pairs"] for row in rows
            ])),
            "data_only_design_consistent_coverage": weighted(
                "data_only_design_consistent_coverage"
            ),
            "risk_controlled_design_consistent_coverage": weighted(
                "risk_controlled_design_consistent_coverage"
            ),
            "mean_delta_design_consistent_coverage": float(np.mean(deltas)),
            "delta_design_consistent_coverage_interval": [
                delta_interval[0], delta_interval[1]
            ],
            "positive_bootstrap_fraction": float(np.mean(deltas > 0)),
            "harm_bootstrap_fraction": float(np.mean(deltas < 0)),
            "data_only_reverse_time_rate": weighted("data_only_reverse_time_rate"),
            "risk_controlled_reverse_time_rate": weighted(
                "risk_controlled_reverse_time_rate"
            ),
            "maximum_support_violations": int(max(
                row["risk_controlled_orientation_audit"]["support_violations"]
                for row in rows
            )),
            "all_outputs_dag": bool(all(
                row["risk_controlled_orientation_audit"]["is_acyclic"]
                for row in rows
            )),
        },
        "replicates": rows,
    }
