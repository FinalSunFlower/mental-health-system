"""Predeclared context-conditioned admission audit on frozen RPE1 sources."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.cuspnet.contracts import (
    ContextualRiskControlledEvidenceGate,
    RiskControlConfig,
)


ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw/causalbench"


def _absolute_control_correlation(
    labels: pd.DataFrame,
    gene_names: np.ndarray,
    control: np.ndarray,
) -> np.ndarray:
    """Compute a label-free, target-assay compatibility score in [0, 1]."""
    index = {str(name): position for position, name in enumerate(gene_names)}
    centered = control - np.mean(control, axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    scores = []
    for row in labels.itertuples(index=False):
        source_index = index[str(row.source)]
        target_index = index[str(row.target)]
        denominator = norms[source_index] * norms[target_index]
        correlation = (
            float(np.dot(centered[:, source_index], centered[:, target_index]) / denominator)
            if denominator > 0.0
            else 0.0
        )
        scores.append(abs(np.clip(correlation, -1.0, 1.0)))
    return np.asarray(scores, dtype=float)


def _run_source(
    label_path: Path,
    meta_path: Path,
    subset_path: Path,
    confidence_scale: float,
    seed: int,
) -> dict[str, Any]:
    labels = pd.read_csv(label_path)
    subset = np.load(subset_path, allow_pickle=False)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    source_confidence = 1.0 - np.exp(
        -np.maximum(labels["chip_weight"].to_numpy(dtype=float), 0.0) / confidence_scale
    )
    context = _absolute_control_correlation(
        labels,
        subset["gene_names"].astype(str),
        subset["control_expression"].astype(float),
    )

    groups = sorted(labels["source"].unique().tolist())
    shuffled = list(groups)
    np.random.RandomState(seed).shuffle(shuffled)
    n_calibration = max(1, min(len(groups) - 1, round(0.67 * len(groups))))
    calibration_groups = set(shuffled[:n_calibration])
    calibration_mask = labels["source"].isin(calibration_groups).to_numpy()
    held_out_mask = ~calibration_mask
    gate = ContextualRiskControlledEvidenceGate(
        RiskControlConfig(
            target_error=0.35,
            failure_probability=0.05,
            candidate_thresholds=(0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            min_accepted_calibration=max(20, int(np.sum(calibration_mask)) // 10),
        ),
        score_families=("source", "product", "geomean", "min"),
    ).fit(
        source_confidence[calibration_mask],
        context[calibration_mask],
        labels.loc[calibration_mask, "label"].astype(bool).to_numpy(),
    )
    if gate.certificate_["status"] == "certified":
        accepted = np.asarray([
            gate.accepts_context(source_score, context_score)
            for source_score, context_score in zip(
                source_confidence[held_out_mask], context[held_out_mask]
            )
        ])
    else:
        accepted = np.zeros(int(np.sum(held_out_mask)), dtype=bool)
    held_out_labels = labels.loc[held_out_mask, "label"].to_numpy(dtype=float)
    return {
        "protocol": {
            "source": metadata.get("knowledge_source", label_path.name),
            "context_score": "absolute Pearson correlation in target control expression",
            "context_score_uses_intervention_labels": False,
            "score_families_fixed_before_calibration": list(gate.score_families),
            "split": "TF-disjoint calibration/held-out split",
            "n_calibration": int(np.sum(calibration_mask)),
            "n_held_out": int(np.sum(held_out_mask)),
        },
        "gate_certificate": gate.certificate_,
        "held_out": {
            "accepted": int(np.sum(accepted)),
            "coverage": float(np.mean(accepted)),
            "support_precision": (
                float(np.mean(held_out_labels[accepted])) if np.any(accepted) else None
            ),
        },
    }


def run_exp21() -> dict[str, Any]:
    return {
        "hepg2_chip": _run_source(
            RAW / "rpe1_intervention_labels.csv",
            RAW / "rpe1_intervention_labels.meta.json",
            RAW / "rpe1_intervention_subset.npz",
            confidence_scale=500.0,
            seed=20260831,
        ),
        "omnipath": _run_source(
            RAW / "rpe1_omnipath_intervention_labels.csv",
            RAW / "rpe1_omnipath_intervention_labels.meta.json",
            RAW / "rpe1_omnipath_intervention_subset.npz",
            confidence_scale=1.0,
            seed=20260833,
        ),
    }

