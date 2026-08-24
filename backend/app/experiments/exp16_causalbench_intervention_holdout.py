"""TF-disjoint CausalBench intervention audit.

The source is K562 ChIP-Atlas evidence; labels are independent intervention
responses extracted from the public Weissmann Perturb-seq H5AD.  The primary
claim is deliberately narrow: risk-controlled admission of *detectable
intervention support*.  A non-significant response is not treated as proof
that a biological edge is absent.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from causallearn.search.FCMBased.ANM.ANM import ANM
from scipy.stats import beta

from app.cuspnet.contracts import RiskControlConfig, RiskControlledEvidenceGate
from app.cuspnet.evidence_semantics import EvidenceSemantics, route_evidence

ROOT = Path(__file__).resolve().parents[3]
LABEL_PATH = ROOT / "data/raw/causalbench/k562_intervention_labels.csv"
META_PATH = ROOT / "data/raw/causalbench/k562_intervention_labels.meta.json"
SUBSET_PATH = ROOT / "data/raw/causalbench/k562_intervention_subset.npz"


def _sanitize_metadata_paths(metadata: dict[str, Any]) -> dict[str, Any]:
    """Avoid leaking local acquisition paths through frozen result metadata."""
    clean = dict(metadata)
    for key, value in clean.items():
        if key.endswith("_path") and isinstance(value, str):
            candidate = Path(value)
            try:
                clean[key] = candidate.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                clean[key] = candidate.name
    return clean


def _confidence(weight: float, scale: float = 500.0) -> float:
    # Fixed before looking at intervention labels; saturates only for very
    # strong ChIP evidence and remains strictly monotone.
    if scale <= 0.0:
        raise ValueError("confidence scale must be positive")
    return float(1.0 - np.exp(-max(float(weight), 0.0) / scale))


def _data_direction(
    control: np.ndarray,
    gene_names: np.ndarray,
    source: str,
    target: str,
    model: ANM,
) -> float:
    source_index = int(np.flatnonzero(gene_names == source)[0])
    target_index = int(np.flatnonzero(gene_names == target)[0])
    x = control[:, source_index].reshape(-1, 1)
    y = control[:, target_index].reshape(-1, 1)
    forward, reverse = model.cause_or_effect(x, y)
    return float(np.log(max(float(forward), 0.0) + 1e-12) - np.log(max(float(reverse), 0.0) + 1e-12))


def _bootstrap(values: np.ndarray, seed: int, n: int = 2000) -> list[float | None]:
    if len(values) == 0:
        return [None, None]
    rng = np.random.RandomState(seed)
    draws = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _clopper_pearson_interval(successes: int, total: int, confidence: float = 0.95) -> list[float | None]:
    """Exact two-sided interval for an observed detectable-response rate."""
    if total <= 0:
        return [None, None]
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if successes == 0 else float(beta.ppf(tail, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1.0 - tail, successes + 1, total - successes))
    return [lower, upper]


def _tf_cluster_bootstrap_precision(frame: pd.DataFrame, seed: int, n: int) -> dict[str, Any]:
    """Respect the TF-disjoint unit when quantifying held-out precision."""
    groups = sorted(frame["source"].astype(str).unique().tolist())
    if not groups:
        return {"n_tf_groups": 0, "support_precision_95": [None, None]}
    by_group = {group: frame[frame["source"].astype(str) == group] for group in groups}
    rng = np.random.RandomState(seed)
    values = []
    for _ in range(n):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        draw = pd.concat([by_group[str(group)] for group in sampled], ignore_index=True)
        values.append(float(draw["label"].mean()))
    return {
        "n_tf_groups": len(groups),
        "support_precision_95": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
    }


def run_exp16(
    calibration_fraction: float = 0.67,
    target_error: float = 0.35,
    seed: int = 20260827,
    n_bootstrap: int = 2000,
    label_path: Path = LABEL_PATH,
    meta_path: Path = META_PATH,
    subset_path: Path = SUBSET_PATH,
    dataset_name: str = "CausalBench / Weissmann K562 day-6 Perturb-seq",
    knowledge_source: str = "K562 ChIP-Atlas TF-target evidence",
    label_source: str = "independent perturbation expression response from the public H5AD",
    missing_hint: str = "run backend/download_causalbench_labels.py",
    confidence_scale: float = 500.0,
    evidence_semantics: EvidenceSemantics | None = None,
) -> dict[str, Any]:
    if not label_path.exists() or not subset_path.exists():
        raise FileNotFoundError(
            f"CausalBench derived labels missing; {missing_hint}"
        )
    labels = pd.read_csv(label_path)
    subset = np.load(subset_path, allow_pickle=False)
    gene_names = subset["gene_names"].astype(str)
    control = subset["control_expression"].astype(float)
    metadata = _sanitize_metadata_paths(json.loads(meta_path.read_text(encoding="utf-8")))
    semantics = evidence_semantics or EvidenceSemantics.detectable_response(
        metadata["label_rule"]
    )
    route = route_evidence(semantics)
    if route.route != "risk_controlled":
        raise ValueError(
            "intervention holdout requires complete detectable-response labels; "
            "route open-world sources through a diagnostic-only experiment"
        )
    if hashlib.sha256(label_path.read_bytes()).hexdigest() != metadata["csv_sha256"]:
        raise RuntimeError("CausalBench label hash does not match its metadata")
    if hashlib.sha256(subset_path.read_bytes()).hexdigest() != metadata["subset_sha256"]:
        raise RuntimeError("CausalBench expression subset hash does not match its metadata")

    labels["confidence"] = labels["chip_weight"].map(
        lambda weight: _confidence(weight, confidence_scale)
    )
    groups = sorted(labels["source"].unique().tolist())
    rng = np.random.RandomState(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    n_calibration_groups = max(1, min(len(groups) - 1, round(calibration_fraction * len(groups))))
    calibration_groups = set(shuffled[:n_calibration_groups])
    labels["partition"] = labels["source"].map(
        lambda value: "calibration" if value in calibration_groups else "held_out"
    )
    calibration = labels[labels["partition"] == "calibration"].copy()
    held_out = labels[labels["partition"] == "held_out"].copy()
    gate = RiskControlledEvidenceGate(RiskControlConfig(
        target_error=target_error,
        failure_probability=0.05,
        candidate_thresholds=(0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_accepted_calibration=max(20, len(calibration) // 10),
    )).fit(calibration["confidence"].to_numpy(), calibration["label"].astype(bool).to_numpy())

    positive_test = held_out[held_out["label"] == 1].copy()
    if len(positive_test) < 5:
        raise RuntimeError(f"held-out intervention-positive labels too few: {len(positive_test)}")

    # The manuscript metric is defined on intervention-confirmed held-out
    # directions.  Compute the observational baseline only on that locked
    # evaluation set; source-admission metrics below still use every candidate.
    model = ANM()
    positive_test = positive_test.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        positive_test["data_score"] = [
            _data_direction(control, gene_names, row.source, row.target, model)
            for row in positive_test.itertuples(index=False)
        ]
    positive_test["data_prediction"] = np.where(positive_test["data_score"] >= 0.0, 1, -1)
    positive_test["source_prediction"] = 1
    positive_test["accepted_source"] = positive_test["confidence"].map(gate.accepts)
    positive_test["rcep_prediction"] = np.where(
        positive_test["accepted_source"], 1, positive_test["data_prediction"]
    )
    labels["source_prediction"] = 1
    labels["accepted_source"] = labels["confidence"].map(gate.accepts)
    held_out = labels[labels["partition"] == "held_out"].copy()

    def directional_accuracy(frame: pd.DataFrame, column: str) -> float:
        return float(np.mean(frame[column].to_numpy() == 1)) if len(frame) else float("nan")

    def support_metrics(frame: pd.DataFrame, accepted_only: bool = False) -> dict[str, Any]:
        selected = frame[frame["accepted_source"]] if accepted_only else frame
        if len(selected) == 0:
            return {
                "n": 0,
                "positive_labels": 0,
                "coverage": 0.0,
                "support_precision": None,
                "support_precision_clopper_pearson_95": [None, None],
                "false_support_rate": None,
            }
        positives = int(selected["label"].sum())
        return {
            "n": len(selected),
            "positive_labels": positives,
            "coverage": float(len(selected) / len(frame)) if len(frame) else 0.0,
            "support_precision": float(selected["label"].mean()),
            "support_precision_clopper_pearson_95": _clopper_pearson_interval(
                positives, len(selected)
            ),
            "false_support_rate": float(1.0 - selected["label"].mean()),
        }

    method_accuracy = {
        "data_only": directional_accuracy(positive_test, "data_prediction"),
        "soft_prior": directional_accuracy(positive_test, "source_prediction"),
        "rcep": directional_accuracy(positive_test, "rcep_prediction"),
    }
    delta = method_accuracy["rcep"] - method_accuracy["data_only"]
    harm = float(np.mean(
        (positive_test["rcep_prediction"] != 1)
        & (positive_test["data_prediction"] == 1)
    ))
    delta_values = (
        (positive_test["rcep_prediction"].to_numpy() == 1).astype(float)
        - (positive_test["data_prediction"].to_numpy() == 1).astype(float)
    )

    sensitivity: dict[str, Any] = {}
    for error in (0.20, 0.35, 0.50):
        candidate = RiskControlledEvidenceGate(RiskControlConfig(
            target_error=error,
            failure_probability=0.05,
            candidate_thresholds=(0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            min_accepted_calibration=max(20, len(calibration) // 10),
        )).fit(calibration["confidence"], calibration["label"])
        accepted = held_out[held_out["confidence"].map(candidate.accepts)]
        positive = accepted[accepted["label"] == 1]
        sensitivity[str(error)] = {
            "status": candidate.certificate_["status"],
            "selected_threshold": candidate.threshold_,
            "held_out_support": support_metrics(held_out, accepted_only=True),
            "positive_directional_accuracy": directional_accuracy(positive.assign(source_prediction=1), "source_prediction") if len(positive) else None,
        }

    return {
        "protocol": {
            "dataset": dataset_name,
            "knowledge_source": knowledge_source,
            "label_source": label_source,
            "source_split": "TF-disjoint calibration/held-out split before gate fitting",
            "calibration_fraction": calibration_fraction,
            "n_calibration_sources": len(calibration_groups),
            "n_held_out_sources": len(groups) - len(calibration_groups),
            "n_calibration_candidates": len(calibration),
            "n_held_out_candidates": len(held_out),
            "n_held_out_positive_labels": len(positive_test),
            "label_rule": metadata["label_rule"],
            "evidence_semantics": semantics.kind,
            "evidence_label_definition": semantics.label_definition,
            "unknown_policy": (
                "unknown_is_not_causal_negative"
                if not semantics.unknown_is_error
                else "unknown_entries_are_ineligible_for_certificate"
            ),
            "certificate_estimand": "accepted_detectable_response_error",
            "evidence_route": route.route,
            "confidence_transform": (
                f"1 - exp(-source_weight / {confidence_scale:g})"
            ),
            "claim_boundary": (
                "intervention-confirmed detectable support and directional transfer; "
                "unknown responses are not treated as absent causal edges, so this is "
                "not a complete biological DAG benchmark"
            ),
        },
        "source_metadata": metadata,
        "gate_certificate": gate.certificate_,
        "split": {
            "calibration_sources": sorted(calibration_groups),
            "held_out_sources": sorted(set(groups) - calibration_groups),
        },
        "summary": {
            "directional_positive_labels": {
                "data_only_accuracy": method_accuracy["data_only"],
                "soft_prior_accuracy": method_accuracy["soft_prior"],
                "rcep_accuracy": method_accuracy["rcep"],
                "rcep_delta_vs_data": delta,
                "rcep_harm_fraction_vs_data": harm,
                "rcep_delta_bootstrap_95": _bootstrap(delta_values, seed + 1, n_bootstrap),
            },
            "held_out_support": support_metrics(held_out),
            "held_out_accepted_support": support_metrics(held_out, accepted_only=True),
            "held_out_tf_cluster_bootstrap": _tf_cluster_bootstrap_precision(
                held_out, seed + 2, n_bootstrap
            ),
        },
        "gate_target_error_sensitivity": sensitivity,
        "candidate_rows": labels.to_dict(orient="records"),
    }
