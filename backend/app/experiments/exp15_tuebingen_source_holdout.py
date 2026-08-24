"""Real-pair source holdout audit on the Tuebingen cause-effect benchmark.

This experiment deliberately treats IGCI as a noisy auxiliary source, not as
ground-truth knowledge.  Pair directions are used only for the fixed
calibration/test split and final evaluation.  The split is by manually
declared benchmark family to avoid putting obvious replicated measurements in
both partitions.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from app.cuspnet.contracts import (
    RiskControlConfig,
    RiskControlledEvidenceGate,
    StratifiedRiskControlledEvidenceGate,
)
from app.data.paths import RAW_DIR
from causallearn.search.FCMBased.ANM.ANM import ANM


_PAIR_GROUPS = (
    (1, 4, "dwd_early"), (5, 11, "abalone"), (12, 12, "census"),
    (13, 16, "auto_mpg"), (17, 17, "census"), (18, 19, "single_early"),
    (20, 21, "dwd_late"), (22, 24, "arrhythmia"), (25, 32, "concrete"),
    (33, 37, "liver"), (38, 41, "pima"), (42, 42, "single_mid"),
    (43, 46, "ncep"), (47, 48, "single_traffic"), (49, 51, "bafu"),
    (56, 63, "undata_life"), (64, 64, "single_undata"),
    (65, 67, "yahoo"), (68, 72, "single_late"), (73, 76, "undata_macro"),
    (77, 77, "single_late"), (78, 80, "moffat"), (81, 83, "mahecha"),
    (84, 88, "single_final"), (89, 92, "solly"), (93, 93, "single_final"),
    (94, 96, "tarim"), (97, 98, "janzing_motion"), (99, 100, "r_datasets"),
    (101, 108, "janzing_final"),
)


def _group_for_pair(pair_id: int) -> str:
    for lower, upper, group in _PAIR_GROUPS:
        if lower <= pair_id <= upper:
            return group
    return f"unlisted_{pair_id:04d}"


def _integral_igci_score(x: np.ndarray, y: np.ndarray) -> float:
    """Return the literature integral-IGCI score, positive for x -> y.

    The implementation uses the log-slope integral estimator from Daniusis et
    al. (UAI 2010).  The sign is expressed in the project convention: positive
    means the first listed variable is predicted to cause the second.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_order = np.argsort(x, kind="mergesort")
    y_order = np.argsort(y, kind="mergesort")
    forward = 0.0
    reverse = 0.0
    for left, right in zip(x_order[:-1], x_order[1:]):
        dx = x[right] - x[left]
        dy = y[right] - y[left]
        if dx != 0.0 and dy != 0.0:
            forward += math.log(abs(dy / dx))
    for left, right in zip(y_order[:-1], y_order[1:]):
        dx = x[right] - x[left]
        dy = y[right] - y[left]
        if dx != 0.0 and dy != 0.0:
            reverse += math.log(abs(dx / dy))
    # The published reference implementation returns forward - reverse; its
    # orientation convention is opposite to the pair files' cause -> effect
    # arrow.  Negating yields the explicit project convention above.
    return float((reverse - forward) / max(len(x), 1))


def _confidence(score: float) -> float:
    """Predeclared monotone confidence transform for the auxiliary source."""
    magnitude = abs(float(score))
    return float(magnitude / (1.0 + magnitude))


def _load_pairs(max_rows: int = 600, seed: int = 20260824) -> list[Dict[str, Any]]:
    root = Path(RAW_DIR) / "tuebingen" / "pairs"
    meta_path = root / "pairmeta.txt"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Tuebingen data missing at {root}; run backend/download_tuebingen.py"
        )
    rows = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 6:
            continue
        pair_id, cause_first, cause_last, effect_first, effect_last, weight = fields
        pair_number = int(pair_id)
        cause_first, cause_last = int(cause_first), int(cause_last)
        effect_first, effect_last = int(effect_first), int(effect_last)
        weight = float(weight)
        # The bivariate benchmark is the only part for which the pairwise
        # estimators below have a defined contract; zero-weight entries are
        # excluded exactly as prescribed by the dataset README.
        if weight <= 0 or cause_first != cause_last or effect_first != effect_last:
            continue
        data = np.loadtxt(root / f"pair{pair_id}.txt", dtype=float)
        if data.ndim != 2 or data.shape[1] < max(cause_first, effect_first):
            continue
        data = data[np.isfinite(data[:, cause_first - 1]) & np.isfinite(data[:, effect_first - 1])]
        if len(data) < 30:
            continue
        local_seed = int(seed + pair_number * 7919)
        if len(data) > max_rows:
            rng = np.random.RandomState(local_seed)
            indices = rng.choice(len(data), size=max_rows, replace=False)
            data = data[np.sort(indices)]
        x = data[:, cause_first - 1]
        y = data[:, effect_first - 1]
        if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
            continue
        rows.append({
            "pair_id": pair_id,
            "pair_number": pair_number,
            "group": _group_for_pair(pair_number),
            "weight": weight,
            "n": int(len(x)),
            "x": x,
            "y": y,
            "truth": 1 if cause_first == 1 and effect_first == 2 else -1,
        })
    if len(rows) < 60:
        raise RuntimeError(f"expected at least 60 usable Tuebingen pairs, got {len(rows)}")
    return rows


def _source_and_data_scores(rows: Iterable[Dict[str, Any]]) -> None:
    model = ANM()
    for row in rows:
        x, y = row["x"], row["y"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forward_p, reverse_p = model.cause_or_effect(
                x.reshape(-1, 1), y.reshape(-1, 1)
            )
        row["source_score"] = _integral_igci_score(x, y)
        row["source_confidence"] = _confidence(row["source_score"])
        row["data_score"] = float(
            np.log(max(float(forward_p), 0.0) + 1e-12)
            - np.log(max(float(reverse_p), 0.0) + 1e-12)
        )
        row["source_prediction"] = int(np.sign(row["source_score"]))
        row["data_prediction"] = int(np.sign(row["data_score"]))


def _fixed_group_split(rows: Sequence[Dict[str, Any]], seed: int, calibration_fraction: float) -> tuple[list[int], list[int]]:
    groups = sorted({row["group"] for row in rows})
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    target = max(1, min(len(groups) - 1, int(round(calibration_fraction * len(groups)))))
    calibration_groups = set(groups[:target])
    calibration = [i for i, row in enumerate(rows) if row["group"] in calibration_groups]
    test = [i for i, row in enumerate(rows) if row["group"] not in calibration_groups]
    if not calibration or not test:
        raise RuntimeError("group split produced an empty partition")
    return calibration, test


def _accuracy(rows: Sequence[Dict[str, Any]], indices: Sequence[int], key: str, weighted: bool = False) -> float:
    if not indices:
        return float("nan")
    values = np.asarray([rows[i]["weight"] if weighted else 1.0 for i in indices], dtype=float)
    correct = np.asarray([rows[i][key] == rows[i]["truth"] for i in indices], dtype=float)
    return float(np.average(correct, weights=values))


def _bootstrap_interval(rows: Sequence[Dict[str, Any]], indices: Sequence[int], values: Dict[str, np.ndarray], seed: int, n_bootstrap: int) -> Dict[str, list[float]]:
    if not indices:
        return {key: [float("nan"), float("nan")] for key in values}
    rng = np.random.RandomState(seed)
    draws = rng.choice(len(indices), size=(n_bootstrap, len(indices)), replace=True)
    output = {}
    for key, array in values.items():
        samples = []
        for draw in draws:
            samples.append(float(np.mean(array[draw])))
        output[key] = [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
    return output


def _family_cluster_bootstrap_interval(
    rows: Sequence[Dict[str, Any]],
    indices: Sequence[int],
    seed: int,
    n_bootstrap: int,
) -> Dict[str, list[float] | int]:
    """Cluster resampling sensitivity for the declared benchmark families."""
    groups = sorted({rows[i]["group"] for i in indices})
    if not groups:
        return {"n_groups": 0, "rcep_delta_accuracy": [float("nan"), float("nan")]}
    by_group = {
        group: [i for i in indices if rows[i]["group"] == group]
        for group in groups
    }
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_indices = [i for group in sampled_groups for i in by_group[str(group)]]
        rcep = _accuracy(rows, sampled_indices, "rcep_prediction")
        data = _accuracy(rows, sampled_indices, "data_prediction")
        deltas.append(rcep - data)
    return {
        "n_groups": len(groups),
        "rcep_delta_accuracy": [
            float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)),
        ],
    }


def run_exp15(
    calibration_fraction: float = 0.67,
    target_error: float = 0.35,
    max_rows: int = 600,
    n_bootstrap: int = 2000,
    seed: int = 20260824,
) -> Dict[str, Any]:
    """Evaluate a fixed real-pair auxiliary-source holdout protocol."""
    if not 0.5 <= calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in [0.5, 1)")
    rows = _load_pairs(max_rows=max_rows, seed=seed)
    _source_and_data_scores(rows)
    calibration, test = _fixed_group_split(rows, seed=seed + 1, calibration_fraction=calibration_fraction)
    calibration_correct = [rows[i]["source_prediction"] == rows[i]["truth"] for i in calibration]
    gate = RiskControlledEvidenceGate(RiskControlConfig(
        target_error=target_error,
        failure_probability=0.05,
        candidate_thresholds=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
        min_accepted_calibration=max(10, len(calibration) // 5),
    )).fit([rows[i]["source_confidence"] for i in calibration], calibration_correct)
    stratified_gate = StratifiedRiskControlledEvidenceGate(RiskControlConfig(
        target_error=target_error,
        failure_probability=0.05,
        candidate_thresholds=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
        min_accepted_calibration=1,
    )).fit(
        [rows[i]["source_confidence"] for i in calibration],
        calibration_correct,
        [rows[i]["group"] for i in calibration],
    )
    for row in rows:
        row["rcep_prediction"] = (
            row["source_prediction"] if gate.accepts(row["source_confidence"])
            else row["data_prediction"]
        )
        row["accepted_source"] = bool(gate.accepts(row["source_confidence"]))
        row["stratified_rcep_prediction"] = (
            row["source_prediction"]
            if stratified_gate.accepts(row["source_confidence"])
            else row["data_prediction"]
        )
        row["stratified_accepted_source"] = bool(
            stratified_gate.accepts(row["source_confidence"])
        )

    source_test = [i for i in test if rows[i]["accepted_source"]]
    metrics = {}
    for weighted in (False, True):
        suffix = "weighted" if weighted else "unweighted"
        data = _accuracy(rows, test, "data_prediction", weighted)
        source = _accuracy(rows, test, "source_prediction", weighted)
        rcep = _accuracy(rows, test, "rcep_prediction", weighted)
        accepted = _accuracy(rows, source_test, "source_prediction", weighted)
        harm = float(np.mean([
            rows[i]["rcep_prediction"] != rows[i]["data_prediction"]
            and rows[i]["rcep_prediction"] != rows[i]["truth"]
            for i in test
        ])) if test else 0.0
        metrics[suffix] = {
            "data_only_accuracy": data,
            "auxiliary_source_accuracy": source,
            "rcep_accuracy": rcep,
            "rcep_delta_accuracy_vs_data": rcep - data,
            "accepted_source_accuracy": accepted,
            "accepted_source_coverage": float(len(source_test) / len(test)) if test else 0.0,
            "rcep_harm_fraction_vs_data": harm,
            "family_stratified_rcep_accuracy": _accuracy(
                rows, test, "stratified_rcep_prediction", weighted
            ),
            "family_stratified_rcep_delta_vs_data": (
                _accuracy(rows, test, "stratified_rcep_prediction", weighted) - data
            ),
            "family_stratified_accepted_coverage": float(np.mean([
                rows[i]["stratified_accepted_source"] for i in test
            ])) if test else 0.0,
        }
    test_arrays = {
        "data_accuracy": np.asarray([rows[i]["data_prediction"] == rows[i]["truth"] for i in test], dtype=float),
        "source_accuracy": np.asarray([rows[i]["source_prediction"] == rows[i]["truth"] for i in test], dtype=float),
        "rcep_accuracy": np.asarray([rows[i]["rcep_prediction"] == rows[i]["truth"] for i in test], dtype=float),
        "delta_accuracy": np.asarray([
            (rows[i]["rcep_prediction"] == rows[i]["truth"])
            - (rows[i]["data_prediction"] == rows[i]["truth"])
            for i in test
        ], dtype=float),
    }
    gate_sensitivity = {}
    for candidate_error in (0.20, 0.35, 0.50):
        sensitivity_gate = RiskControlledEvidenceGate(RiskControlConfig(
            target_error=candidate_error,
            failure_probability=0.05,
            candidate_thresholds=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
            min_accepted_calibration=max(10, len(calibration) // 5),
        )).fit([rows[i]["source_confidence"] for i in calibration], calibration_correct)
        accepted = [i for i in test if sensitivity_gate.accepts(rows[i]["source_confidence"])]
        gate_sensitivity[str(candidate_error)] = {
            "status": sensitivity_gate.certificate_["status"],
            "selected_threshold": sensitivity_gate.threshold_,
            "test_accepted_coverage": float(len(accepted) / len(test)) if test else 0.0,
            "test_accepted_source_accuracy": (
                _accuracy(rows, accepted, "source_prediction") if accepted else None
            ),
        }
    return {
        "protocol": {
            "dataset": "Tuebingen Cause-Effect Pairs (pairs.zip)",
            "dataset_sha256": "C1BC9CD212B2ED1BC18C87D59761052478983623E3F2C4B4166260FA0A23FB11",
            "n_usable_pairs": len(rows),
            "n_groups": len({row["group"] for row in rows}),
            "pair_filter": "positive-weight bivariate continuous entries only; zero-weight/high-dimensional entries excluded",
            "group_split": "fixed benchmark-family split before fitting the gate",
            "calibration_fraction": calibration_fraction,
            "n_calibration_pairs": len(calibration),
            "n_test_pairs": len(test),
            "n_calibration_groups": len({rows[i]["group"] for i in calibration}),
            "n_test_groups": len({rows[i]["group"] for i in test}),
            "source": "integral IGCI log-slope estimator; no pair labels used to construct scores",
            "data_only": "causal-learn ANM GP/KCI direction score",
            "target_error": target_error,
            "bootstrap": {"n": n_bootstrap, "seed": seed + 2, "unit": "held-out pair"},
            "claim_boundary": "real noisy auxiliary-source holdout on a public pair benchmark; not an external biological knowledge population",
        },
        "gate_certificate": gate.certificate_,
        "family_stratified_gate_certificate": stratified_gate.certificate_,
        "split": {
            "calibration_pair_ids": [rows[i]["pair_id"] for i in calibration],
            "test_pair_ids": [rows[i]["pair_id"] for i in test],
            "calibration_groups": sorted({rows[i]["group"] for i in calibration}),
            "test_groups": sorted({rows[i]["group"] for i in test}),
        },
        "summary": metrics,
        "gate_target_error_sensitivity": gate_sensitivity,
        "test_bootstrap_intervals": _bootstrap_interval(rows, test, test_arrays, seed + 2, n_bootstrap),
        "family_cluster_bootstrap": _family_cluster_bootstrap_interval(
            rows, test, seed + 3, n_bootstrap
        ),
        "accepted_source_pair_ids": [rows[i]["pair_id"] for i in source_test],
        "pair_scores": [
            {key: row[key] for key in (
                "pair_id", "group", "weight", "n", "truth", "source_score",
                "source_confidence", "data_score", "source_prediction", "data_prediction",
                "rcep_prediction", "accepted_source",
                "stratified_rcep_prediction", "stratified_accepted_source",
            )}
            for row in rows
        ],
    }
