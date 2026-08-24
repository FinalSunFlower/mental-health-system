"""Pre-registered source-construction audit for the CausalBench holdout.

This experiment asks whether independent regulatory-source replication can
improve the K562 ChIP proposal population.  The intervention labels remain
locked to the existing CausalBench extraction and are never used to construct
the source variants.  Unknown responses are retained as unknown; they are not
negative causal labels.
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

from app.cuspnet.contracts import RiskControlConfig, RiskControlledEvidenceGate

ROOT = Path(__file__).resolve().parents[3]
LABEL_PATH = ROOT / "data/raw/causalbench/k562_intervention_labels.csv"
META_PATH = ROOT / "data/raw/causalbench/k562_intervention_labels.meta.json"
SUBSET_PATH = ROOT / "data/raw/causalbench/k562_intervention_subset.npz"
HEPG2_PATH = ROOT / "data/raw/causalbench/repo/causalscbench/data_access/data/Hep_G2_ChipSeq.csv"
DOROTHEA_PATH = ROOT / "data/raw/causalbench/dorothea_abc.tsv"


def _display_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _confidence(weight: float) -> float:
    return float(1.0 - np.exp(-max(float(weight), 0.0) / 500.0))


def _bootstrap(values: np.ndarray, seed: int, n: int) -> list[float | None]:
    if len(values) == 0:
        return [None, None]
    rng = np.random.RandomState(seed)
    draws = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _data_direction(
    control: np.ndarray,
    gene_names: np.ndarray,
    source: str,
    target: str,
    model: ANM,
) -> float:
    source_index = int(np.flatnonzero(gene_names == source)[0])
    target_index = int(np.flatnonzero(gene_names == target)[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forward, reverse = model.cause_or_effect(
            control[:, source_index].reshape(-1, 1),
            control[:, target_index].reshape(-1, 1),
        )
    return float(np.log(max(float(forward), 0.0) + 1e-12) - np.log(max(float(reverse), 0.0) + 1e-12))


def _support_metrics(frame: pd.DataFrame, accepted_only: bool = False) -> dict[str, Any]:
    selected = frame[frame["accepted_source"]] if accepted_only else frame
    if len(selected) == 0:
        return {"n": 0, "coverage": 0.0, "support_precision": None, "false_support_rate": None}
    precision = float(selected["label"].mean())
    return {
        "n": int(len(selected)),
        "coverage": float(len(selected) / len(frame)) if len(frame) else 0.0,
        "support_precision": precision,
        "false_support_rate": float(1.0 - precision),
    }


def _directional_accuracy(frame: pd.DataFrame, column: str) -> float | None:
    return float(np.mean(frame[column].to_numpy() == 1)) if len(frame) else None


def _source_tables() -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = pd.read_csv(LABEL_PATH)
    labels["source"] = labels["source"].astype(str).str.upper()
    labels["target"] = labels["target"].astype(str).str.upper()
    labels["confidence"] = labels["chip_weight"].map(_confidence)
    if hashlib.sha256(LABEL_PATH.read_bytes()).hexdigest() != json.loads(META_PATH.read_text(encoding="utf-8"))["csv_sha256"]:
        raise RuntimeError("CausalBench label hash does not match metadata")

    metadata: dict[str, Any] = {
        "k562_label_sha256": hashlib.sha256(LABEL_PATH.read_bytes()).hexdigest(),
    }
    if HEPG2_PATH.exists():
        hep = pd.read_csv(HEPG2_PATH, usecols=["source", "target"])
        hep["source"] = hep["source"].astype(str).str.upper()
        hep["target"] = hep["target"].astype(str).str.upper()
        replicated = hep.drop_duplicates(["source", "target"])
        replicated["replicated"] = True
        labels = labels.merge(replicated, on=["source", "target"], how="left")
        replicated_mask = labels["replicated"].eq(True)
        metadata.update({
            "hepg2_path": _display_path(HEPG2_PATH),
            "hepg2_sha256": hashlib.sha256(HEPG2_PATH.read_bytes()).hexdigest(),
            "k562_hepg2_overlap_rows": int(replicated_mask.sum()),
        })
    else:
        labels["replicated"] = False
        metadata["hepg2_missing"] = True

    if DOROTHEA_PATH.exists():
        dorothea = pd.read_csv(DOROTHEA_PATH, sep="\t", usecols=["source", "target", "confidence"])
        dorothea["source"] = dorothea["source"].astype(str).str.upper()
        dorothea["target"] = dorothea["target"].astype(str).str.upper()
        dorothea = dorothea.drop_duplicates(["source", "target"])
        dorothea["dorothea"] = True
        labels = labels.merge(dorothea[["source", "target", "confidence", "dorothea"]], on=["source", "target"], how="left", suffixes=("", "_dorothea"))
        dorothea_mask = labels["dorothea"].eq(True)
        metadata.update({
            "dorothea_path": _display_path(DOROTHEA_PATH),
            "dorothea_sha256": hashlib.sha256(DOROTHEA_PATH.read_bytes()).hexdigest(),
            "dorothea_overlap_rows": int(dorothea_mask.sum()),
            "dorothea_source": "OmniPath DoRothEA levels A/B/C, downloaded before label evaluation",
        })
    else:
        labels["dorothea"] = False
        labels["confidence_dorothea"] = None
        metadata["dorothea_missing"] = True
    return labels, metadata


def run_exp17(
    calibration_fraction: float = 0.67,
    target_error: float = 0.35,
    seed: int = 20260827,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    """Compare source variants under one immutable TF-disjoint split."""
    if not LABEL_PATH.exists() or not SUBSET_PATH.exists():
        raise FileNotFoundError("CausalBench derived labels and expression subset are required")
    labels, source_metadata = _source_tables()
    subset = np.load(SUBSET_PATH, allow_pickle=False)
    gene_names = subset["gene_names"].astype(str)
    control = subset["control_expression"].astype(float)
    groups = sorted(labels["source"].unique().tolist())
    rng = np.random.RandomState(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    n_cal = max(1, min(len(groups) - 1, round(calibration_fraction * len(groups))))
    calibration_groups = set(shuffled[:n_cal])
    labels["partition"] = labels["source"].map(lambda value: "calibration" if value in calibration_groups else "held_out")

    replicated_mask = labels["replicated"].eq(True).to_numpy(dtype=bool)
    dorothea_mask = labels["dorothea"].eq(True).to_numpy(dtype=bool)
    variant_masks = {
        "k562_chip_only": np.ones(len(labels), dtype=bool),
        "k562_hepg2_replicated": replicated_mask,
        "k562_dorothea_abc_consensus": dorothea_mask,
        "k562_hepg2_dorothea_consensus": replicated_mask & dorothea_mask,
    }
    model = ANM()
    data_score_cache: dict[tuple[str, str], float] = {}
    variants: dict[str, Any] = {}
    for name, mask in variant_masks.items():
        frame = labels.loc[mask].copy()
        calibration = frame[frame["partition"] == "calibration"]
        held_out = frame[frame["partition"] == "held_out"].copy()
        gate = RiskControlledEvidenceGate(RiskControlConfig(
            target_error=target_error,
            failure_probability=0.05,
            candidate_thresholds=(0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            min_accepted_calibration=max(20, len(calibration) // 10),
        )).fit(calibration["confidence"], calibration["label"].astype(bool))
        frame["accepted_source"] = frame["confidence"].map(gate.accepts)
        held_out["accepted_source"] = held_out["confidence"].map(gate.accepts)
        positive = held_out[held_out["label"] == 1].copy()
        if len(positive):
            scores = []
            for row in positive.itertuples(index=False):
                key = (str(row.source), str(row.target))
                if key not in data_score_cache:
                    data_score_cache[key] = _data_direction(control, gene_names, key[0], key[1], model)
                scores.append(data_score_cache[key])
            positive["data_prediction"] = np.where(np.asarray(scores) >= 0.0, 1, -1)
            positive["source_prediction"] = 1
            positive["rcep_prediction"] = np.where(positive["accepted_source"], 1, positive["data_prediction"])
        else:
            positive["data_prediction"] = []
            positive["source_prediction"] = []
            positive["rcep_prediction"] = []
        if len(positive):
            delta_values = (
                (positive["rcep_prediction"].to_numpy() == 1).astype(float)
                - (positive["data_prediction"].to_numpy() == 1).astype(float)
            )
            direction = {
                "data_only_accuracy": _directional_accuracy(positive, "data_prediction"),
                "source_accuracy": _directional_accuracy(positive, "source_prediction"),
                "rcep_accuracy": _directional_accuracy(positive, "rcep_prediction"),
                "rcep_delta_vs_data": float(np.mean(delta_values)),
                "rcep_harm_fraction_vs_data": float(np.mean((positive["rcep_prediction"] != 1) & (positive["data_prediction"] == 1))),
                "rcep_delta_bootstrap_95": _bootstrap(delta_values, seed + 1, n_bootstrap),
            }
        else:
            direction = {
                "data_only_accuracy": None, "source_accuracy": None,
                "rcep_accuracy": None, "rcep_delta_vs_data": None,
                "rcep_harm_fraction_vs_data": None, "rcep_delta_bootstrap_95": [None, None],
            }
        variants[name] = {
            "n_candidates": int(len(frame)),
            "n_calibration_candidates": int(len(calibration)),
            "n_held_out_candidates": int(len(held_out)),
            "n_held_out_positive_labels": int(len(positive)),
            "gate_certificate": gate.certificate_,
            "held_out_support": _support_metrics(held_out),
            "held_out_accepted_support": _support_metrics(held_out, accepted_only=True),
            "directional_positive_labels": direction,
        }

    return {
        "protocol": {
            "dataset": "CausalBench / Weissmann K562 day-6 Perturb-seq",
            "source_variants": list(variant_masks),
            "split": "one fixed TF-disjoint split shared by every source variant",
            "calibration_fraction": calibration_fraction,
            "n_sources": len(groups),
            "n_calibration_sources": len(calibration_groups),
            "n_held_out_sources": len(groups) - len(calibration_groups),
            "target_error": target_error,
            "claim_boundary": "source-construction sensitivity only; unknown intervention responses remain unknown and no variant is a complete DAG truth",
        },
        "source_metadata": source_metadata,
        "split": {
            "calibration_sources": sorted(calibration_groups),
            "held_out_sources": sorted(set(groups) - calibration_groups),
        },
        "variants": variants,
    }
