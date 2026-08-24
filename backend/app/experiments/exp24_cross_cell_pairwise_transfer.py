"""Cross-cell transfer of pairwise intervention directions with fail-closed admission.

K562 and RPE1 are queried on a frozen population of replicated-ChIP TF-target
pairs.  A positive target label means that perturbing the TF produces a
detectable response for that target under a complete, assay-specific rule.
It establishes the total-effect direction for that pair, but does not establish
a direct edge or a complete cell-line DAG.
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
from sklearn.metrics import average_precision_score, roc_auc_score

from app.cuspnet.contracts import ContextualRiskControlledEvidenceGate, RiskControlConfig
from app.cuspnet.evidence_semantics import EvidenceSemantics, route_evidence
from .exp23_cross_cell_context_representation import _dataset, _model


ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw/causalbench"
PAIR_SOURCE = RAW / "cross_cell_replicated_chip_pairs.csv"
PAIR_SOURCE_META = RAW / "cross_cell_replicated_chip_pairs.meta.json"
K562_LABELS = RAW / "cross_cell_k562_pair_labels.csv"
K562_META = RAW / "cross_cell_k562_pair_labels.meta.json"
RPE1_LABELS = RAW / "cross_cell_rpe1_pair_labels.csv"
RPE1_META = RAW / "cross_cell_rpe1_pair_labels.meta.json"
RPE1_SUBSET = RAW / "cross_cell_rpe1_pair_subset.npz"
SEED = 20260912
SUPPORT_CORRELATION = 0.05


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> tuple[pd.DataFrame, dict[str, Any]]:
    required = (PAIR_SOURCE, PAIR_SOURCE_META, K562_LABELS, K562_META, RPE1_LABELS, RPE1_META, RPE1_SUBSET)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "cross-cell pair labels are missing; run backend/download_cross_cell_pair_labels.py: "
            f"{missing}"
        )
    source_meta = json.loads(PAIR_SOURCE_META.read_text(encoding="utf-8"))
    k_meta = json.loads(K562_META.read_text(encoding="utf-8"))
    r_meta = json.loads(RPE1_META.read_text(encoding="utf-8"))
    source_hash = _sha256(PAIR_SOURCE)
    if k_meta["chipseq_sha256"] != source_hash or r_meta["chipseq_sha256"] != source_hash:
        raise RuntimeError("K562/RPE1 pair-label artifact does not match frozen proposal population")
    k562 = pd.read_csv(K562_LABELS)
    rpe1 = pd.read_csv(RPE1_LABELS)
    pairs = k562.merge(
        rpe1,
        on=["source", "target", "chip_weight"],
        suffixes=("_k562", "_rpe1"),
        validate="one_to_one",
    )
    if len(pairs) != source_meta["n_sources"] * source_meta["targets_per_source"]:
        raise RuntimeError("cross-cell pair table size differs from frozen candidate protocol")
    if set(pairs["source"].unique()) != set(source_meta["sources"]):
        raise RuntimeError("cross-cell pair source groups differ from frozen candidate protocol")
    metadata = {
        "candidate_source_sha256": source_hash,
        "k562_label_sha256": _sha256(K562_LABELS),
        "rpe1_label_sha256": _sha256(RPE1_LABELS),
        "k562_subset_sha256": _sha256(RAW / "cross_cell_k562_pair_subset.npz"),
        "rpe1_subset_sha256": _sha256(RPE1_SUBSET),
        "candidate_protocol": source_meta,
        "k562_sampling": k_meta["sampling"],
        "rpe1_sampling": r_meta["sampling"],
    }
    return pairs, metadata


def _fixed_source_split(sources: list[str]) -> tuple[set[str], set[str]]:
    shuffled = list(sorted(sources))
    np.random.RandomState(SEED).shuffle(shuffled)
    calibration = set(shuffled[:10])
    return calibration, set(shuffled[10:])


def _context_scores(sources: set[str], excluded_genes: set[str]) -> dict[str, float]:
    """Fit the exp23 encoder without any candidate endpoint or pair label."""
    features, labels, genes = _dataset()
    positions = {gene: index for index, gene in enumerate(genes)}
    missing = sorted(sources - set(positions))
    if missing:
        raise RuntimeError(f"candidate source absent from summary statistics: {missing}")
    training = np.asarray([gene not in excluded_genes for gene in genes], dtype=bool)
    model = _model(SEED).fit(features.loc[training], labels[training])
    return {
        source: float(model.predict_proba(features.iloc[[positions[source]]])[:, 1][0])
        for source in sources
    }


def _add_observational_support(pairs: pd.DataFrame) -> pd.DataFrame:
    subset = np.load(RPE1_SUBSET, allow_pickle=False)
    genes = subset["gene_names"].astype(str)
    control = subset["control_expression"].astype(float)
    index = {gene: position for position, gene in enumerate(genes)}
    if not set(pairs["source"]).issubset(index) or not set(pairs["target"]).issubset(index):
        raise RuntimeError("frozen RPE1 control subset is missing a pair endpoint")
    output = pairs.copy()
    output["control_abs_correlation"] = [
        abs(float(np.corrcoef(control[:, index[source]], control[:, index[target]])[0, 1]))
        for source, target in output[["source", "target"]].itertuples(index=False)
    ]
    output["data_supported"] = output["control_abs_correlation"] >= SUPPORT_CORRELATION
    return output


def _source_confidence(frame: pd.DataFrame) -> np.ndarray:
    """K562-only proposal strength; undetectable K562 responses cannot propose."""
    effect = np.clip(np.abs(frame["log1p_effect_k562"].to_numpy()) / 0.50, 0.0, 1.0)
    significance = np.clip(1.0 - frame["q_value_k562"].to_numpy(), 0.0, 1.0)
    return frame["label_k562"].to_numpy(dtype=float) * effect * significance


def _data_direction(frame: pd.DataFrame) -> np.ndarray:
    subset = np.load(RPE1_SUBSET, allow_pickle=False)
    genes = subset["gene_names"].astype(str)
    control = subset["control_expression"].astype(float)
    index = {gene: position for position, gene in enumerate(genes)}
    model = ANM()
    scores: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for source, target in frame[["source", "target"]].itertuples(index=False):
            forward, reverse = model.cause_or_effect(
                control[:, index[source]].reshape(-1, 1),
                control[:, index[target]].reshape(-1, 1),
            )
            scores.append(float(np.log(max(float(forward), 1e-12)) - np.log(max(float(reverse), 1e-12))))
    return np.asarray(scores)


def _classification_summary(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return {"average_precision": None, "roc_auc": None}
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def _evaluate_admission(
    pairs: pd.DataFrame, calibration_sources: set[str]
) -> tuple[pd.DataFrame, ContextualRiskControlledEvidenceGate]:
    calibration = pairs[pairs["source"].isin(calibration_sources)].copy()
    test = pairs[~pairs["source"].isin(calibration_sources)].copy()
    gate = ContextualRiskControlledEvidenceGate(
        RiskControlConfig(
            target_error=0.20,
            failure_probability=0.05,
            candidate_thresholds=(0.0001, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10),
            min_accepted_calibration=10,
        )
    ).fit(
        calibration["source_confidence"],
        calibration["context_compatibility"],
        calibration["label_rpe1"].astype(bool),
    )
    if gate.certificate_["status"] == "certified":
        test["accepted"] = [
            gate.accepts_context(source, context)
            for source, context in test[["source_confidence", "context_compatibility"]].itertuples(index=False)
        ]
    else:
        test["accepted"] = False
    return test, gate


def run_exp24() -> dict[str, Any]:
    pairs, input_metadata = _load()
    semantics = EvidenceSemantics.detectable_response(
        "BH q <= 0.05 and absolute log1p normalized source-perturbation effect >= 0.10"
    )
    route = route_evidence(semantics)
    calibration_sources, test_sources = _fixed_source_split(sorted(pairs["source"].unique()))
    pairs = _add_observational_support(pairs)
    context_by_source = _context_scores(
        set(pairs["source"]), set(pairs["source"]) | set(pairs["target"])
    )
    pairs["source_context"] = pairs["source"].map(context_by_source)
    pairs["source_confidence"] = _source_confidence(pairs)
    # A zero support score cannot be admitted at any fixed positive threshold.
    pairs["context_compatibility"] = np.where(
        pairs["data_supported"],
        pairs["source_context"] * np.clip(pairs["control_abs_correlation"] / 0.20, 0.0, 1.0),
        0.0,
    )
    test, gate = _evaluate_admission(pairs, calibration_sources)
    calibration = pairs[pairs["source"].isin(calibration_sources)].copy()
    accepted = test[test["accepted"]]
    positive_test = test[(test["label_rpe1"] == 1) & test["data_supported"]].copy()
    if len(positive_test):
        positive_test["data_score"] = _data_direction(positive_test)
        positive_test["data_prediction"] = positive_test["data_score"] >= 0.0
        positive_test["rcep_prediction"] = np.where(
            positive_test["accepted"], True, positive_test["data_prediction"]
        )
        data_accuracy = float(positive_test["data_prediction"].mean())
        rcep_accuracy = float(positive_test["rcep_prediction"].mean())
        harm = float(np.mean(
            positive_test["data_prediction"] & ~positive_test["rcep_prediction"]
        ))
    else:
        data_accuracy = None
        rcep_accuracy = None
        harm = None
    test_scores = (
        test["source_confidence"].to_numpy()
        * test["context_compatibility"].to_numpy()
    )
    shuffled_sources = sorted(pairs["source"].unique())
    np.random.RandomState(SEED).shuffle(shuffled_sources)
    fold_sensitivity = []
    for fold_index, fold_test_sources in enumerate(np.array_split(shuffled_sources, 3)):
        fold_test_sources = set(fold_test_sources.tolist())
        fold_calibration_sources = set(shuffled_sources) - fold_test_sources
        fold_test, fold_gate = _evaluate_admission(pairs, fold_calibration_sources)
        fold_scores = (
            fold_test["source_confidence"].to_numpy()
            * fold_test["context_compatibility"].to_numpy()
        )
        fold_sensitivity.append({
            "fold": fold_index,
            "calibration_sources": sorted(fold_calibration_sources),
            "test_sources": sorted(fold_test_sources),
            "gate_status": fold_gate.certificate_["status"],
            "accepted_coverage": float(fold_test["accepted"].mean()),
            "test_score": _classification_summary(
                fold_test["label_rpe1"].to_numpy(), fold_scores
            ),
            "n_test_positive": int(fold_test["label_rpe1"].sum()),
        })
    return {
        "protocol": {
            "dataset": "CausalBench / Weissmann K562 day-6 to RPE1 day-7",
            "task": "cross-cell pairwise detectable intervention-response transfer",
            "candidate_population": "replicated K562/HepG2 ChIP TF-target pairs frozen before response extraction",
            "source_split": "fixed 10/5 TF-disjoint calibration/test split before gate fitting",
            "data_support": f"RPE1 control absolute Pearson correlation >= {SUPPORT_CORRELATION:.2f}",
            "source_context": "exp23 K562 day-6/day-8 encoder fit excluding every candidate source/target endpoint and all pair labels",
            "source_proposal": "K562 detectable pair response direction source -> target with K562-only strength",
            "evidence_semantics": semantics.kind,
            "evidence_route": route.route,
            "certificate_estimand": "accepted RPE1 detectable pair-response error",
            "target_error": 0.20,
            "claim_boundary": (
                "a positive RPE1 label identifies an assay-specific total-effect direction under targeted perturbation; "
                "it does not establish direct adjacency, absence of an edge, or complete DAG recovery"
            ),
            "no_test_label_use": "RPE1 pair labels are used only for held-out evaluation after source-group split and for calibration TFs",
        },
        "input_metadata": input_metadata,
        "split": {
            "calibration_sources": sorted(calibration_sources),
            "test_sources": sorted(test_sources),
            "source_sets_disjoint": calibration_sources.isdisjoint(test_sources),
        },
        "gate_certificate": gate.certificate_,
        "summary": {
            "n_pairs": int(len(pairs)),
            "n_calibration_pairs": int(len(calibration)),
            "n_test_pairs": int(len(test)),
            "n_k562_detectable_pairs": int(pairs["label_k562"].sum()),
            "n_rpe1_detectable_pairs": int(pairs["label_rpe1"].sum()),
            "n_cross_cell_replicated_detectable_pairs": int(
                ((pairs["label_k562"] == 1) & (pairs["label_rpe1"] == 1)).sum()
            ),
            "test_score": _classification_summary(test["label_rpe1"].to_numpy(), test_scores),
            "accepted": {
                "n": int(len(accepted)),
                "coverage": float(len(accepted) / len(test)),
                "precision": float(accepted["label_rpe1"].mean()) if len(accepted) else None,
                "empirical_error": float(1.0 - accepted["label_rpe1"].mean()) if len(accepted) else None,
            },
            "positive_direction": {
                "n_data_supported_rpe1_positive_pairs": int(len(positive_test)),
                "data_only_accuracy": data_accuracy,
                "rcep_accuracy": rcep_accuracy,
                "rcep_delta_vs_data": (
                    rcep_accuracy - data_accuracy
                    if rcep_accuracy is not None and data_accuracy is not None
                    else None
                ),
                "rcep_harm_fraction_vs_data": harm,
            },
            "support_violations": 0,
            "fail_closed_equals_data_only": bool(
                gate.certificate_["status"] != "certified"
                and (rcep_accuracy == data_accuracy if data_accuracy is not None else True)
            ),
        },
        "source_group_sensitivity": {
            "folds": fold_sensitivity,
            "all_folds_abstain": all(
                fold["gate_status"] == "abstain_no_certified_threshold"
                for fold in fold_sensitivity
            ),
            "maximum_accepted_coverage": float(max(
                fold["accepted_coverage"] for fold in fold_sensitivity
            )),
        },
    }
