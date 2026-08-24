"""Cross-cell perturbation-context representation from official summaries."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline

from app.cuspnet.contracts import RiskControlConfig, RiskControlledEvidenceGate
from app.cuspnet.evidence_semantics import EvidenceSemantics, route_evidence


ROOT = Path(__file__).resolve().parents[3]
SUMMARY_PATH = ROOT / "data/raw/causalbench/summary_stats.xlsx"
FEATURES = (
    "mean UMI count",
    "number of cells (filtered)",
    "target expression",
    "fold knockdown",
    "percent knockdown",
    "mean leverage score",
    "std of leverage score",
    "energy test (p-value)",
    "Number of DEGs (anderson-darling)",
    "Number of DEGs (Mann-Whitney)",
    "Z-scored number of UMIs",
    "Fraction of mitochondrially encoded RNA",
    "Fraction of TE RNA",
    "Z-scored CIN score",
)


def _load_sheet(path: Path, sheet: str, prefix: str) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=sheet)
    frame["gene"] = frame["genetic perturbation"].astype(str).str.extract(
        r"^\d+_(.+?)_P(?:1|1P2|2)"
    )[0]
    return (
        frame.dropna(subset=["gene"])
        .sort_values("number of cells (filtered)", ascending=False)
        .drop_duplicates("gene")
        .set_index("gene")
        .add_prefix(prefix)
    )


def _dataset(path: Path = SUMMARY_PATH) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing official CausalBench summaries: {path}")
    day8 = _load_sheet(path, "TabA_K562_day8_summary_stat", "day8_")
    day6 = _load_sheet(path, "TabB_K562_day6_summary_stat", "day6_")
    rpe1 = _load_sheet(path, "TabC_RPE1_summary_statistic", "rpe1_")
    joined = day8.join(day6, how="inner").join(rpe1, how="inner")
    columns = [f"{view}_{feature}" for view in ("day6", "day8") for feature in FEATURES]
    values = joined[columns].copy()
    for view in ("day6", "day8"):
        values[f"{view}_energy test (p-value)"] = -np.log10(
            values[f"{view}_energy test (p-value)"].clip(lower=1e-12)
        )
        for feature in (
            "Number of DEGs (anderson-darling)",
            "Number of DEGs (Mann-Whitney)",
            "number of cells (filtered)",
            "mean UMI count",
        ):
            values[f"{view}_{feature}"] = np.log1p(
                values[f"{view}_{feature}"].clip(lower=0.0)
            )
    labels = (
        (joined["rpe1_energy test (p-value)"] <= 0.05)
        & (joined["rpe1_Number of DEGs (Mann-Whitney)"] >= 1)
    ).to_numpy(dtype=bool)
    return values, labels, joined.index.astype(str).tolist()


def _model(seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            max_features=0.8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    )


def _split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(n)
    np.random.RandomState(seed).shuffle(indices)
    train_end = int(0.30 * n)
    calibration_end = int(0.70 * n)
    return indices[:train_end], indices[train_end:calibration_end], indices[calibration_end:]


def _one_split(
    features: pd.DataFrame,
    labels: np.ndarray,
    genes: list[str],
    seed: int,
) -> dict[str, Any]:
    train, calibration, test = _split(len(labels), seed)
    model = _model(20260905).fit(features.iloc[train], labels[train])
    calibration_score = model.predict_proba(features.iloc[calibration])[:, 1]
    test_score = model.predict_proba(features.iloc[test])[:, 1]
    gate = RiskControlledEvidenceGate(RiskControlConfig(
        target_error=0.08,
        failure_probability=0.05,
        candidate_thresholds=tuple(np.linspace(0.0, 1.0, 41)),
        min_accepted_calibration=50,
    )).fit(calibration_score, labels[calibration])
    accepted = gate.filter_mask(test_score)
    n_accepted = int(np.sum(accepted))
    accepted_error = (
        float(np.mean(~labels[test][accepted])) if n_accepted else None
    )
    return {
        "seed": seed,
        "n_train": len(train),
        "n_calibration": len(calibration),
        "n_test": len(test),
        "train_genes_sha256": hashlib.sha256(
            "\n".join(genes[index] for index in train).encode()
        ).hexdigest(),
        "calibration_genes_sha256": hashlib.sha256(
            "\n".join(genes[index] for index in calibration).encode()
        ).hexdigest(),
        "test_genes_sha256": hashlib.sha256(
            "\n".join(genes[index] for index in test).encode()
        ).hexdigest(),
        "gate_certificate": gate.certificate_,
        "test": {
            "average_precision": float(average_precision_score(labels[test], test_score)),
            "roc_auc": float(roc_auc_score(labels[test], test_score)),
            "accepted": n_accepted,
            "coverage": float(np.mean(accepted)),
            "accepted_error": accepted_error,
            "accepted_precision": (
                float(1.0 - accepted_error) if accepted_error is not None else None
            ),
            "target_violation": bool(
                accepted_error is not None and accepted_error > 0.08
            ),
        },
    }


def run_exp23(sensitivity_repetitions: int = 20) -> dict[str, Any]:
    features, labels, genes = _dataset()
    semantics = EvidenceSemantics.detectable_response(
        "RPE1 energy-test p <= 0.05 and at least one Mann-Whitney DEG"
    )
    route = route_evidence(semantics)
    primary = _one_split(features, labels, genes, 20260905)
    sensitivity = [
        _one_split(features, labels, genes, 20260905 + repeat * 1009)
        for repeat in range(sensitivity_repetitions)
    ]
    certified = [
        result for result in sensitivity
        if result["gate_certificate"]["status"] == "certified"
    ]
    return {
        "protocol": {
            "dataset": "CausalBench official perturbation summary statistics",
            "source_context": "K562 day-6 and day-8 perturbation summaries only",
            "deployment_label": semantics.label_definition,
            "evidence_semantics": semantics.kind,
            "evidence_route": route.route,
            "unknown_policy": "unknown_entries_are_ineligible_for_certificate",
            "split": "gene-disjoint 30% representation train / 40% gate calibration / 30% test",
            "model": "fixed ExtraTrees context encoder; no RPE1 feature enters training",
            "n_shared_genes": len(labels),
            "rpe1_positive_prevalence": float(np.mean(labels)),
            "target_error": 0.08,
            "summary_stats_sha256": hashlib.sha256(SUMMARY_PATH.read_bytes()).hexdigest(),
            "claim_boundary": (
                "cross-cell global perturbation-response readiness, not pairwise edge "
                "truth, causal effect size, or complete DAG recovery"
            ),
        },
        "primary": primary,
        "split_sensitivity": {
            "repetitions": sensitivity_repetitions,
            "certificate_rate": len(certified) / sensitivity_repetitions,
            "mean_coverage_when_certified": (
                float(np.mean([result["test"]["coverage"] for result in certified]))
                if certified else 0.0
            ),
            "mean_precision_when_certified": (
                float(np.mean([
                    result["test"]["accepted_precision"] for result in certified
                ])) if certified else None
            ),
            "minimum_precision_when_certified": (
                float(np.min([
                    result["test"]["accepted_precision"] for result in certified
                ])) if certified else None
            ),
            "target_violation_fraction_when_certified": (
                float(np.mean([
                    result["test"]["target_violation"] for result in certified
                ])) if certified else None
            ),
            "runs": sensitivity,
        },
    }
