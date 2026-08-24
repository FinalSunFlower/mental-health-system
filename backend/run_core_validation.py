"""Run the strict, manuscript-facing CuspNet validation suite."""
import argparse
import json
import os
import hashlib
import platform
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.experiments import (
    run_exp7,
    run_exp7_sensitivity,
    run_gate_calibration_sensitivity,
    run_exp8,
    run_exp9,
    run_exp10,
    run_exp11,
    run_exp12,
    run_exp13,
    run_exp14,
    run_exp15,
    run_exp16,
    run_exp17,
    run_exp18,
    run_exp23,
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _environment_metadata() -> dict:
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "pandas", "networkx", "causal-learn"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "packages": packages,
        "accelerator": "none; all manuscript-facing experiments run on CPU",
    }


def _input_manifest() -> dict:
    root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "data/raw/sachs_data.csv",
        "backend/resources/omnipath_sachs_prior.json",
        "data/raw/psychology/jobs_ii.csv",
        "data/raw/psychology/star.csv",
        "data/raw/tuebingen/pairs.zip",
        "data/raw/causalbench/K562_ChipSeq.csv",
        "data/raw/causalbench/k562_intervention_labels.csv",
        "data/raw/causalbench/k562_intervention_labels.meta.json",
        "data/raw/causalbench/k562_intervention_subset.npz",
        "data/raw/causalbench/repo/causalscbench/data_access/data/Hep_G2_ChipSeq.csv",
        "data/raw/causalbench/dorothea_abc.tsv",
        "data/raw/causalbench/summary_stats.xlsx",
    )
    manifest = {}
    for relative in relative_paths:
        path = root / relative
        manifest[relative] = {
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
            "sha256": _sha256(path) if path.exists() else None,
        }
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="Explicit output path; use only when intentionally refreshing a lock.",
    )
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or Path(__file__).resolve().parent / "results" / (
        f"cuspnet_strict_core_{timestamp}.json"
    )
    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "scope": (
                "risk-controlled graph projection, psychology design-safety, "
                "external-transfer, source-held-out baselines, factorial robustness, "
                "and solver/gate audits"
                ", plus CausalBench intervention-source holdout, independent source-construction audit, "
                "and cross-cell context-representation validation"
            ),
            "environment": _environment_metadata(),
            "input_manifest": _input_manifest(),
        },
        "experiments": {
            "exp7_risk_controlled_projection": run_exp7(),
            "exp7_scale_sensitivity": run_exp7_sensitivity(),
            "exp7_gate_sensitivity": run_gate_calibration_sensitivity(),
            "exp8_external_prior_transfer": run_exp8(),
            "exp9_psychology_design_validation": run_exp9(),
            "exp10_solver_scaling": run_exp10(),
            "exp11_finite_sample_gate_audit": run_exp11(),
            "exp12_cross_study_design_transfer": run_exp12(),
            "exp13_source_holdout_and_baselines": run_exp13(),
            "exp14_hyperparameter_robustness": run_exp14(),
            "exp15_tuebingen_source_holdout": run_exp15(),
            "exp16_causalbench_intervention_holdout": run_exp16(),
            "exp17_causalbench_source_variants": run_exp17(),
            "exp18_stratified_shift_audit": run_exp18(),
            "exp23_cross_cell_context_representation": run_exp23(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonable(payload), handle, ensure_ascii=False, indent=2,
            allow_nan=False,
        )
    print(output)


if __name__ == "__main__":
    main()
