"""Run both frozen, public RPE1 source audits and save one compact artifact."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.experiments import run_exp19, run_exp20, run_exp21, run_exp22, run_exp23


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    paths = (
        ROOT / "data/raw/causalbench/rpe1_intervention_labels.csv",
        ROOT / "data/raw/causalbench/rpe1_intervention_labels.meta.json",
        ROOT / "data/raw/causalbench/rpe1_intervention_subset.npz",
        ROOT / "data/raw/causalbench/rpe1_omnipath_source.csv",
        ROOT / "data/raw/causalbench/rpe1_omnipath_source.meta.json",
        ROOT / "data/raw/causalbench/rpe1_omnipath_intervention_labels.csv",
        ROOT / "data/raw/causalbench/rpe1_omnipath_intervention_labels.meta.json",
        ROOT / "data/raw/causalbench/rpe1_omnipath_intervention_subset.npz",
        ROOT / "data/raw/causalbench/summary_stats.xlsx",
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen RPE1 inputs: {missing}")
    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "scope": "public zero-application RPE1 external-source audit",
            "input_manifest": {
                str(path.relative_to(ROOT)).replace("\\", "/"): {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in paths
            },
        },
        "experiments": {
            "rpe1_hepg2_chip": run_exp19(),
            "rpe1_omnipath": run_exp20(),
            "rpe1_context_conditioned": run_exp21(),
            "contextual_controlled_validation": run_exp22(),
            "cross_cell_context_representation": run_exp23(),
        },
    }
    output = ROOT / "backend/results/rpe1_public_source_audit.json"
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(output)
    print(_sha256(output).upper())


if __name__ == "__main__":
    main()
