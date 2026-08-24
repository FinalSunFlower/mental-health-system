"""RPE1 cross-cell-line source audit with independently frozen interventions."""
from pathlib import Path
from typing import Any

from .exp16_causalbench_intervention_holdout import run_exp16


ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw/causalbench"


def run_exp19() -> dict[str, Any]:
    return run_exp16(
        calibration_fraction=0.67,
        target_error=0.35,
        seed=20260831,
        n_bootstrap=2000,
        label_path=RAW / "rpe1_intervention_labels.csv",
        meta_path=RAW / "rpe1_intervention_labels.meta.json",
        subset_path=RAW / "rpe1_intervention_subset.npz",
        dataset_name="CausalBench / Weissmann RPE1 day-7 Perturb-seq",
        knowledge_source="HepG2 ChIP-Atlas TF-target evidence",
        label_source=(
            "independent RPE1 CRISPR perturbation expression response from the public H5AD"
        ),
        missing_hint="run backend/download_rpe1_labels.py",
    )
