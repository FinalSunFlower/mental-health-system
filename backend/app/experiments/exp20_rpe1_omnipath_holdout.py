"""Independent OmniPath-to-RPE1 intervention-response holdout audit."""
from pathlib import Path
from typing import Any

from .exp16_causalbench_intervention_holdout import run_exp16


ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw/causalbench"


def run_exp20() -> dict[str, Any]:
    return run_exp16(
        calibration_fraction=0.67,
        target_error=0.35,
        seed=20260833,
        n_bootstrap=2000,
        label_path=RAW / "rpe1_omnipath_intervention_labels.csv",
        meta_path=RAW / "rpe1_omnipath_intervention_labels.meta.json",
        subset_path=RAW / "rpe1_omnipath_intervention_subset.npz",
        dataset_name="CausalBench / Weissmann RPE1 day-7 Perturb-seq",
        knowledge_source=(
            "frozen OmniPath consensus directions with curation effort at least 1"
        ),
        label_source=(
            "independent RPE1 CRISPR perturbation expression response from the public H5AD"
        ),
        missing_hint=(
            "run backend/download_rpe1_omnipath_source.py and "
            "backend/download_rpe1_omnipath_labels.py"
        ),
        confidence_scale=1.0,
    )
