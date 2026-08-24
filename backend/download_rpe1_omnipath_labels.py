"""Freeze RPE1 responses for the independent OmniPath proposal source."""
from pathlib import Path

from download_causalbench_labels import extract


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/causalbench"


def main() -> None:
    extract(
        output=RAW / "rpe1_omnipath_intervention_labels.csv",
        metadata_output=RAW / "rpe1_omnipath_intervention_labels.meta.json",
        metadata_cache=RAW / "rpe1_h5ad_metadata.npz",
        subset_output=RAW / "rpe1_omnipath_intervention_subset.npz",
        endpoint="https://ndownloader.figshare.com/files/35775606",
        file_size=8_700_873_216,
        chip_path=RAW / "rpe1_omnipath_source.csv",
        dataset_name="CausalBench / Weissmann RPE1 day-7 Perturb-seq",
        sources=50,
        targets_per_source=30,
        cells_per_source=100,
        control_cells=500,
        seed=20260832,
        source_ranking="candidate_count",
    )


if __name__ == "__main__":
    main()
