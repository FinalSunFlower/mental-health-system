"""Freeze a compact RPE1 Perturb-seq audit set from public CausalBench data."""
from pathlib import Path

from download_causalbench_labels import extract


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/causalbench"
FILE_URL = "https://ndownloader.figshare.com/files/35775606"
FILE_BYTES = 8_700_873_216
HEPG2_CHIP = (
    RAW
    / "repo/causalscbench/data_access/data/Hep_G2_ChipSeq.csv"
)


def main() -> None:
    extract(
        output=RAW / "rpe1_intervention_labels.csv",
        metadata_output=RAW / "rpe1_intervention_labels.meta.json",
        metadata_cache=RAW / "rpe1_h5ad_metadata.npz",
        subset_output=RAW / "rpe1_intervention_subset.npz",
        endpoint=FILE_URL,
        file_size=FILE_BYTES,
        chip_path=HEPG2_CHIP,
        dataset_name="CausalBench / Weissmann RPE1 day-7 Perturb-seq",
        sources=50,
        targets_per_source=30,
        cells_per_source=100,
        control_cells=500,
        seed=20260830,
    )


if __name__ == "__main__":
    main()
