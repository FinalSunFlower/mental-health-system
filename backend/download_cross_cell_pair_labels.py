"""Freeze a TF-disjoint K562-to-RPE1 pairwise intervention-transfer dataset.

The candidate population is constructed only from cell-line-specific ChIP
replication and H5AD metadata.  Intervention response labels are read only
after this population and its source groups have been frozen.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from download_causalbench_labels import (
    FILE_BYTES as K562_BYTES,
    FILE_URL as K562_URL,
    _metadata,
    extract,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/causalbench"
RPE1_URL = "https://ndownloader.figshare.com/files/35775606"
RPE1_BYTES = 8_700_873_216
K562_CHIP = RAW / "K562_ChipSeq.csv"
HEPG2_CHIP = RAW / "repo/causalscbench/data_access/data/Hep_G2_ChipSeq.csv"
PAIR_SOURCE = RAW / "cross_cell_replicated_chip_pairs.csv"
PAIR_SOURCE_META = RAW / "cross_cell_replicated_chip_pairs.meta.json"
K562_LABELS = RAW / "cross_cell_k562_pair_labels.csv"
K562_LABELS_META = RAW / "cross_cell_k562_pair_labels.meta.json"
K562_SUBSET = RAW / "cross_cell_k562_pair_subset.npz"
RPE1_LABELS = RAW / "cross_cell_rpe1_pair_labels.csv"
RPE1_LABELS_META = RAW / "cross_cell_rpe1_pair_labels.meta.json"
RPE1_SUBSET = RAW / "cross_cell_rpe1_pair_subset.npz"

# Frozen before response extraction.  The size supports source-group holdout
# and a zero-error finite-sample certificate if the two assays agree.
N_SOURCES = 15
TARGETS_PER_SOURCE = 100
MIN_K562_CELLS_FOR_SOURCE = 50
MIN_RPE1_CELLS_FOR_SOURCE = 50
K562_CELLS_PER_SOURCE = 50
RPE1_CELLS_PER_SOURCE = 40
K562_CONTROL_CELLS = 300
RPE1_CONTROL_CELLS = 200
SEED = 20260912


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata_sets(cache: Path, endpoint: str, file_size: int) -> tuple[set[str], set[str], Counter]:
    interventions, variables, _, _, _ = _metadata(
        cache, endpoint=endpoint, file_size=file_size
    )
    names = set(variables.astype(str).tolist())
    interventions = interventions.astype(str)
    return set(interventions.tolist()), names, Counter(interventions.tolist())


def freeze_candidate_population() -> tuple[pd.DataFrame, list[str]]:
    k_interventions, k_genes, k_counts = _metadata_sets(
        RAW / "k562_h5ad_metadata.npz", K562_URL, K562_BYTES
    )
    r_interventions, r_genes, r_counts = _metadata_sets(
        RAW / "rpe1_h5ad_metadata.npz", RPE1_URL, RPE1_BYTES
    )
    shared_sources = (k_interventions & k_genes) & (r_interventions & r_genes)
    shared_targets = k_genes & r_genes
    k_chip = pd.read_csv(K562_CHIP, usecols=["source", "target", "weight"]).dropna()
    r_chip = pd.read_csv(HEPG2_CHIP, usecols=["source", "target", "weight"]).dropna()
    for frame in (k_chip, r_chip):
        frame["source"] = frame["source"].astype(str)
        frame["target"] = frame["target"].astype(str)
    k_chip = k_chip[
        k_chip["source"].isin(shared_sources) & k_chip["target"].isin(shared_targets)
    ]
    r_chip = r_chip[
        r_chip["source"].isin(shared_sources) & r_chip["target"].isin(shared_targets)
    ]
    merged = k_chip.merge(r_chip, on=["source", "target"], suffixes=("_k562", "_hepg2"))
    merged = merged[
        (merged["source"].map(k_counts) >= MIN_K562_CELLS_FOR_SOURCE)
        & (merged["source"].map(r_counts) >= MIN_RPE1_CELLS_FOR_SOURCE)
    ].copy()
    merged["joint_chip_weight"] = merged[["weight_k562", "weight_hepg2"]].min(axis=1)
    ranked: list[pd.DataFrame] = []
    source_rows: list[tuple[str, int, int]] = []
    for source, group in merged.groupby("source", sort=True):
        top = group.sort_values(
            ["joint_chip_weight", "target"], ascending=[False, True]
        ).head(TARGETS_PER_SOURCE)
        if len(top) == TARGETS_PER_SOURCE:
            ranked.append(top)
            source_rows.append((source, min(k_counts[source], r_counts[source]), len(group)))
    source_rows.sort(key=lambda row: (-row[1], -row[2], row[0]))
    selected_sources = [row[0] for row in source_rows[:N_SOURCES]]
    if len(selected_sources) != N_SOURCES:
        raise RuntimeError(f"only {len(selected_sources)} shared sources satisfy the frozen protocol")
    candidates = pd.concat(
        [group[group["source"].isin(selected_sources)] for group in ranked],
        ignore_index=True,
    )
    candidates = candidates[candidates["source"].isin(selected_sources)].copy()
    candidates = candidates.sort_values(
        ["source", "joint_chip_weight", "target"], ascending=[True, False, True]
    )
    expected_rows = N_SOURCES * TARGETS_PER_SOURCE
    if len(candidates) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} frozen pairs, obtained {len(candidates)}")
    candidates.rename(columns={"joint_chip_weight": "weight"}).to_csv(PAIR_SOURCE, index=False)
    payload = {
        "protocol": "K562/HepG2 replicated ChIP pairs selected before intervention-response extraction",
        "seed": SEED,
        "n_sources": N_SOURCES,
        "targets_per_source": TARGETS_PER_SOURCE,
        "sources": selected_sources,
        "selection": {
            "source_present_as_intervention_and_measured_in_both_cell_lines": True,
            "target_measured_in_both_cell_lines": True,
            "minimum_k562_cells_for_source": MIN_K562_CELLS_FOR_SOURCE,
            "minimum_rpe1_cells_for_source": MIN_RPE1_CELLS_FOR_SOURCE,
            "rank": "minimum of K562 and HepG2 ChIP weight; source rank by minimum intervention-cell count then candidate count",
            "uses_intervention_response_labels": False,
        },
        "inputs": {
            "k562_chip_sha256": _sha256(K562_CHIP),
            "hepg2_chip_sha256": _sha256(HEPG2_CHIP),
            "k562_metadata_sha256": _sha256(RAW / "k562_h5ad_metadata.npz"),
            "rpe1_metadata_sha256": _sha256(RAW / "rpe1_h5ad_metadata.npz"),
        },
        "pair_csv_sha256": _sha256(PAIR_SOURCE),
    }
    PAIR_SOURCE_META.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return candidates, selected_sources


def main() -> None:
    candidates, sources = freeze_candidate_population()
    print(f"frozen {len(candidates)} pairs across {len(sources)} TF-disjoint groups")
    if K562_LABELS.exists() and K562_LABELS_META.exists() and K562_SUBSET.exists():
        print("preserving existing frozen K562 source-label artifact")
    else:
        extract(
            output=K562_LABELS,
            metadata_output=K562_LABELS_META,
            metadata_cache=RAW / "k562_h5ad_metadata.npz",
            subset_output=K562_SUBSET,
            endpoint=K562_URL,
            file_size=K562_BYTES,
            chip_path=PAIR_SOURCE,
            dataset_name="CausalBench / Weissmann K562 day-6 pairwise transfer source",
            sources=N_SOURCES,
            targets_per_source=TARGETS_PER_SOURCE,
            cells_per_source=K562_CELLS_PER_SOURCE,
            control_cells=K562_CONTROL_CELLS,
            seed=SEED,
            fixed_sources=sources,
            range_workers=64,
        )
    extract(
        output=RPE1_LABELS,
        metadata_output=RPE1_LABELS_META,
        metadata_cache=RAW / "rpe1_h5ad_metadata.npz",
        subset_output=RPE1_SUBSET,
        endpoint=RPE1_URL,
        file_size=RPE1_BYTES,
        chip_path=PAIR_SOURCE,
        dataset_name="CausalBench / Weissmann RPE1 day-7 pairwise transfer target",
        sources=N_SOURCES,
        targets_per_source=TARGETS_PER_SOURCE,
        cells_per_source=RPE1_CELLS_PER_SOURCE,
        control_cells=RPE1_CONTROL_CELLS,
        seed=SEED,
        fixed_sources=sources,
        range_workers=64,
    )


if __name__ == "__main__":
    main()
