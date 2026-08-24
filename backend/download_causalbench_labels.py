"""Extract a compact, reproducible CausalBench intervention-label table.

The public CausalBench H5AD files are too large to vendor (8--11 GB).  This
script therefore reads only the metadata and deterministic row ranges needed
for a predeclared K562 subset through the official Figshare Range endpoint.
The resulting CSV contains intervention-confirmed directional pairs and all
statistics needed by the manuscript-facing experiment.  It never treats a
non-significant perturbation as a negative causal label.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
from scipy import stats

FILE_URL = "https://ndownloader.figshare.com/files/35773219"
FILE_BYTES = 10_661_879_995
CHIP_PATH = Path(__file__).resolve().parents[1] / "data/raw/causalbench/K562_ChipSeq.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data/raw/causalbench/k562_intervention_labels.csv"
DEFAULT_METADATA = Path(__file__).resolve().parents[1] / "data/raw/causalbench/k562_intervention_labels.meta.json"
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "data/raw/causalbench/k562_h5ad_metadata.npz"
DEFAULT_SUBSET = Path(__file__).resolve().parents[1] / "data/raw/causalbench/k562_intervention_subset.npz"


class _RangeFile:
    """Small file-like adapter with signed-URL refresh on expiry."""

    def __init__(self, endpoint: str, size: int):
        self.endpoint = endpoint
        self.size = size
        self.position = 0
        self._refresh()

    def _refresh(self) -> None:
        response = requests.get(self.endpoint, allow_redirects=False, timeout=60)
        response.raise_for_status()
        self.url = response.headers["Location"]

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.position = offset
        elif whence == 1:
            self.position += offset
        elif whence == 2:
            self.position = self.size + offset
        else:
            raise ValueError("invalid whence")
        return self.position

    def tell(self) -> int:
        return self.position

    def read(self, count: int = -1) -> bytes:
        if count < 0:
            count = self.size - self.position
        if count == 0:
            return b""
        end = min(self.position + count, self.size) - 1
        for _ in range(8):
            response = requests.get(
                self.url,
                headers={"Range": f"bytes={self.position}-{end}"},
                timeout=180,
            )
            if response.status_code in (200, 206):
                payload = response.content
                self.position += len(payload)
                return payload
            if response.status_code == 403:
                self._refresh()
                continue
            response.raise_for_status()
        raise RuntimeError("signed Figshare URL repeatedly expired")

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return True


def _metadata(
    cache_path: Path = DEFAULT_CACHE,
    endpoint: str = FILE_URL,
    file_size: int = FILE_BYTES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        return (
            cached["interventions"],
            cached["variable_names"],
            np.asarray([], dtype=str),
            int(cached["offset"]),
            int(cached["row_stride"]),
        )
    remote = _RangeFile(endpoint, file_size)
    with h5py.File(remote, "r", driver="fileobj") as handle:
        categories = np.asarray([item.decode() for item in handle["obs/__categories/gene"][:]])
        intervention_codes = handle["obs/gene"][:]
        interventions = categories[intervention_codes]
        variable_names = np.asarray(
            [item.decode() for item in handle["var/__categories/gene_name"][:]]
        )
        dataset = handle["X"]
        offset = int(dataset.id.get_offset())
        row_stride = int(dataset.shape[1] * dataset.dtype.itemsize)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        interventions=interventions,
        variable_names=variable_names,
        offset=np.asarray(offset),
        row_stride=np.asarray(row_stride),
    )
    return interventions, variable_names, categories, offset, row_stride


def _read_rows(
    rows: Iterable[int],
    offset: int,
    row_stride: int,
    workers: int = 24,
    endpoint: str = FILE_URL,
) -> dict[int, np.ndarray]:
    row_list = sorted({int(row) for row in rows})
    output: dict[int, np.ndarray] = {}

    def read_one(row: int, signed_url: str) -> tuple[int, np.ndarray] | None:
        start = offset + row * row_stride
        end = start + row_stride - 1
        try:
            response = requests.get(
                signed_url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=30,
            )
            if response.status_code == 206 and len(response.content) == row_stride:
                return row, np.frombuffer(response.content, dtype=np.float32).copy()
        except requests.RequestException:
            pass
        return None

    for start in range(0, len(row_list), workers):
        pending = row_list[start : start + workers]
        for _ in range(6):
            response = requests.get(endpoint, allow_redirects=False, timeout=30)
            response.raise_for_status()
            signed_url = response.headers["Location"]
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(pending)) as pool:
                results = list(pool.map(partial(read_one, signed_url=signed_url), pending))
            pending = []
            for result in results:
                if result is None:
                    continue
                row, value = result
                output[row] = value
            pending = [row for row in row_list[start : start + workers] if row not in output]
            if not pending:
                break
        if pending:
            raise RuntimeError(f"failed to fetch rows after retries: {pending[:5]}")
        completed = min(start + workers, len(row_list))
        if completed % 96 == 0 or completed == len(row_list):
            print(f"fetched {completed}/{len(row_list)} rows", flush=True)
    return output


def _bh(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def extract(
    output: Path = DEFAULT_OUTPUT,
    metadata_output: Path = DEFAULT_METADATA,
    sources: int = 35,
    targets_per_source: int = 20,
    cells_per_source: int = 80,
    control_cells: int = 300,
    seed: int = 20260826,
    metadata_cache: Path = DEFAULT_CACHE,
    subset_output: Path = DEFAULT_SUBSET,
    endpoint: str = FILE_URL,
    file_size: int = FILE_BYTES,
    chip_path: Path = CHIP_PATH,
    dataset_name: str = "CausalBench / Weissmann K562 Perturb-seq",
    source_ranking: str = "intervention_count",
    fixed_sources: Iterable[str] | None = None,
    range_workers: int = 24,
) -> pd.DataFrame:
    interventions, variable_names, _, offset, row_stride = _metadata(
        metadata_cache, endpoint=endpoint, file_size=file_size
    )
    intervention_counts = Counter(interventions.tolist())
    variable_set = set(variable_names.tolist())
    chip = pd.read_csv(chip_path, usecols=["source", "target", "weight"]).dropna()
    candidates: list[tuple[str, int, pd.DataFrame]] = []
    for source, group in chip.groupby("source", sort=True):
        if source not in intervention_counts or source not in variable_set:
            continue
        targets = group[group["target"].isin(variable_set)].sort_values(
            ["weight", "target"], ascending=[False, True]
        ).head(targets_per_source)
        if intervention_counts[source] >= cells_per_source and len(targets) > 0:
            candidates.append((source, intervention_counts[source], targets))
    if source_ranking == "intervention_count":
        candidates.sort(key=lambda item: (-item[1], item[0]))
    elif source_ranking == "candidate_count":
        candidates.sort(key=lambda item: (-len(item[2]), -item[1], item[0]))
    else:
        raise ValueError("source_ranking must be intervention_count or candidate_count")
    if fixed_sources is not None:
        fixed = tuple(dict.fromkeys(str(source) for source in fixed_sources))
        if not fixed:
            raise ValueError("fixed_sources must be non-empty when supplied")
        by_source = {source: (source, count, targets) for source, count, targets in candidates}
        missing = [source for source in fixed if source not in by_source]
        if missing:
            raise RuntimeError(
                "predeclared intervention sources are ineligible for this dataset: "
                f"{missing}"
            )
        candidates = [by_source[source] for source in fixed]
    else:
        candidates = candidates[:sources]
        if len(candidates) < 10:
            raise RuntimeError(f"only {len(candidates)} eligible intervention sources found")

    rng = np.random.RandomState(seed)
    row_groups: dict[str, np.ndarray] = {}
    for source, _, _ in candidates:
        available = np.flatnonzero(interventions == source)
        row_groups[source] = np.sort(rng.choice(available, size=cells_per_source, replace=False))
    controls = np.flatnonzero(interventions == "non-targeting")
    controls = np.sort(rng.choice(controls, size=control_cells, replace=False))
    values = _read_rows(
        np.concatenate([controls, *row_groups.values()]),
        offset,
        row_stride,
        workers=range_workers,
        endpoint=endpoint,
    )
    name_to_col = {name: idx for idx, name in enumerate(variable_names)}
    control_matrix = np.vstack([values[int(row)] for row in controls])
    # Counts are normalized per cell and log-transformed, matching CausalBench.
    control_matrix = np.log1p(control_matrix / np.maximum(control_matrix.sum(axis=1, keepdims=True), 1.0) * 1e4)
    selected_names = sorted({
        str(target)
        for _, _, targets in candidates
        for target in targets["target"].tolist()
    } | {source for source, _, _ in candidates})
    selected_columns = np.asarray(
        [name_to_col[name] for name in selected_names], dtype=int
    )
    subset_source_rows: list[np.ndarray] = []
    subset_source_names: list[str] = []
    subset_control = control_matrix[:, selected_columns]
    records: list[dict[str, object]] = []
    for source, count, targets in candidates:
        source_rows = row_groups[source]
        source_matrix = np.vstack([values[int(row)] for row in source_rows])
        source_matrix = np.log1p(source_matrix / np.maximum(source_matrix.sum(axis=1, keepdims=True), 1.0) * 1e4)
        subset_source_rows.append(source_matrix[:, selected_columns])
        subset_source_names.extend([source] * len(source_matrix))
        p_values = []
        temporary: list[dict[str, object]] = []
        for target, weight in targets[["target", "weight"]].itertuples(index=False):
            col = name_to_col[str(target)]
            effect = float(np.mean(source_matrix[:, col]) - np.mean(control_matrix[:, col]))
            p_value = float(stats.ttest_ind(source_matrix[:, col], control_matrix[:, col], equal_var=False).pvalue)
            temporary.append({
                "source": source,
                "target": str(target),
                "chip_weight": float(weight),
                "intervention_cells": int(count),
                "sampled_intervention_cells": int(cells_per_source),
                "sampled_control_cells": int(control_cells),
                "log1p_effect": effect,
                "p_value": p_value,
            })
            p_values.append(p_value)
        q_values = _bh(np.asarray(p_values, dtype=float))
        for record, q_value in zip(temporary, q_values):
            record["q_value"] = float(q_value)
            record["label"] = int(
                float(q_value) <= 0.05 and abs(float(record["log1p_effect"])) >= 0.10
            )
            records.append(record)
    frame = pd.DataFrame.from_records(records)
    frame = frame.sort_values(["source", "chip_weight", "target"], ascending=[True, False, True])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    subset_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        subset_output,
        gene_names=np.asarray(selected_names),
        control_expression=subset_control.astype(np.float32),
        intervention_expression=np.vstack(subset_source_rows).astype(np.float32),
        intervention_source=np.asarray(subset_source_names),
        seed=np.asarray(seed),
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata = {
        "dataset": dataset_name,
        "source_url": endpoint,
        "source_file_bytes": file_size,
        "chipseq_path": str(chip_path),
        "chipseq_sha256": hashlib.sha256(chip_path.read_bytes()).hexdigest(),
        "seed": seed,
        "n_sources": len(candidates),
        "n_rows": len(frame),
        "n_positive_labels": int(frame["label"].sum()),
        "sampling": {
            "targets_per_source": targets_per_source,
            "cells_per_source": cells_per_source,
            "control_cells": control_cells,
            "source_ranking": source_ranking,
            "fixed_sources": [source for source, _, _ in candidates],
            "range_workers": range_workers,
        },
        "label_rule": "BH q <= 0.05 and absolute log1p normalized expression effect >= 0.10; non-significant edges remain unknown, not negative",
        "csv_sha256": digest,
        "subset_path": str(subset_output),
        "subset_sha256": hashlib.sha256(subset_output.read_bytes()).hexdigest(),
        "subset_gene_count": len(selected_names),
    }
    metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--sources", type=int, default=35)
    parser.add_argument("--targets-per-source", type=int, default=20)
    parser.add_argument("--cells-per-source", type=int, default=80)
    parser.add_argument("--control-cells", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--metadata-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--subset-output", type=Path, default=DEFAULT_SUBSET)
    args = parser.parse_args()
    extract(**vars(args))
