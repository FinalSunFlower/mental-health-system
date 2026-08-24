"""Freeze high-curation OmniPath directions that are measurable in RPE1."""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/causalbench"
URL = (
    "https://omnipathdb.org/interactions?format=tsv&genesymbols=1"
    "&datasets=omnipath&fields=sources,references,curation_effort"
    "&license=academic"
)
DEFAULT_OUTPUT = RAW / "rpe1_omnipath_source.csv"
DEFAULT_METADATA = RAW / "rpe1_omnipath_source.meta.json"
RPE1_CACHE = RAW / "rpe1_h5ad_metadata.npz"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(
    output: Path = DEFAULT_OUTPUT,
    metadata_output: Path = DEFAULT_METADATA,
    min_curation_effort: int = 1,
) -> pd.DataFrame:
    if not RPE1_CACHE.exists():
        raise FileNotFoundError(
            "RPE1 metadata missing; run backend/download_rpe1_labels.py first"
        )
    cached = np.load(RPE1_CACHE, allow_pickle=False)
    intervention_set = set(cached["interventions"].astype(str).tolist())
    variable_set = set(cached["variable_names"].astype(str).tolist())

    response = requests.get(URL, timeout=240)
    response.raise_for_status()
    raw = pd.read_csv(io.StringIO(response.text), sep="\t")
    selected = raw[
        raw["is_directed"].astype(bool)
        & raw["consensus_direction"].astype(bool)
        & (raw["curation_effort"].fillna(0).astype(int) >= min_curation_effort)
        & raw["source_genesymbol"].isin(intervention_set & variable_set)
        & raw["target_genesymbol"].isin(variable_set)
        & (raw["source_genesymbol"] != raw["target_genesymbol"])
    ].copy()
    frame = (
        selected.groupby(["source_genesymbol", "target_genesymbol"], as_index=False)
        .agg(weight=("curation_effort", "max"))
        .rename(columns={
            "source_genesymbol": "source",
            "target_genesymbol": "target",
        })
        .sort_values(["source", "weight", "target"], ascending=[True, False, True])
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    metadata = {
        "resource": "OmniPath curated directed interaction network",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_url": URL,
        "selection": {
            "is_directed": True,
            "consensus_direction": True,
            "min_curation_effort": min_curation_effort,
            "source_is_rpe1_intervention_and_measured": True,
            "target_is_rpe1_measured": True,
            "used_intervention_response_labels_for_selection": False,
        },
        "n_intervention_names": len(intervention_set),
        "n_variable_names": len(variable_set),
        "n_source_genes": int(frame["source"].nunique()),
        "n_interactions": int(len(frame)),
        "csv_sha256": _sha256(output),
        "rpe1_metadata_sha256": _sha256(RPE1_CACHE),
    }
    metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame


if __name__ == "__main__":
    result = download()
    print(f"wrote {len(result)} OmniPath-RPE1 interactions to {DEFAULT_OUTPUT}")
