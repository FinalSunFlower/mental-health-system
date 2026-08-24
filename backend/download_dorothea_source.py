"""Download the independent DoRothEA A/B/C source snapshot from OmniPath."""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import requests

URL = (
    "https://omnipathdb.org/interactions/?genesymbols=1&datasets=dorothea"
    "&dorothea_levels=A,B,C&fields=dorothea_level&license=academic"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "data/raw/causalbench/dorothea_abc.tsv"
)


def download(output: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    response = requests.get(URL, timeout=180)
    response.raise_for_status()
    raw = pd.read_csv(io.StringIO(response.text), sep="\t")
    raw["dorothea_level"] = raw["dorothea_level"].str.split(";").str[0]
    raw = raw[raw["dorothea_level"].isin(["A", "B", "C"])].copy()
    raw["weight"] = 1.0
    raw.loc[raw["consensus_inhibition"].astype(bool), "weight"] = -1.0
    frame = raw[[
        "source_genesymbol", "target_genesymbol", "weight", "dorothea_level",
    ]].rename(columns={
        "source_genesymbol": "source",
        "target_genesymbol": "target",
        "dorothea_level": "confidence",
    })
    frame = frame.drop_duplicates(["source", "target", "confidence"])
    frame = frame.sort_values(["source", "target", "confidence"])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, sep="\t", index=False)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = download(args.output)
    print(f"wrote {len(result)} DoRothEA A/B/C interactions to {args.output}")
