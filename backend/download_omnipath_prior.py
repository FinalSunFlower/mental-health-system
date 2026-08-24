"""Freeze an auditable OmniPath prior for proteins in the Sachs benchmark."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


GENES = (
    "PLCG1", "PRKCA", "PRKACA", "RAF1", "MAP2K1",
    "MAPK1", "AKT1", "MAPK8", "MAPK14",
)


def _normalized_values(value: object) -> list[str]:
    """Return sorted, nonempty API values without splitting strings into characters."""
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]
    normalized: list[str] = []
    for item in values:
        normalized.extend(part.strip() for part in str(item).split(";") if part.strip())
    return sorted(set(normalized))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "resources" / "omnipath_sachs_prior.json",
    )
    parser.add_argument("--min-curation-effort", type=int, default=3)
    args = parser.parse_args()
    query = urlencode({
        "format": "json",
        "genesymbols": 1,
        "sources": ",".join(GENES),
        "targets": ",".join(GENES),
        "fields": "sources,references,curation_effort",
    })
    url = f"https://omnipathdb.org/interactions?{query}"
    request = Request(url, headers={"User-Agent": "CuspNet-research/1.0"})
    with urlopen(request, timeout=60) as response:
        raw = json.loads(response.read().decode("utf-8"))

    gene_set = set(GENES)
    interactions = []
    for row in raw:
        if row.get("source_genesymbol") not in gene_set:
            continue
        if row.get("target_genesymbol") not in gene_set:
            continue
        if not row.get("is_directed") or not row.get("consensus_direction"):
            continue
        if int(row.get("curation_effort") or 0) < args.min_curation_effort:
            continue
        interactions.append({
            "source_genesymbol": row["source_genesymbol"],
            "target_genesymbol": row["target_genesymbol"],
            "is_stimulation": bool(row.get("is_stimulation")),
            "is_inhibition": bool(row.get("is_inhibition")),
            "curation_effort": int(row.get("curation_effort") or 0),
            "sources": _normalized_values(row.get("sources")),
            "references": _normalized_values(row.get("references")),
        })
    interactions.sort(key=lambda row: (
        row["source_genesymbol"], row["target_genesymbol"]
    ))
    payload = {
        "resource": "OmniPath aggregated interaction database",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_url": url,
        "selection": {
            "genes": list(GENES),
            "is_directed": True,
            "consensus_direction": True,
            "min_curation_effort": args.min_curation_effort,
            "used_sachs_labels_for_selection": False,
        },
        "limitations": (
            "The current database can contain literature overlapping the Sachs "
            "reference. This snapshot is an external-transfer diagnostic, not an "
            "independent proof of causal recovery."
        ),
        "interactions": interactions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(interactions)} interactions to {args.output}")


if __name__ == "__main__":
    main()
