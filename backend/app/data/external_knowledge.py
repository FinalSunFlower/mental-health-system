"""Load the frozen, non-redistributive external-prior snapshot."""

import json
from pathlib import Path

from app.cuspnet.projection import TheoryConstraint


def load_omnipath_sachs_prior(path: Path | None = None):
    path = Path(path) if path is not None else Path(__file__).resolve().parents[2] / "resources" / "omnipath_sachs_prior.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    constraints = []
    for interaction in payload.get("interactions", []):
        if not interaction.get("is_stimulation", False):
            continue
        source = interaction.get("source_genesymbol")
        target = interaction.get("target_genesymbol")
        if not source or not target or source == target:
            continue
        constraints.append(TheoryConstraint(
            cause=source,
            effect=target,
            theory="omnipath_consensus",
            confidence=1.0,
            citation="OmniPath frozen snapshot",
        ))
    metadata = {
        key: value for key, value in payload.items() if key != "interactions"
    }
    return constraints, metadata
