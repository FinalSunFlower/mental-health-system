"""Loader for the public Sachs protein-signalling benchmark.

The raw CSV is intentionally not redistributed.  This module only defines the
column order, deterministic parsing, and the reference adjacency used by the
paper's optional external-transfer diagnostics.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.data.paths import RAW_DIR


class SachsLoader:
    # bnlearn's published column order and the canonical paper variable order.
    BNLEARN_SACHS_COLUMNS = (
        "Raf", "Mek", "Plcg", "PIP2", "PIP3", "Erk", "Akt", "PKA", "Jnk", "p38", "PKC"
    )
    SACHS_VAR_NAMES = list(BNLEARN_SACHS_COLUMNS)

    # Established Sachs reference edges, represented in canonical variable order.
    REFERENCE_EDGES = (
        ("Raf", "Mek"), ("Mek", "Erk"),
        ("Plcg", "PIP2"), ("Plcg", "PIP3"), ("PIP2", "PIP3"),
        ("PIP3", "Akt"), ("Akt", "Erk"), ("PKA", "Raf"),
        ("PKA", "Jnk"), ("PKA", "p38"), ("Jnk", "p38"),
    )

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else RAW_DIR / "sachs_data.csv"

    def load(self):
        if not self.path.exists():
            raise FileNotFoundError(
                f"Sachs input not found at {self.path}. Acquire the public benchmark "
                "listed in DATA_AVAILABILITY.md; raw inputs are not bundled."
            )
        frame = pd.read_csv(self.path)
        missing = [name for name in self.BNLEARN_SACHS_COLUMNS if name not in frame.columns]
        if missing:
            raise ValueError(f"Sachs CSV is missing required columns: {missing}")
        X = frame.loc[:, self.BNLEARN_SACHS_COLUMNS].to_numpy(dtype=float)
        index = {name: i for i, name in enumerate(self.SACHS_VAR_NAMES)}
        adjacency = np.zeros((len(self.SACHS_VAR_NAMES), len(self.SACHS_VAR_NAMES)), dtype=float)
        for cause, effect in self.REFERENCE_EDGES:
            adjacency[index[cause], index[effect]] = 1.0
        return X, adjacency, list(self.SACHS_VAR_NAMES)
