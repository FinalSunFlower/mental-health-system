"""Run the frozen cross-cell pairwise intervention-transfer audit."""
from __future__ import annotations

import json
import os
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.experiments.exp24_cross_cell_pairwise_transfer import run_exp24


if __name__ == "__main__":
    payload = run_exp24()
    output = Path(__file__).resolve().parent / "results" / "cross_cell_pairwise_transfer_audit.json"
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(output)
    print(hashlib.sha256(output.read_bytes()).hexdigest().upper())
