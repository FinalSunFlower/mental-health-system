"""Run the cross-study JOBS II to Project STAR design-source transfer."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.experiments import run_exp12


def main() -> None:
    result = run_exp12()
    output = Path(__file__).resolve().parent / "results" / (
        f"exp12_cross_study_design_transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
