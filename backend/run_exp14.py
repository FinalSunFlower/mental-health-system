"""Run the full-factorial RCEP robustness audit."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from app.experiments import run_exp14


def main() -> None:
    result = run_exp14()
    path = Path(__file__).resolve().parent / "results" / (
        f"exp14_hyperparameter_robustness_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
