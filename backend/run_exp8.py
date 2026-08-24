"""Run the frozen external-prior Sachs diagnostic."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.experiments.exp8_external_prior_transfer import run_exp8


def main() -> None:
    result = run_exp8()
    output = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results",
        f"exp8_external_prior_transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
