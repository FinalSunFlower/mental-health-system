"""Run the risk-controlled contract projection benchmark."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.experiments.exp7_risk_controlled_projection import run_exp7


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--variables", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    result = run_exp7(
        n_repeats=args.repeats,
        n_samples=args.samples,
        n_variables=args.variables,
    )
    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        f"exp7_risk_controlled_projection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["aggregate"], indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
