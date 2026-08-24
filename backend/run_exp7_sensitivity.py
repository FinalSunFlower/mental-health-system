"""Run scale and calibration sensitivity for risk-controlled projection."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.experiments.exp7_risk_controlled_projection import (
    run_exp7_sensitivity,
    run_gate_calibration_sensitivity,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--gate-repeats", type=int, default=200)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    payload = {
        "scale_sensitivity": run_exp7_sensitivity(n_repeats=args.repeats),
        "gate_sensitivity": run_gate_calibration_sensitivity(
            n_repeats=args.gate_repeats
        ),
    }
    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        f"exp7_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["scale_sensitivity"]["aggregate"], indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
