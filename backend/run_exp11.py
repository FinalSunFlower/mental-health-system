"""Run the independent finite-sample gate audit."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.experiments.exp11_finite_sample_gate_audit import run_exp11


def main() -> None:
    result = run_exp11()
    output = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        f"exp11_finite_sample_gate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["aggregate"], indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
