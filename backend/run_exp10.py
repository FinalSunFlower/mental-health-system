"""Run the scalable DAG-order solver audit."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.experiments.exp10_solver_scaling import run_exp10


def main() -> None:
    result = run_exp10()
    output = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        f"exp10_solver_scaling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["aggregate"], indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
