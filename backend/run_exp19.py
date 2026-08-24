import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.experiments.exp19_rpe1_intervention_holdout import run_exp19


if __name__ == "__main__":
    print(json.dumps(run_exp19(), indent=2, allow_nan=False))
