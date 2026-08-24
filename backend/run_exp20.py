import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.experiments.exp20_rpe1_omnipath_holdout import run_exp20


if __name__ == "__main__":
    print(json.dumps(run_exp20(), indent=2, allow_nan=False))
