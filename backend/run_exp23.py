"""Run the K562-to-RPE1 context-representation validation."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.experiments import run_exp23


if __name__ == "__main__":
    print(json.dumps(run_exp23(), indent=2))

