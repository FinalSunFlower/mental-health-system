"""Run the context-conditioned RPE1 admission audit."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.experiments import run_exp21


if __name__ == "__main__":
    print(json.dumps(run_exp21(), indent=2))

