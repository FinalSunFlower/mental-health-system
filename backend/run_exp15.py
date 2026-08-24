"""Run the Tuebingen real-pair source-holdout audit."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.experiments import run_exp15


if __name__ == "__main__":
    output = run_exp15()
    print(json.dumps(output, ensure_ascii=False, indent=2))
