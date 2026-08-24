"""Run the known-stratum mixture-shift admission audit."""
import json

from app.experiments import run_exp18


if __name__ == "__main__":
    print(json.dumps(run_exp18(), indent=2))
