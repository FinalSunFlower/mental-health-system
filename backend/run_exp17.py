"""Run the CausalBench source-construction sensitivity audit."""
import json

from app.experiments import run_exp17


if __name__ == "__main__":
    print(json.dumps(run_exp17(), indent=2))
