import sys
import os
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("EXPERIMENT 5: End-to-End Depression Prediction (NHANES)")
print("=" * 60)

from app.experiments.exp5_end_to_end import run_exp5

try:
    results = run_exp5(dataset="nhanes", test_size=0.2, random_state=42)

    print("\n--- Dataset Info ---")
    di = results["dataset_info"]
    print("  Dataset:", di["name"])
    print("  N samples:", di["n_samples"])
    print("  N features:", di["n_features"])
    print("  Prevalence: %.2f%%" % (di["prevalence"] * 100))

    print("\n--- Method Comparison ---")
    comparison = results.get("comparison", {})
    for method, metrics in comparison.items():
        print(f"\n  {method}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    if "best_method" in results:
        print(f"\n  Best method (by AUC): {results['best_method']}")

    print("\n--- Explainability ---")
    expl = results.get("explainability", {})
    for method, info in expl.items():
        print(f"  {method}: interpretability={info.get('interpretability_score', 'N/A')}, "
              f"causal={info.get('causal_explanation', 'N/A')}, "
              f"intervention={info.get('intervention_guidance', 'N/A')}")

    print("\n--- Intervention Quality ---")
    intv = results.get("intervention_quality", {})
    for method, info in intv.items():
        print(f"  {method}: quality={info.get('quality_score', 'N/A')}, "
              f"personalized={info.get('personalized', 'N/A')}, "
              f"theory={info.get('theory_based', 'N/A')}")

    print("\nExperiment 5 (NHANES) COMPLETED SUCCESSFULLY!")

except Exception as e:
    traceback.print_exc()
    print("\nEXPERIMENT FAILED: %s" % e)
