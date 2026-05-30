import sys
import os
import traceback
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("EXPERIMENT 1: Causal Discovery")
print("=" * 60)

from app.experiments.exp1_causal_discovery import run_exp1

try:
    results = run_exp1(
        dataset="sachs", ebic_gamma=0.5, score_threshold=0.1, pc_alpha=0.01
    )

    print("\n--- Dataset Info ---")
    di = results["dataset_info"]
    print("  Dataset:", di["name"])
    print("  Samples:", di["n_samples"], "  Variables:", di["n_variables"])
    print("  Ground Truth:", di["has_ground_truth"])
    print("  Variables:", di["variable_names"])

    print("\n--- CuspNet Results ---")
    cn = results["cuspnet"]
    if "metrics" in cn:
        m = cn["metrics"]
        print("  Precision: %.4f" % m["precision"])
        print("  Recall:    %.4f" % m["recall"])
        print("  F1:        %.4f" % m["f1"])
        print("  SHD:       %d" % m["shd"])
        print("  SID:       %d" % m["sid"])
        print("  TP/FP/FN:  %d/%d/%d" % (m["tp"], m["fp"], m["fn"]))
    print("  Topo order:", cn["topological_order"])
    print("  Bridge symptoms:", cn["bridge_symptoms"])
    print("  Feedback loops: %d found" % len(cn["positive_feedback_loops"]))

    for method in ["pc", "ges", "notears", "ebicglasso_only", "score_only"]:
        if method in results:
            r = results[method]
            if "metrics" in r:
                m = r["metrics"]
                print("\n--- %s ---" % method.upper())
                print(
                    "  Precision: %.4f, Recall: %.4f, F1: %.4f, SHD: %d, SID: %d"
                    % (
                        m["precision"],
                        m["recall"],
                        m["f1"],
                        m["shd"],
                        m["sid"],
                    )
                )
            elif "error" in r:
                print("\n--- %s --- ERROR: %s" % (method.upper(), r["error"][:150]))

    if "comparison" in results:
        print("\n" + "=" * 60)
        print("COMPARISON TABLE")
        print("=" * 60)
        print("%-20s %10s %10s %10s %6s %6s" % ("Method", "Precision", "Recall", "F1", "SHD", "SID"))
        print("-" * 60)
        for method, m in results["comparison"].items():
            print(
                "%-20s %10.4f %10.4f %10.4f %6d %6d"
                % (
                    method,
                    m["precision"],
                    m["recall"],
                    m["f1"],
                    m["shd"],
                    m["sid"],
                )
            )

    print("\nExperiment 1 COMPLETED SUCCESSFULLY!")

except Exception as e:
    traceback.print_exc()
    print("\nEXPERIMENT FAILED: %s" % e)
