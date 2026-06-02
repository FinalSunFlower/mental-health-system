import sys
import os
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("EXPERIMENT 4: Unconstrained LLM vs Lazarus-Constrained LLM")
print("=" * 60)

from app.experiments.exp4_llm_appraisal import run_exp4

_MODEL_NAME = os.environ.get(
    "CUSPNET_LLM_MODEL",
    r"D:\Models\huggingface\Qwen3.5-2B",
)

try:
    results = run_exp4(
        dataset="erisk",
        max_samples=80,
        model_name=_MODEL_NAME,
    )

    print("\n--- Dataset Info ---")
    di = results["dataset_info"]
    print("  Dataset:", di["name"])
    print("  N samples:", di["n_samples"])
    print("  Label distribution:", di["label_distribution"])

    if "lazarus_constrained" in results:
        lc = results["lazarus_constrained"]
        dl = lc["depression_level_accuracy"]
        print("\n--- Lazarus-Constrained LLM ---")
        print("  Depression level acc:   %.4f" % dl["accuracy"])
        print("  Depression level kappa: %.4f" % dl["kappa"])
        if "accuracy_ci" in lc:
            ci = lc["accuracy_ci"]
            print("  Accuracy 95%% CI: [%.4f, %.4f]" % (ci.get("lower", 0), ci.get("upper", 0)))

    if "unconstrained" in results:
        uc = results["unconstrained"]
        dl = uc["depression_level_accuracy"]
        print("\n--- Unconstrained LLM (no theory) ---")
        print("  Depression level acc:   %.4f" % dl["accuracy"])
        print("  Depression level kappa: %.4f" % dl["kappa"])

    if "comparison" in results:
        comp = results["comparison"]["lazarus_vs_unconstrained"]
        print("\n--- Lazarus Framework Effect ---")
        print("  Accuracy delta:  %+.4f" % comp["accuracy_delta"])
        print("  Kappa delta:     %+.4f" % comp["kappa_delta"])
        print("  %s" % results["comparison"]["interpretation"])

    if "pearson_correlation" in results:
        pc = results["pearson_correlation"]
        print("\n--- Pearson Correlation ---")
        for key, val in pc.items():
            print("  %s: r=%.4f, p=%.6f" % (key, val["r"], val["p"]))

    if "binary_classification" in results:
        bc = results["binary_classification"]
        print("\n--- Binary Classification (Clinical vs Non-clinical) ---")
        if "lazarus_constrained" in bc:
            lc_bin = bc["lazarus_constrained"]
            print("  Lazarus:    acc=%.4f F1=%.4f prec=%.4f rec=%.4f AUC=%.4f" % (
                lc_bin["accuracy"], lc_bin["f1"], lc_bin["precision"], lc_bin["recall"], lc_bin.get("auc", 0)))
        if "unconstrained" in bc:
            uc_bin = bc["unconstrained"]
            print("  Unconstr:   acc=%.4f F1=%.4f prec=%.4f rec=%.4f AUC=%.4f" % (
                uc_bin["accuracy"], uc_bin["f1"], uc_bin["precision"], uc_bin["recall"], uc_bin.get("auc", 0)))

    print("\nExperiment 4 COMPLETED SUCCESSFULLY!")

except Exception as e:
    traceback.print_exc()
    print("\nEXPERIMENT FAILED: %s" % e)
