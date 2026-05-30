import sys
import os
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("EXPERIMENT 2: Cusp Dynamics Fitting")
print("=" * 60)

from app.experiments.exp2_cusp_fitting import run_exp2

try:
    results = run_exp2(dataset="kossakowski", window_size=7)

    print("\n--- Dataset Info ---")
    di = results["dataset_info"]
    print("  Dataset:", di["name"])
    print("  Observations:", di["n_observations"])
    print("  Variables:", di["n_variables"])
    print("  Window size:", results["window_size"])
    print("  N windows:", results["n_windows"])

    print("\n--- Model Fitting Results ---")
    fits = results["fits"]
    for model_name in ["linear", "logistic", "cusp"]:
        f = fits[model_name]
        print(f"\n  {model_name.upper()}:")
        print(f"    AIC: {f['aic']:.4f}")
        print(f"    BIC: {f['bic']:.4f}")
        print(f"    Params: {f['params']}")

    print("\n--- Best Model ---")
    bm = results["best_model"]
    print(f"  By AIC: {bm['aic']}")
    print(f"  By BIC: {bm['bic']}")

    print("\n--- Pseudo R-squared ---")
    r2 = results["pseudo_r2"]
    for name, val in r2.items():
        print(f"  {name}: {val:.4f}")

    print("\n--- Cusp Prediction Accuracy ---")
    pa = results["prediction_accuracy"]
    for name, val in pa.items():
        print(f"  {name}: {val:.4f}")

    cusp_aic = fits["cusp"]["aic"]
    linear_aic = fits["linear"]["aic"]
    logistic_aic = fits["logistic"]["aic"]
    delta_aic_lin = linear_aic - cusp_aic
    delta_aic_log = logistic_aic - cusp_aic
    print(f"\n  Delta AIC (linear - cusp): {delta_aic_lin:.4f}")
    print(f"  Delta AIC (logistic - cusp): {delta_aic_log:.4f}")
    if cusp_aic < linear_aic and cusp_aic < logistic_aic:
        print("  => CUSP model is BEST by AIC!")
    elif cusp_aic < linear_aic:
        print("  => CUSP beats Linear but not Logistic")
    else:
        print("  => CUSP does NOT beat alternatives")

    cusp_bic = fits["cusp"]["bic"]
    linear_bic = fits["linear"]["bic"]
    logistic_bic = fits["logistic"]["bic"]
    delta_bic_lin = linear_bic - cusp_bic
    delta_bic_log = logistic_bic - cusp_bic
    print(f"\n  Delta BIC (linear - cusp): {delta_bic_lin:.4f}")
    print(f"  Delta BIC (logistic - cusp): {delta_bic_log:.4f}")
    if cusp_bic < linear_bic and cusp_bic < logistic_bic:
        print("  => CUSP model is BEST by BIC!")
    elif cusp_bic < linear_bic:
        print("  => CUSP beats Linear but not Logistic by BIC")
    else:
        print("  => CUSP does NOT beat alternatives by BIC")

    print("\nExperiment 2 COMPLETED SUCCESSFULLY!")

except Exception as e:
    traceback.print_exc()
    print("\nEXPERIMENT FAILED: %s" % e)
