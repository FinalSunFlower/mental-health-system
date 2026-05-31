import sys
import os
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("EXPERIMENT 3: Resilience Prediction")
print("=" * 60)

from app.experiments.exp3_resilience_prediction import run_exp3

try:
    results = run_exp3(dataset="studentlife", window_size=7)

    print("\n--- Dataset Info ---")
    di = results["dataset_info"]
    print("  Dataset:", di["name"])
    print("  Observations:", di["n_observations"])
    print("  Window size:", di["window_size"])

    print("\n--- Resilience Reserve (Delta V) Statistics ---")
    dv = results["delta_v_stats"]
    print("  Mean:", f"{dv['mean']:.4f}" if dv['mean'] != float('inf') else "N/A")
    print("  Std:", f"{dv['std']:.4f}")
    print("  Min:", f"{dv['min']:.4f}" if dv['min'] != float('inf') else "N/A")
    print("  Max:", f"{dv['max']:.4f}" if dv['max'] != float('inf') else "N/A")
    print("  N low resilience (delta_v < 0.5):", dv['n_low_resilience'])
    print("  N critical (delta_v < 0.1):", dv['n_critical'])

    if "delta_v_cusp_stats" in results:
        print("\n--- Cusp-Based Resilience Reserve (Delta V Cusp) ---")
        dvc = results["delta_v_cusp_stats"]
        print("  Mean:", f"{dvc['mean']:.4f}")
        print("  Std:", f"{dvc['std']:.4f}")
        print("  Min:", f"{dvc['min']:.4f}")
        print("  Max:", f"{dvc['max']:.4f}")
        print("  N low resilience (delta_v < 0.5):", dvc['n_low_resilience'])
        print("  N critical (delta_v < 0.1):", dvc['n_critical'])

    if "resilience_ratio_stats" in results:
        print("\n--- Resilience Ratio Statistics ---")
        rr = results["resilience_ratio_stats"]
        print("  Mean:", f"{rr['mean']:.4f}")
        print("  Std:", f"{rr['std']:.4f}")
        print("  Min:", f"{rr['min']:.4f}")
        print("  Max:", f"{rr['max']:.4f}")
        print("  N low (ratio < 0.3):", rr['n_low'])
        print("  N critical (ratio < 0.1):", rr['n_critical'])

    if "resilience_ews_stats" in results:
        print("\n--- EWS-Based Resilience Index ---")
        rews = results["resilience_ews_stats"]
        print("  Mean:", f"{rews['mean']:.4f}")
        print("  Std:", f"{rews['std']:.4f}")
        print("  Min:", f"{rews['min']:.4f}")
        print("  Max:", f"{rews['max']:.4f}")
        print("  N low (ews < 0.3):", rews['n_low'])
        print("  N critical (ews < 0.1):", rews['n_critical'])

    if "stability_stats" in results:
        print("\n--- Window Stability Statistics ---")
        ss = results["stability_stats"]
        print("  Mean:", f"{ss['mean']:.4f}")
        print("  Std:", f"{ss['std']:.4f}")
        print("  Min:", f"{ss['min']:.4f}")
        print("  Max:", f"{ss['max']:.4f}")

    if "cusp_params" in results:
        print("\n--- Fitted Cusp Parameters ---")
        cp = results["cusp_params"]
        print(f"  a = {cp['a']:.6f}")
        print(f"  b = {cp['b']:.6f}")
        print(f"  c = {cp['c']:.6f}")

    print("\n--- Tipping Point Detection ---")
    tp = results["tipping_points"]
    print("  Threshold sigma:", tp["threshold_sigma"])
    print("  N tipping points detected:", tp["n_detected"])
    if tp["n_detected"] > 0 and len(tp["indices"]) <= 20:
        print("  Indices:", tp["indices"])

    print("\n--- Early Warning Signals ---")
    ews = results["early_warning_signals"]
    print("  Variance trend:", f"{ews['variance_trend']:.6f}")
    print("  Variance increasing:", ews["variance_increasing"])
    print("  Autocorrelation trend:", f"{ews['autocorrelation_trend']:.6f}")
    print("  Autocorrelation increasing:", ews["autocorrelation_increasing"])

    if results["pre_tipping_delta_v"]["mean"] is not None:
        print("\n--- Pre-Tipping Delta V ---")
        pt = results["pre_tipping_delta_v"]
        print("  Mean:", f"{pt['mean']:.4f}")
        print("  Std:", f"{pt['std']:.4f}")

    if "roc_analysis" in results:
        print("\n--- ROC Analysis ---")
        roc = results["roc_analysis"]
        if "auc_roc" in roc:
            print("  AUC-ROC:", f"{roc['auc_roc']:.4f}")
            print("  Predictor:", roc.get("predictor", "N/A"))
            print("  N positive:", roc.get("n_positive", "N/A"))
            print("  N negative:", roc.get("n_negative", "N/A"))
            if "auc_ci" in roc:
                ci = roc["auc_ci"]
                print(f"  AUC 95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        elif "note" in roc:
            print("  Note:", roc["note"])

    if "cox_regression" in results:
        print("\n--- Cox Regression / Survival Analysis ---")
        cox = results["cox_regression"]
        if "error" in cox:
            print("  Error:", cox["error"])
        elif "note" in cox:
            print("  Note:", cox["note"])
            if "hazard_ratio_proxy" in cox:
                print("  Hazard ratio proxy:", f"{cox['hazard_ratio_proxy']:.4f}")
                print("  Coefficient:", f"{cox['coefficient']:.4f}")
                print("  AUC-ROC:", f"{cox['auc_roc']:.4f}")
        else:
            print("  Hazard ratio:", f"{cox['hazard_ratio']:.4f}")
            print("  P-value:", f"{cox['p_value']:.6f}")
            print("  Concordance index:", f"{cox['concordance_index']:.4f}")
            print("  Coefficient:", f"{cox['coefficient']:.4f}")

    print("\n--- Prospective Prediction ---")
    pp = results["prospective_prediction"]
    for k in [1, 3, 5, 7]:
        key = f"lookahead_{k}"
        if key in pp and isinstance(pp[key], dict) and "auc_roc_multifeature" in pp[key]:
            lk = pp[key]
            print(f"\n  Lookahead {k} steps:")
            print(f"    Delta-V Cusp AUC-ROC: {lk['auc_roc_delta_v']:.4f}")
            print(f"    Delta-V Cusp Accuracy: {lk['accuracy_delta_v']:.4f}")
            print(f"    Multi-feature AUC-ROC: {lk['auc_roc_multifeature']:.4f}")
            print(f"    Multi-feature Accuracy: {lk['accuracy_multifeature']:.4f}")
            print(f"    Multi-feature F1: {lk['f1_multifeature']:.4f}")
            print(f"    Multi-feature Precision: {lk['precision_multifeature']:.4f}")
            print(f"    Multi-feature Recall: {lk['recall_multifeature']:.4f}")
            if lk.get("cv_auc_roc_mean") is not None:
                print(f"    CV AUC-ROC: {lk['cv_auc_roc_mean']:.4f} +/- {lk['cv_auc_roc_std']:.4f}")
            if lk.get("cv_f1_mean") is not None:
                print(f"    CV F1: {lk['cv_f1_mean']:.4f} +/- {lk['cv_f1_std']:.4f}")
            print(f"    Point-biserial r: {lk['point_biserial_r']:.4f} (p={lk['point_biserial_p']:.6f})")
            print(f"    N samples: {lk['n_samples']}, N transitions: {lk['n_transitions']}")
            print(f"    Transition rate: {lk['transition_rate']:.4f}")
        elif key in pp and "note" in pp[key]:
            print(f"\n  Lookahead {k} steps: {pp[key]['note']}")

    if "best_lookahead" in pp and pp["best_lookahead"]["k"] is not None:
        bl = pp["best_lookahead"]
        print(f"\n  Best lookahead: k={bl['k']}, AUC-ROC={bl['auc_roc']:.4f}")

    print("\nExperiment 3 COMPLETED SUCCESSFULLY!")

except Exception as e:
    traceback.print_exc()
    print("\nEXPERIMENT FAILED: %s" % e)
