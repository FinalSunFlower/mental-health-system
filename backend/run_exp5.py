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
    results = run_exp5(dataset="nhanes", n_folds=3, random_state=42, max_samples=500)

    print("\n--- Dataset Info ---")
    di = results["dataset_info"]
    print("  Dataset:", di["name"])
    print("  N samples:", di["n_samples"])
    print("  N features:", di["n_features"])
    print("  Prevalence: %.2f%%" % (di["prevalence"] * 100))

    print("\n--- Method Comparison ---")
    comparison = results.get("comparison_table", {})
    for method, metrics in comparison.items():
        print(f"\n  {method}:")
        for k, v in metrics.items():
            print(f"    {k}: {v}")

    cv_results = results.get("cv_results", {})
    if cv_results:
        print("\n--- Cross-Validation Results ---")
        for method, metrics in cv_results.items():
            if "note" in metrics:
                continue
            print(f"\n  {method}:")
            for metric_name, vals in metrics.items():
                if isinstance(vals, dict) and "mean" in vals:
                    print(f"    {metric_name}: {vals['mean']:.4f} +/- {vals['std']:.4f}")

    sig = results.get("statistical_significance", {})
    if sig:
        print("\n--- Statistical Significance (vs CuspNet) ---")
        for method, info in sig.items():
            if method == "multiple_comparison":
                mc = info
                bonf = mc.get("bonferroni", {})
                fdr = mc.get("fdr_bh", {})
                print(f"  Multiple Comparison Correction:")
                if isinstance(bonf, dict):
                    print(f"    Bonferroni: significant={bonf.get('significant', 'N/A')}")
                if isinstance(fdr, dict):
                    print(f"    FDR-BH: significant={fdr.get('significant', 'N/A')}")
                continue
            delong = info.get("delong_summary", {})
            mcnemar = info.get("mcnemar_summary", {})
            effect = info.get("effect_size", {})
            print(f"  {method}:")
            if delong:
                print(f"    DeLong: mean_p={delong.get('mean_p_value', 'N/A'):.4f}, "
                      f"significant_folds={delong.get('n_significant_folds', 'N/A')}/{delong.get('n_total_folds', 'N/A')}, "
                      f"consistent={delong.get('significant_consistent', 'N/A')}")
            if mcnemar:
                print(f"    McNemar: mean_p={mcnemar.get('mean_p_value', 'N/A'):.4f}, "
                      f"significant_folds={mcnemar.get('n_significant_folds', 'N/A')}/{mcnemar.get('n_total_folds', 'N/A')}")
            if effect:
                auc_diff = effect.get('mean_auc_difference', 'N/A')
                cohens = effect.get('cohens_d', 'N/A')
                auc_diff_str = f"{auc_diff:.4f}" if isinstance(auc_diff, (int, float)) else str(auc_diff)
                cohens_str = f"{cohens:.3f}" if isinstance(cohens, (int, float)) else str(cohens)
                print(f"    Effect: mean_AUC_diff={auc_diff_str}, Cohen's_d={cohens_str}")

    proxy = results.get("proxy_validity", {})
    if proxy and "note" not in proxy:
        print("\n--- Proxy Validity ---")
        for k, v in proxy.items():
            if isinstance(v, dict):
                print(f"  {k}: r={v.get('r', 'N/A'):.3f}, p={v.get('p', 'N/A'):.4f}, valid={v.get('valid', 'N/A')}")

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
