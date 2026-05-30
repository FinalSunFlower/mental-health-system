import sys
import os
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("EXPERIMENT 4: LLM-based Lazarus Appraisal")
print("=" * 60)

from app.experiments.exp4_llm_appraisal import run_exp4

try:
    results = run_exp4(
        dataset="daic_woz",
        use_lazarus_constraints=True,
        n_reflection_steps=3,
        max_samples=10,
        model_name=r"D:\Models\huggingface\Qwen3.5-2B",
    )

    print("\n--- Dataset Info ---")
    di = results["dataset_info"]
    print("  Dataset:", di["name"])
    print("  N interviews:", di["n_interviews"])
    print("  N with PHQ-8:", di["n_with_phq8"])

    if "full_chain" in results:
        fc = results["full_chain"]
        aa = fc["appraisal_accuracy"]
        dl = fc["depression_level_accuracy"]
        print("\n--- Full Lazarus Chain ---")
        print("  Overall accuracy:       %.4f" % aa["overall_accuracy"])
        print("  Primary appraisal acc:  %.4f" % aa["primary_appraisal_accuracy"])
        print("  Secondary appraisal acc:%.4f" % aa["secondary_appraisal_accuracy"])
        print("  Reappraisal accuracy:   %.4f" % aa["reappraisal_accuracy"])
        print("  Distortion F1:          %.4f" % aa["distortion_detection_f1"])
        print("  Depression level acc:   %.4f" % dl["accuracy"])
        print("  Depression level kappa: %.4f" % dl["kappa"])

    if "primary_only" in results:
        po = results["primary_only"]
        aa_po = po["appraisal_accuracy"]
        dl_po = po["depression_level_accuracy"]
        print("\n--- Primary Only (No Lazarus Chain) ---")
        print("  Overall accuracy:       %.4f" % aa_po["overall_accuracy"])
        print("  Depression level acc:   %.4f" % dl_po["accuracy"])
        print("  Depression level kappa: %.4f" % dl_po["kappa"])

    if "comparison" in results:
        comp = results["comparison"]["lazarus_chain_improvement"]
        print("\n--- Lazarus Chain Improvement ---")
        print("  Overall accuracy delta:  %+.4f" % comp["overall_accuracy"])
        print("  Distortion F1 delta:     %+.4f" % comp["distortion_f1"])
        print("  Depression level acc:    %+.4f" % comp["depression_level_accuracy"])
        print("  Depression level kappa:  %+.4f" % comp["depression_level_kappa"])

    if "pearson_correlation" in results:
        pc = results["pearson_correlation"]
        print("\n--- Pearson Correlation ---")
        for key, val in pc.items():
            print(f"  {key}: r={val['r']:.4f}, p={val['p']:.4f}")

    if "icc" in results:
        icc = results["icc"]
        print("\n--- ICC ---")
        for key, val in icc.items():
            print(f"  {key}: {val}")

    print("\nExperiment 4 COMPLETED SUCCESSFULLY!")

except Exception as e:
    traceback.print_exc()
    print("\nEXPERIMENT FAILED: %s" % e)
