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
        dataset="erisk",
        use_lazarus_constraints=True,
        n_reflection_steps=3,
        max_samples=80,
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

    if "binary_classification" in results:
        bc = results["binary_classification"]
        print("\n--- Binary Classification (Clinical vs Non-clinical) ---")
        fc_bin = bc["full_chain"]
        po_bin = bc["primary_only"]
        print(f"  Full Chain:  acc={fc_bin['accuracy']:.4f} F1={fc_bin['f1']:.4f} prec={fc_bin['precision']:.4f} rec={fc_bin['recall']:.4f} AUC={fc_bin.get('auc',0):.4f} thresh={fc_bin.get('optimal_threshold',0.3):.2f}")
        print(f"  Primary Only: acc={po_bin['accuracy']:.4f} F1={po_bin['f1']:.4f} prec={po_bin['precision']:.4f} rec={po_bin['recall']:.4f} AUC={po_bin.get('auc',0):.4f} thresh={po_bin.get('optimal_threshold',0.3):.2f}")
        print(f"  N samples: {bc['n_samples']} (pos={bc['n_positive']}, neg={bc['n_negative']})")

    if "pearson_correlation" in results:
        pc = results["pearson_correlation"]
        print("\n--- Pearson Correlation ---")
        for key, val in pc.items():
            print(f"  {key}: r={val['r']:.4f}, p={val['p']:.4f}")

    if "composite_correlation" in results:
        cc = results["composite_correlation"]
        print("\n--- Composite Correlation ---")
        for key, val in cc.items():
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
