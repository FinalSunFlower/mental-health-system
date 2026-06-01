"""
Statistical analysis utilities for experimental validation.
Provides bootstrap confidence intervals, significance testing, multiple comparison
corrections (Bonferroni, FDR), McNemar test, DeLong ROC test, and effect sizes.
"""
import numpy as np
from typing import Dict, List
from scipy import stats


def bootstrap_ci(
    values: np.ndarray,
    statistic_fn=np.mean,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict:
    rng = np.random.RandomState(seed)
    n = len(values)
    if n < 3:
        stat = statistic_fn(values) if n > 0 else 0.0
        return {"estimate": float(stat), "ci_lower": float(stat), "ci_upper": float(stat), "n": n}

    boot_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = statistic_fn(sample)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    estimate = float(statistic_fn(values))

    return {
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "n": n,
    }


def bootstrap_ci_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn=None,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict:
    if metric_fn is None:
        from sklearn.metrics import roc_auc_score
        metric_fn = roc_auc_score

    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n < 10:
        metric_val = float(metric_fn(y_true, y_pred))
        return {"estimate": metric_val, "ci_lower": metric_val, "ci_upper": metric_val, "n": n}

    boot_stats = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            val = metric_fn(yt, y_pred[idx])
            boot_stats.append(val)
        except Exception:
            continue

    if len(boot_stats) < 100:
        metric_val = float(metric_fn(y_true, y_pred))
        return {"estimate": metric_val, "ci_lower": metric_val, "ci_upper": metric_val, "n": n}

    boot_stats = np.array(boot_stats)
    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    estimate = float(metric_fn(y_true, y_pred))

    return {
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence": confidence,
        "n_bootstrap": len(boot_stats),
        "n": n,
    }


def paired_significance_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    test_type: str = "auto",
) -> Dict:
    n = len(scores_a)
    if n < 3:
        return {
            "test": "insufficient_data",
            "p_value": 1.0,
            "significant_at_005": False,
            "n": n,
        }

    diff = scores_a - scores_b
    diff_mean = np.mean(diff)
    diff_std = np.std(diff, ddof=1)

    if test_type == "auto":
        try:
            _, p_normality = stats.shapiro(diff)
            test_type = "paired_t" if p_normality > 0.05 else "wilcoxon"
        except Exception:
            test_type = "wilcoxon"

    if test_type == "paired_t":
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        test_name = "paired_t_test"
        df = n - 1
    elif test_type == "wilcoxon":
        try:
            w_stat, p_value = stats.wilcoxon(scores_a, scores_b)
            test_name = "wilcoxon_signed_rank"
            df = None
        except ValueError:
            return {
                "test": "wilcoxon_failed",
                "p_value": 1.0,
                "significant_at_005": False,
                "n": n,
                "note": "All differences are zero",
            }
    else:
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        test_name = "paired_t_test"
        df = n - 1

    cohens_d = diff_mean / (diff_std + 1e-10) if diff_std > 1e-10 else 0.0

    return {
        "test": test_name,
        "p_value": float(p_value),
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01,
        "effect_size_cohens_d": float(cohens_d),
        "effect_size_interpretation": _interpret_cohens_d(cohens_d),
        "mean_diff": float(diff_mean),
        "std_diff": float(diff_std),
        "n": n,
        "df": df,
    }


def _interpret_cohens_d(d: float) -> str:
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict:
    n_tests = len(p_values)
    if n_tests == 0:
        return {"adjusted_p_values": [], "significant": [], "method": "bonferroni"}

    adjusted = [min(p * n_tests, 1.0) for p in p_values]
    significant = [p < alpha for p in adjusted]

    return {
        "adjusted_p_values": adjusted,
        "significant": significant,
        "n_tests": n_tests,
        "alpha": alpha,
        "method": "bonferroni",
    }


def fdr_correction(p_values: List[float], alpha: float = 0.05) -> Dict:
    n_tests = len(p_values)
    if n_tests == 0:
        return {"adjusted_p_values": [], "significant": [], "method": "benjamini_hochberg"}

    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    adjusted = np.zeros(n_tests)
    for i in range(n_tests - 1, -1, -1):
        if i == n_tests - 1:
            adjusted[i] = sorted_p[i] * n_tests / (i + 1)
        else:
            adjusted[i] = min(adjusted[i + 1], sorted_p[i] * n_tests / (i + 1))
    adjusted = np.clip(adjusted, 0, 1)

    final_adjusted = np.zeros(n_tests)
    final_adjusted[sorted_indices] = adjusted

    significant = final_adjusted < alpha

    return {
        "adjusted_p_values": final_adjusted.tolist(),
        "significant": significant.tolist(),
        "n_tests": n_tests,
        "alpha": alpha,
        "method": "benjamini_hochberg",
    }


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict:
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        return {"test": "mcnemar", "p_value": 1.0, "significant_at_005": False, "note": "no_discordant_pairs"}

    if b + c < 25:
        from scipy.stats import binom
        p_value = 2.0 * min(
            binom.cdf(min(b, c), b + c, 0.5),
            1.0 - binom.cdf(min(b, c) - 1, b + c, 0.5),
        )
        p_value = min(p_value, 1.0)
        exact = True
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1.0 - stats.chi2.cdf(chi2, df=1)
        exact = False

    return {
        "test": "mcnemar",
        "p_value": float(p_value),
        "significant_at_005": p_value < 0.05,
        "discordant_b": b,
        "discordant_c": c,
        "exact": exact,
    }


def delong_roc_test(y_true: np.ndarray, y_prob_a: np.ndarray, y_prob_b: np.ndarray) -> Dict:
    n = len(y_true)
    if n < 10:
        return {"test": "delong", "p_value": 1.0, "significant_at_005": False, "note": "insufficient_samples"}

    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos < 2 or n_neg < 2:
        return {"test": "delong", "p_value": 1.0, "significant_at_005": False, "note": "insufficient_class_samples"}

    def _compute_placement(probabilities, labels, is_model_a_for_V10):
        if is_model_a_for_V10:
            pos_probs = probabilities[labels == 1]
            neg_probs = probabilities[labels == 0]
            placements = np.zeros(len(neg_probs))
            for i, neg_p in enumerate(neg_probs):
                placements[i] = np.mean(pos_probs > neg_p) + 0.5 * np.mean(pos_probs == neg_p)
            return placements
        else:
            neg_probs = probabilities[labels == 0]
            pos_probs = probabilities[labels == 1]
            placements = np.zeros(len(pos_probs))
            for i, pos_p in enumerate(pos_probs):
                placements[i] = np.mean(neg_probs > pos_p) + 0.5 * np.mean(neg_probs == pos_p)
            return placements

    V10_a = _compute_placement(y_prob_a, y_true, True)
    V01_a = _compute_placement(y_prob_a, y_true, False)
    V10_b = _compute_placement(y_prob_b, y_true, True)
    V01_b = _compute_placement(y_prob_b, y_true, False)

    auc_a = 1.0 - np.mean(V01_a)
    auc_b = 1.0 - np.mean(V01_b)

    S10_a = np.cov(np.column_stack([V10_a, V10_b]), rowvar=False) if len(V10_a) > 1 else np.zeros((2, 2))
    S01_a = np.cov(np.column_stack([V01_a, V01_b]), rowvar=False) if len(V01_a) > 1 else np.zeros((2, 2))

    S = S10_a / n_neg + S01_a / n_pos

    try:
        diff = np.array([auc_a - auc_b])
        if S[0, 0] + S[1, 1] - 2 * S[0, 1] <= 0:
            return {"test": "delong", "p_value": 1.0, "significant_at_005": False,
                    "auc_a": float(auc_a), "auc_b": float(auc_b), "note": "non_positive_variance"}
        var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
        z_stat = (auc_a - auc_b) / np.sqrt(var_diff)
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
    except Exception:
        return {"test": "delong", "p_value": 1.0, "significant_at_005": False, "note": "computation_failed"}

    return {
        "test": "delong",
        "p_value": float(p_value),
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01,
        "z_statistic": float(z_stat),
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "auc_difference": float(auc_a - auc_b),
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> Dict:
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return {"d": 0.0, "interpretation": "insufficient_data"}

    mean_diff = np.mean(a) - np.mean(b)
    pooled_std = np.sqrt(((n_a - 1) * np.std(a, ddof=1) ** 2 + (n_b - 1) * np.std(b, ddof=1) ** 2) / (n_a + n_b - 2))
    d = mean_diff / (pooled_std + 1e-10)

    abs_d = abs(d)
    if abs_d < 0.2:
        interp = "negligible"
    elif abs_d < 0.5:
        interp = "small"
    elif abs_d < 0.8:
        interp = "medium"
    else:
        interp = "large"

    return {"d": float(d), "interpretation": interp}


def compute_full_statistical_report(
    method_scores: Dict[str, np.ndarray],
    reference_method: str = None,
    alpha: float = 0.05,
) -> Dict:
    report = {}
    method_names = list(method_scores.keys())

    for name, scores in method_scores.items():
        report[name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=1)),
            "n": len(scores),
            "ci_95": bootstrap_ci(scores, confidence=0.95),
        }

    if reference_method is None and len(method_names) > 0:
        sorted_methods = sorted(method_names, key=lambda n: np.mean(method_scores[n]), reverse=True)
        reference_method = sorted_methods[0]

    if reference_method and reference_method in method_scores:
        ref_scores = method_scores[reference_method]
        pairwise = {}
        p_values = []
        for name in method_names:
            if name == reference_method:
                continue
            other_scores = method_scores[name]
            n_min = min(len(ref_scores), len(other_scores))
            result = paired_significance_test(
                ref_scores[:n_min], other_scores[:n_min]
            )
            pairwise[f"{reference_method}_vs_{name}"] = result
            p_values.append(result["p_value"])

        if p_values:
            bonferroni = bonferroni_correction(p_values, alpha)
            fdr = fdr_correction(p_values, alpha)
            report["multiple_comparison"] = {
                "reference_method": reference_method,
                "pairwise_tests": pairwise,
                "bonferroni": bonferroni,
                "fdr_bh": fdr,
            }

    return report
