"""Statistical utilities used by the manuscript-facing experiments."""
import numpy as np
from typing import Dict
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
    if np.allclose(diff, 0.0, atol=1e-12):
        return {
            "test": "all_paired_differences_zero",
            "p_value": 1.0,
            "significant_at_005": False,
            "significant_at_001": False,
            "effect_size_cohens_d": 0.0,
            "effect_size_interpretation": "negligible",
            "mean_diff": float(diff_mean),
            "std_diff": float(diff_std),
            "n": n,
            "df": None,
        }

    if test_type == "auto":
        try:
            _, p_normality = stats.shapiro(diff)
            test_type = "paired_t" if p_normality > 0.05 else "wilcoxon"
        except Exception:
            test_type = "wilcoxon"

    if test_type == "paired_t":
        _, p_value = stats.ttest_rel(scores_a, scores_b)
        test_name = "paired_t_test"
        df = n - 1
    elif test_type == "wilcoxon":
        try:
            _, p_value = stats.wilcoxon(scores_a, scores_b)
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
        _, p_value = stats.ttest_rel(scores_a, scores_b)
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
