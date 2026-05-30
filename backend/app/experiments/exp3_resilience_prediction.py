import numpy as np
from typing import Dict, List, Optional
from app.cuspnet.utils import (
    compute_resilience_reserve,
    compute_potential,
    find_fixed_points,
    classify_fixed_points,
)
from app.data.loaders import KossakowskiLoader, StudentLifeLoader


def _sliding_window_delta_v(
    series: np.ndarray,
    window_size: int = 7,
    theta_bifurcation: float = 0.5,
) -> List[Dict]:
    n = len(series)
    delta_v_list = []
    for start in range(0, n - window_size + 1):
        window = series[start : start + window_size]
        pss_norm = np.mean(window[:, 0])
        cdrisc_norm = np.mean(window[:, 1])
        mspss_norm = np.mean(window[:, 2])
        a = pss_norm - cdrisc_norm
        b = cdrisc_norm * mspss_norm - theta_bifurcation
        c = mspss_norm * 0.5
        delta_v = compute_resilience_reserve(a, b, c)
        delta_v_list.append(
            {
                "window_start": start,
                "window_end": start + window_size,
                "a": a,
                "b": b,
                "c": c,
                "delta_v": delta_v,
                "mean_x": np.mean(window[:, 3]) if window.shape[1] > 3 else 0.0,
            }
        )
    return delta_v_list


def _compute_rolling_variance(series: np.ndarray, window_size: int = 7) -> np.ndarray:
    n = len(series)
    rolling_var = np.full(n, np.nan)
    for i in range(window_size - 1, n):
        window = series[i - window_size + 1 : i + 1]
        rolling_var[i] = np.var(window)
    return rolling_var


def _compute_rolling_autocorr(series: np.ndarray, window_size: int = 7, lag: int = 1) -> np.ndarray:
    n = len(series)
    rolling_ac = np.full(n, np.nan)
    for i in range(window_size - 1, n):
        window = series[i - window_size + 1 : i + 1]
        if np.var(window) < 1e-10:
            rolling_ac[i] = 0.0
            continue
        if lag >= len(window):
            rolling_ac[i] = 0.0
            continue
        mean_w = np.mean(window)
        centered = window - mean_w
        denom = np.sum(centered ** 2)
        if denom < 1e-10:
            rolling_ac[i] = 0.0
            continue
        numer = np.sum(centered[: len(centered) - lag] * centered[lag:])
        rolling_ac[i] = numer / denom
    return rolling_ac


def _detect_tipping_points(
    delta_v_series: np.ndarray,
    threshold_sigma: float = 2.0,
    min_interval: int = 5,
) -> List[int]:
    valid_mask = np.isfinite(delta_v_series)
    if np.sum(valid_mask) < 3:
        return []
    valid_vals = delta_v_series[valid_mask]
    mean_dv = np.mean(valid_vals)
    std_dv = np.std(valid_vals)
    if std_dv < 1e-10:
        return []
    z_scores = (delta_v_series - mean_dv) / std_dv
    tipping_points = []
    last_tip = -min_interval
    for i in range(len(z_scores)):
        if np.isnan(z_scores[i]):
            continue
        if z_scores[i] < -threshold_sigma and (i - last_tip) >= min_interval:
            tipping_points.append(i)
            last_tip = i
    return tipping_points


def _compute_early_warning_signals(
    x_series: np.ndarray,
    window_size: int = 7,
) -> Dict:
    rolling_var = _compute_rolling_variance(x_series, window_size)
    rolling_ac = _compute_rolling_autocorr(x_series, window_size, lag=1)
    valid_var = rolling_var[~np.isnan(rolling_var)]
    valid_ac = rolling_ac[~np.isnan(rolling_ac)]
    var_trend = 0.0
    ac_trend = 0.0
    if len(valid_var) > 2:
        t = np.arange(len(valid_var))
        var_trend = np.polyfit(t, valid_var, 1)[0]
    if len(valid_ac) > 2:
        t = np.arange(len(valid_ac))
        ac_trend = np.polyfit(t, valid_ac, 1)[0]
    return {
        "rolling_variance": rolling_var.tolist(),
        "rolling_autocorrelation": rolling_ac.tolist(),
        "variance_trend": float(var_trend),
        "autocorrelation_trend": float(ac_trend),
        "variance_increasing": var_trend > 0,
        "autocorrelation_increasing": ac_trend > 0,
    }


def run_exp3(
    dataset: str = "kossakowski",
    window_size: int = 7,
    tipping_sigma: float = 2.0,
    theta_bifurcation: float = 0.5,
) -> Dict:
    results = {}

    if dataset.lower() == "kossakowski":
        loader = KossakowskiLoader()
        df = loader.load()
        if df.empty:
            raise ValueError("Kossakowski dataset is empty")
        series = df[["stress", "resilience", "social_support", "mood"]].values
        stress_norm = series[:, 0] / (series[:, 0].max() + 1e-10)
        resilience_norm = series[:, 1] / (series[:, 1].max() + 1e-10)
        social_norm = series[:, 2] / (series[:, 2].max() + 1e-10)
        combined = np.column_stack([stress_norm, resilience_norm, social_norm, series[:, 3]])
    elif dataset.lower() == "studentlife":
        loader = StudentLifeLoader()
        data = loader.load()
        phq9 = data.get("phq9_series", {})
        stress = data.get("stress_series", {})
        if not phq9 and not stress:
            raise ValueError("StudentLife dataset is empty")
        all_series = []
        for pid in phq9:
            phq_arr = np.array(phq9[pid])
            stress_arr = np.array(stress.get(pid, [[0.5]] * len(phq_arr)))
            if phq_arr.ndim == 1:
                phq_arr = phq_arr.reshape(-1, 1)
            if stress_arr.ndim == 1:
                stress_arr = stress_arr.reshape(-1, 1)
            phq_mean = np.mean(phq_arr, axis=1)
            stress_mean = np.mean(stress_arr, axis=1)
            n_rows = min(len(phq_mean), len(stress_mean))
            combined_pid = np.column_stack([
                stress_mean[:n_rows],
                1.0 - phq_mean[:n_rows] / (phq_mean[:n_rows].max() + 1e-10),
                np.ones(n_rows) * 0.5,
                phq_mean[:n_rows],
            ])
            all_series.append(combined_pid)
        combined = np.vstack(all_series)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'kossakowski' or 'studentlife'.")

    delta_v_list = _sliding_window_delta_v(
        combined, window_size=window_size, theta_bifurcation=theta_bifurcation
    )
    delta_v_series = np.array([d["delta_v"] for d in delta_v_list])
    x_series = np.array([d["mean_x"] for d in delta_v_list])

    finite_dv = delta_v_series[np.isfinite(delta_v_series)]
    results["delta_v_stats"] = {
        "mean": float(np.mean(finite_dv)) if len(finite_dv) > 0 else float("inf"),
        "std": float(np.std(finite_dv)) if len(finite_dv) > 0 else 0.0,
        "min": float(np.min(finite_dv)) if len(finite_dv) > 0 else float("inf"),
        "max": float(np.max(finite_dv)) if len(finite_dv) > 0 else float("inf"),
        "n_low_resilience": int(np.sum(finite_dv < 0.5)) if len(finite_dv) > 0 else 0,
        "n_critical": int(np.sum(finite_dv < 0.1)) if len(finite_dv) > 0 else 0,
    }

    tipping_points = _detect_tipping_points(
        delta_v_series, threshold_sigma=tipping_sigma, min_interval=window_size
    )
    results["tipping_points"] = {
        "indices": tipping_points,
        "n_detected": len(tipping_points),
        "threshold_sigma": tipping_sigma,
    }

    ews = _compute_early_warning_signals(x_series, window_size=window_size)
    results["early_warning_signals"] = {
        "variance_trend": ews["variance_trend"],
        "autocorrelation_trend": ews["autocorrelation_trend"],
        "variance_increasing": ews["variance_increasing"],
        "autocorrelation_increasing": ews["autocorrelation_increasing"],
    }

    results["delta_v_series"] = delta_v_series.tolist()
    results["x_series"] = x_series.tolist()

    if len(tipping_points) > 0:
        pre_tip_dv = []
        for tip_idx in tipping_points:
            start = max(0, tip_idx - window_size)
            pre_tip_dv.extend(delta_v_series[start:tip_idx].tolist())
        results["pre_tipping_delta_v"] = {
            "mean": float(np.nanmean(pre_tip_dv)) if pre_tip_dv else float("inf"),
            "std": float(np.nanstd(pre_tip_dv)) if pre_tip_dv else 0.0,
        }
    else:
        results["pre_tipping_delta_v"] = {"mean": None, "std": None}

    finite_mask = np.isfinite(delta_v_series)
    if np.sum(finite_mask) > 10 and len(tipping_points) > 0:
        tipping_binary = np.zeros(len(delta_v_series))
        for tp_idx in tipping_points:
            if tp_idx < len(tipping_binary):
                tipping_binary[tp_idx] = 1.0
        dv_finite = delta_v_series[finite_mask]
        tip_finite = tipping_binary[finite_mask]
        n_finite = len(dv_finite)
        if np.sum(tip_finite) > 0 and np.sum(tip_finite) < n_finite:
            try:
                from sklearn.metrics import roc_auc_score
                dv_neg = -dv_finite
                auc = roc_auc_score(tip_finite, dv_neg)
                results["roc_analysis"] = {
                    "auc_roc": float(auc),
                    "predictor": "negative_delta_v",
                    "n_positive": int(np.sum(tip_finite)),
                    "n_negative": int(n_finite - np.sum(tip_finite)),
                }
            except Exception as e:
                results["roc_analysis"] = {"error": str(e)}

            try:
                from lifelines import CoxPHFitter
                import pandas as pd
                time_to_event = np.ones(n_finite)
                for i in range(n_finite):
                    future_tips = [tp for tp in tipping_points if tp >= i]
                    if future_tips:
                        time_to_event[i] = min(future_tips) - i + 1
                    else:
                        time_to_event[i] = n_finite - i
                cox_df = pd.DataFrame({
                    "T": time_to_event,
                    "E": tip_finite,
                    "delta_v": dv_finite,
                })
                cox_df = cox_df[cox_df["T"] > 0]
                if len(cox_df) > 5 and cox_df["E"].sum() > 0:
                    cph = CoxPHFitter()
                    cph.fit(cox_df, duration_col="T", event_col="E")
                    results["cox_regression"] = {
                        "hazard_ratio": float(np.exp(cph.params_["delta_v"])),
                        "p_value": float(cph.summary.loc["delta_v", "p"]),
                        "concordance_index": float(cph.concordance_index_),
                        "coefficient": float(cph.params_["delta_v"]),
                    }
            except ImportError:
                try:
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.metrics import roc_auc_score
                    dv_2d = dv_finite.reshape(-1, 1)
                    lr = LogisticRegression(max_iter=1000, random_state=42)
                    lr.fit(dv_2d, tip_finite)
                    pred_prob = lr.predict_proba(dv_2d)[:, 1]
                    auc_lr = roc_auc_score(tip_finite, pred_prob)
                    coef = float(lr.coef_[0, 0])
                    results["cox_regression"] = {
                        "note": "lifelines unavailable, logistic regression used as proxy",
                        "hazard_ratio_proxy": float(np.exp(coef)),
                        "coefficient": coef,
                        "auc_roc": float(auc_lr),
                    }
                except Exception as e:
                    results["cox_regression"] = {"error": str(e)}
            except Exception as e:
                results["cox_regression"] = {"error": str(e)}
    else:
        results["roc_analysis"] = {"note": "insufficient data for ROC analysis"}
        results["cox_regression"] = {"note": "insufficient data for Cox regression"}

    results["dataset_info"] = {
        "name": dataset,
        "n_observations": combined.shape[0],
        "window_size": window_size,
    }

    return results
