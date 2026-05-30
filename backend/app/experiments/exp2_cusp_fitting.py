import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from app.cuspnet.utils import compute_potential, find_fixed_points, classify_fixed_points
from app.data.loaders import KossakowskiLoader, StudentLifeLoader


def _sliding_window_params(
    series: np.ndarray,
    window_size: int = 7,
    pss10_range: float = 50.0,
    cdrisc_range: float = 40.0,
    mspss_range: float = 84.0,
) -> List[Dict]:
    n = len(series)
    params_list = []
    for start in range(0, n - window_size + 1):
        window = series[start : start + window_size]
        pss10 = np.mean(window[:, 0]) * pss10_range
        cdrisc = np.mean(window[:, 1]) * cdrisc_range
        mspss = np.mean(window[:, 2]) * mspss_range
        pss_norm = pss10 / pss10_range
        cdrisc_norm = cdrisc / cdrisc_range
        mspss_norm = mspss / mspss_range
        a = pss_norm - cdrisc_norm
        b = cdrisc_norm * mspss_norm - 0.5
        c = mspss_norm * 0.5
        params_list.append(
            {
                "window_start": start,
                "window_end": start + window_size,
                "a": a,
                "b": b,
                "c": c,
                "pss10_raw": pss10,
                "cdrisc_raw": cdrisc,
                "mspss_raw": mspss,
                "mean_x": np.mean(window[:, 3]) if window.shape[1] > 3 else 0.0,
            }
        )
    return params_list


def _fit_linear(x: np.ndarray, y: np.ndarray) -> Dict:
    n = len(x)
    if n < 3:
        return {"slope": 0.0, "intercept": np.mean(y) if n > 0 else 0.0, "rss": 0.0, "k": 2}
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx < 1e-10:
        return {"slope": 0.0, "intercept": y_mean, "rss": np.sum((y - y_mean) ** 2), "k": 2}
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept
    rss = np.sum((y - y_pred) ** 2)
    return {"slope": float(slope), "intercept": float(intercept), "rss": float(rss), "k": 2}


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> Dict:
    n = len(x)
    if n < 3:
        return {"L": 1.0, "k_param": 1.0, "x0": np.mean(x) if n > 0 else 0.0, "rss": 0.0, "k": 2}
    from scipy.optimize import minimize

    def logistic_model(params, x):
        L, k_param, x0 = params
        return L / (1.0 + np.exp(-k_param * (x - x0)))

    def objective(params):
        y_pred = logistic_model(params, x)
        return np.sum((y - y_pred) ** 2)

    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    init_params = [y_range, 1.0 / (x_range + 1e-10), np.mean(x)]
    try:
        result = minimize(
            objective, init_params, method="Nelder-Mead", options={"maxiter": 5000}
        )
        L, k_param, x0 = result.x
        y_pred = logistic_model(result.x, x)
        rss = np.sum((y - y_pred) ** 2)
    except Exception:
        L, k_param, x0 = init_params
        rss = objective(init_params)
    return {
        "L": float(L),
        "k_param": float(k_param),
        "x0": float(x0),
        "rss": float(rss),
        "k": 2,
    }


def _fit_cusp(x: np.ndarray, y: np.ndarray) -> Dict:
    n = len(x)
    if n < 4:
        return {"a": 0.0, "b": 0.0, "c": 1.0, "rss": 0.0, "k": 3}
    from scipy.optimize import minimize

    def cusp_model(params, x):
        a, b, c = params
        c_safe = max(abs(c), 0.01)
        roots = np.roots([c_safe, 0, -b, -a])
        real_roots = sorted([r.real for r in roots if abs(r.imag) < 1e-6])
        if len(real_roots) == 0:
            return np.full_like(x, 0.0)
        elif len(real_roots) == 1:
            return np.full_like(x, real_roots[0])
        else:
            y_pred = np.where(x < np.mean(real_roots), real_roots[0], real_roots[-1])
            return y_pred

    def objective(params):
        a, b, c = params
        c_safe = max(abs(c), 0.01)
        roots = np.roots([c_safe, 0, -b, -a])
        real_roots = sorted([r.real for r in roots if abs(r.imag) < 1e-6])
        if len(real_roots) == 0:
            return 1e10
        elif len(real_roots) == 1:
            y_pred = np.full(n, real_roots[0])
        else:
            mid = (real_roots[0] + real_roots[-1]) / 2
            y_pred = np.where(y < mid, real_roots[0], real_roots[-1])
        return np.sum((y - y_pred) ** 2)

    init_params = [0.0, 0.5, 1.0]
    try:
        result = minimize(
            objective, init_params, method="Nelder-Mead", options={"maxiter": 5000}
        )
        a, b, c = result.x
        rss = result.fun
    except Exception:
        a, b, c = init_params
        rss = objective(init_params)
    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "rss": float(rss),
        "k": 3,
    }


def _compute_aic_bic(rss: float, n: int, k: int) -> Dict:
    if n <= 0 or rss <= 0:
        return {"aic": float("inf"), "bic": float("inf")}
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return {"aic": float(aic), "bic": float(bic)}


def run_exp2(
    dataset: str = "kossakowski",
    window_size: int = 7,
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

    params_list = _sliding_window_params(combined, window_size=window_size)
    results["n_windows"] = len(params_list)
    results["window_size"] = window_size

    time_index = np.arange(len(params_list))
    a_series = np.array([p["a"] for p in params_list])
    b_series = np.array([p["b"] for p in params_list])
    c_series = np.array([p["c"] for p in params_list])
    x_series = np.array([p["mean_x"] for p in params_list])

    linear_fit = _fit_linear(time_index, x_series)
    logistic_fit = _fit_logistic(time_index, x_series)
    cusp_fit = _fit_cusp(time_index, x_series)

    n_obs = len(time_index)
    linear_ic = _compute_aic_bic(linear_fit["rss"], n_obs, linear_fit["k"])
    logistic_ic = _compute_aic_bic(logistic_fit["rss"], n_obs, logistic_fit["k"])
    cusp_ic = _compute_aic_bic(cusp_fit["rss"], n_obs, cusp_fit["k"])

    results["fits"] = {
        "linear": {
            "params": {k: v for k, v in linear_fit.items() if k != "k"},
            "k": linear_fit["k"],
            "aic": linear_ic["aic"],
            "bic": linear_ic["bic"],
        },
        "logistic": {
            "params": {k: v for k, v in logistic_fit.items() if k != "k"},
            "k": logistic_fit["k"],
            "aic": logistic_ic["aic"],
            "bic": logistic_ic["bic"],
        },
        "cusp": {
            "params": {k: v for k, v in cusp_fit.items() if k != "k"},
            "k": cusp_fit["k"],
            "aic": cusp_ic["aic"],
            "bic": cusp_ic["bic"],
        },
    }

    best_aic = min(linear_ic["aic"], logistic_ic["aic"], cusp_ic["aic"])
    best_bic = min(linear_ic["bic"], logistic_ic["bic"], cusp_ic["bic"])
    results["best_model"] = {
        "aic": "cusp" if cusp_ic["aic"] == best_aic else ("logistic" if logistic_ic["aic"] == best_aic else "linear"),
        "bic": "cusp" if cusp_ic["bic"] == best_bic else ("logistic" if logistic_ic["bic"] == best_bic else "linear"),
    }

    ss_total = np.sum((x_series - np.mean(x_series)) ** 2) if len(x_series) > 0 else 1.0
    linear_r2 = 1.0 - linear_fit["rss"] / ss_total if ss_total > 0 else 0.0
    logistic_r2 = 1.0 - logistic_fit["rss"] / ss_total if ss_total > 0 else 0.0
    cusp_r2 = 1.0 - cusp_fit["rss"] / ss_total if ss_total > 0 else 0.0
    results["pseudo_r2"] = {
        "linear": float(max(0.0, linear_r2)),
        "logistic": float(max(0.0, logistic_r2)),
        "cusp": float(max(0.0, cusp_r2)),
    }

    cusp_prediction_accuracy = 0.0
    if len(x_series) > 1:
        cusp_a, cusp_b, cusp_c = cusp_fit["a"], cusp_fit["b"], cusp_fit["c"]
        cusp_roots = find_fixed_points(cusp_a, cusp_b, cusp_c)
        cusp_classified = classify_fixed_points(cusp_roots, cusp_b, cusp_c)
        cusp_stable = [p for p in cusp_classified if p["stability"] == "stable"]
        if len(cusp_stable) >= 2:
            x_median = np.median(x_series)
            cusp_pred = np.where(x_series < x_median, cusp_stable[0]["value"], cusp_stable[-1]["value"])
        elif len(cusp_stable) == 1:
            cusp_pred = np.full_like(x_series, cusp_stable[0]["value"])
        else:
            cusp_pred = np.full_like(x_series, np.mean(x_series))
        cusp_prediction_accuracy = float(np.mean(np.abs(x_series - cusp_pred) < 0.5))
    results["prediction_accuracy"] = {
        "cusp": cusp_prediction_accuracy,
    }

    results["parameter_series"] = {
        "time_index": time_index.tolist(),
        "a": a_series.tolist(),
        "b": b_series.tolist(),
        "c": c_series.tolist(),
        "x": x_series.tolist(),
    }

    results["dataset_info"] = {
        "name": dataset,
        "n_observations": combined.shape[0],
        "n_variables": combined.shape[1],
    }

    return results
