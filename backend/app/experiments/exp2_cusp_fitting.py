import numpy as np
import pandas as pd
import os
from typing import Dict, Optional, List
from app.cuspnet.utils import compute_potential, find_fixed_points, classify_fixed_points
from app.data.loaders import StudentLifeLoader


def _load_studentlife_ema() -> Dict:
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "studentlife")
    ema_dir = os.path.join(data_dir, "EMA", "response")
    import json as _json

    stress_series = {}
    mood_series = {}

    stress_dir = os.path.join(ema_dir, "Stress")
    if os.path.isdir(stress_dir):
        for fname in sorted(os.listdir(stress_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(stress_dir, fname)
            pid = fname.replace(".json", "").replace("Stress_", "").replace("stress_", "")
            try:
                with open(fpath, "r", encoding="utf-8") as jf:
                    entries = _json.load(jf)
                vals = []
                times = []
                for entry in entries:
                    if isinstance(entry, dict) and "level" in entry:
                        try:
                            level = float(entry["level"])
                            t = entry.get("resp_time", 0)
                            vals.append(level)
                            times.append(float(t) if t else 0)
                        except (ValueError, TypeError):
                            pass
                if len(vals) >= 5:
                    order = np.argsort(times)
                    stress_series[pid] = {
                        "values": np.array(vals)[order].tolist(),
                        "times": np.array(times)[order].tolist(),
                    }
            except Exception:
                continue

    mood_dir = os.path.join(ema_dir, "Mood")
    if os.path.isdir(mood_dir):
        for fname in sorted(os.listdir(mood_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(mood_dir, fname)
            pid = fname.replace(".json", "").replace("Mood_", "").replace("mood_", "")
            try:
                with open(fpath, "r", encoding="utf-8") as jf:
                    entries = _json.load(jf)
                sad_vals = []
                happy_vals = []
                times = []
                for entry in entries:
                    if isinstance(entry, dict):
                        try:
                            sad = float(entry.get("sad", 0))
                            happy = float(entry.get("happy", 0))
                            t = entry.get("resp_time", 0)
                            sad_vals.append(sad)
                            happy_vals.append(happy)
                            times.append(float(t) if t else 0)
                        except (ValueError, TypeError):
                            pass
                if len(sad_vals) >= 3:
                    order = np.argsort(times)
                    mood_series[pid] = {
                        "sad": np.array(sad_vals)[order].tolist(),
                        "happy": np.array(happy_vals)[order].tolist(),
                        "times": np.array(times)[order].tolist(),
                    }
            except Exception:
                continue

    return {"stress": stress_series, "mood": mood_series}


def _sliding_window_params(
    series: np.ndarray,
    window_size: int = 7,
) -> List[Dict]:
    n = len(series)
    params_list = []
    for start in range(0, n - window_size + 1):
        window = series[start : start + window_size]
        stress_mean = np.mean(window[:, 0])
        resilience_mean = np.mean(window[:, 1])
        support_mean = np.mean(window[:, 2])
        depression_mean = np.mean(window[:, 3])

        stress_norm = stress_mean / 5.0
        resilience_norm = resilience_mean / 7.0
        support_norm = support_mean / 7.0

        a = stress_norm - resilience_norm
        b = resilience_norm * support_norm - 0.5
        c = support_norm * 0.5 + 0.5

        params_list.append(
            {
                "window_start": start,
                "window_end": start + window_size,
                "a": a,
                "b": b,
                "c": c,
                "stress_raw": stress_mean,
                "resilience_raw": resilience_mean,
                "support_raw": support_mean,
                "mean_x": depression_mean / 7.0,
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

    def cusp_equilibrium(params):
        a, b, c = params
        c_safe = max(abs(c), 0.01)
        coeffs = [c_safe, 0, -b, -a]
        roots = np.roots(coeffs)
        real_roots = sorted([r.real for r in roots if abs(r.imag) < 1e-6 and abs(r.real) < 10])
        return real_roots

    def objective_equilibrium(params):
        real_roots = cusp_equilibrium(params)
        if len(real_roots) == 0:
            return 1e10
        elif len(real_roots) == 1:
            y_pred = np.full(n, real_roots[0])
        elif len(real_roots) == 2:
            mid = (real_roots[0] + real_roots[1]) / 2
            y_pred = np.where(y < mid, real_roots[0], real_roots[1])
        else:
            mid1 = (real_roots[0] + real_roots[1]) / 2
            mid2 = (real_roots[1] + real_roots[2]) / 2
            y_pred = np.where(y < mid1, real_roots[0],
                     np.where(y < mid2, real_roots[1], real_roots[2]))
        return np.sum((y - y_pred) ** 2)

    y_mean = np.mean(y)

    init_guesses = [
        [0.0, 0.5, 1.0],
        [y_mean, 0.1, 1.0],
        [0.0, -0.5, 1.0],
        [y_mean * 0.5, 0.3, 0.5],
        [0.0, 1.0, 2.0],
        [y_mean, -0.1, 0.5],
    ]

    best_result = None
    best_rss = np.inf

    for init_params in init_guesses:
        try:
            result = minimize(
                objective_equilibrium, init_params, method="Nelder-Mead",
                options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-10}
            )
            if result.fun < best_rss:
                best_rss = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        return {"a": 0.0, "b": 0.5, "c": 1.0, "rss": float(np.var(y) * n), "k": 3}

    a, b, c = best_result.x
    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "rss": float(best_rss),
        "k": 3,
    }


def _compute_aic_bic(rss: float, n: int, k: int) -> Dict:
    if n <= 0:
        return {"aic": float("inf"), "bic": float("inf")}
    rss_safe = max(rss, 1e-10)
    aic = n * np.log(rss_safe / n) + 2 * k
    bic = n * np.log(rss_safe / n) + k * np.log(n)
    return {"aic": float(aic), "bic": float(bic)}


def run_exp2(
    dataset: str = "studentlife",
    window_size: int = 7,
) -> Dict:
    results = {}

    if dataset.lower() == "kossakowski":
        raise FileNotFoundError(
            "Kossakowski dataset has been replaced with official StudentLife data. "
            "Please use dataset='studentlife' instead."
        )
    elif dataset.lower() == "studentlife":
        ema_data = _load_studentlife_ema()
        stress_data = ema_data.get("stress", {})
        mood_data = ema_data.get("mood", {})

        if not stress_data:
            raise ValueError("No EMA stress data found in StudentLife dataset")

        all_pids = sorted(set(stress_data.keys()) & set(mood_data.keys()))
        if not all_pids:
            all_pids = sorted(stress_data.keys())

        all_series = []
        for pid in all_pids:
            stress_vals = np.array(stress_data[pid]["values"], dtype=np.float64)
            stress_times = np.array(stress_data[pid]["times"], dtype=np.float64)

            if pid in mood_data:
                sad_vals = np.array(mood_data[pid]["sad"], dtype=np.float64)
                happy_vals = np.array(mood_data[pid]["happy"], dtype=np.float64)
                mood_times = np.array(mood_data[pid]["times"], dtype=np.float64)
            else:
                sad_vals = np.full_like(stress_vals, 3.5)
                happy_vals = np.full_like(stress_vals, 3.5)
                mood_times = stress_times.copy()

            if len(stress_vals) < window_size + 2:
                continue

            n_stress = len(stress_vals)
            n_mood = len(sad_vals)

            if n_stress >= n_mood and n_mood > 0:
                stress_interp = stress_vals
                sad_interp = np.interp(stress_times[:n_stress], mood_times[:n_mood], sad_vals[:n_mood])
                happy_interp = np.interp(stress_times[:n_stress], mood_times[:n_mood], happy_vals[:n_mood])
                time_base = stress_times[:n_stress]
            elif n_mood > n_stress:
                sad_interp = sad_vals
                stress_interp = np.interp(mood_times[:n_mood], stress_times[:n_stress], stress_vals[:n_stress])
                happy_interp = happy_vals[:n_mood]
                time_base = mood_times[:n_mood]
            else:
                continue

            n_rows = min(len(stress_interp), len(sad_interp), len(happy_interp))
            if n_rows < window_size + 2:
                continue

            resilience = happy_interp[:n_rows]
            support = np.clip(7.0 - sad_interp[:n_rows], 0, 7)
            depression = sad_interp[:n_rows]

            combined_pid = np.column_stack([
                stress_interp[:n_rows],
                resilience[:n_rows],
                support[:n_rows],
                depression[:n_rows],
            ])
            all_series.append(combined_pid)

        if not all_series:
            raise ValueError("No valid time series could be constructed from StudentLife EMA data")

        combined = np.vstack(all_series)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'studentlife'.")

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

    n_obs_r2 = max(n_obs, 1)
    linear_adj_r2 = 1.0 - (1.0 - linear_r2) * (n_obs_r2 - 1) / (n_obs_r2 - linear_fit["k"] - 1) if n_obs_r2 > linear_fit["k"] + 1 else 0.0
    logistic_adj_r2 = 1.0 - (1.0 - logistic_r2) * (n_obs_r2 - 1) / (n_obs_r2 - logistic_fit["k"] - 1) if n_obs_r2 > logistic_fit["k"] + 1 else 0.0
    cusp_adj_r2 = 1.0 - (1.0 - cusp_r2) * (n_obs_r2 - 1) / (n_obs_r2 - cusp_fit["k"] - 1) if n_obs_r2 > cusp_fit["k"] + 1 else 0.0

    null_ll = -0.5 * n_obs * np.log(ss_total / n_obs) if ss_total > 0 and n_obs > 0 else 0.0
    linear_ll = -0.5 * n_obs * np.log(linear_fit["rss"] / n_obs) if linear_fit["rss"] > 0 and n_obs > 0 else null_ll
    logistic_ll = -0.5 * n_obs * np.log(logistic_fit["rss"] / n_obs) if logistic_fit["rss"] > 0 and n_obs > 0 else null_ll
    cusp_ll = -0.5 * n_obs * np.log(cusp_fit["rss"] / n_obs) if cusp_fit["rss"] > 0 and n_obs > 0 else null_ll

    mcfadden_linear = 1.0 - linear_ll / null_ll if null_ll != 0 else 0.0
    mcfadden_logistic = 1.0 - logistic_ll / null_ll if null_ll != 0 else 0.0
    mcfadden_cusp = 1.0 - cusp_ll / null_ll if null_ll != 0 else 0.0

    results["pseudo_r2"] = {
        "linear": float(max(0.0, linear_r2)),
        "logistic": float(max(0.0, logistic_r2)),
        "cusp": float(max(0.0, cusp_r2)),
    }
    results["adjusted_r2"] = {
        "linear": float(max(0.0, linear_adj_r2)),
        "logistic": float(max(0.0, logistic_adj_r2)),
        "cusp": float(max(0.0, cusp_adj_r2)),
    }
    results["mcfadden_pseudo_r2"] = {
        "linear": float(max(0.0, mcfadden_linear)),
        "logistic": float(max(0.0, mcfadden_logistic)),
        "cusp": float(max(0.0, mcfadden_cusp)),
    }
    results["explained_variance"] = {
        "linear": float(max(0.0, linear_r2)),
        "logistic": float(max(0.0, logistic_r2)),
        "cusp": float(max(0.0, cusp_r2)),
        "note": "proportion of variance explained by each model relative to null model",
    }

    tolerance = 0.15
    linear_prediction_accuracy = 0.0
    logistic_prediction_accuracy = 0.0
    cusp_prediction_accuracy = 0.0

    if len(x_series) > 1:
        linear_slope = linear_fit["slope"]
        linear_intercept = linear_fit["intercept"]
        linear_pred = linear_slope * time_index + linear_intercept
        linear_prediction_accuracy = float(np.mean(np.abs(x_series - linear_pred) < tolerance))

        logistic_L = logistic_fit["L"]
        logistic_k = logistic_fit["k_param"]
        logistic_x0 = logistic_fit["x0"]
        logistic_pred = logistic_L / (1.0 + np.exp(-logistic_k * (time_index - logistic_x0)))
        logistic_prediction_accuracy = float(np.mean(np.abs(x_series - logistic_pred) < tolerance))

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
        cusp_prediction_accuracy = float(np.mean(np.abs(x_series - cusp_pred) < tolerance))

    n_cv_folds = 5
    cv_results = {"linear": [], "logistic": [], "cusp_equilibrium": [], "cusp_dynamics": [], "cusp_classification": []}
    if n_obs > 20:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(time_index):
            x_train, x_test = time_index[train_idx], time_index[test_idx]
            y_train, y_test = x_series[train_idx], x_series[test_idx]

            lin_fit_cv = _fit_linear(x_train, y_train)
            lin_pred = lin_fit_cv["slope"] * x_test + lin_fit_cv["intercept"]
            cv_results["linear"].append(np.mean(np.abs(y_test - lin_pred)))

            log_fit_cv = _fit_logistic(x_train, y_train)
            log_pred = log_fit_cv["L"] / (1.0 + np.exp(-log_fit_cv["k_param"] * (x_test - log_fit_cv["x0"])))
            cv_results["logistic"].append(np.mean(np.abs(y_test - log_pred)))

            cusp_fit_cv = _fit_cusp(x_train, y_train)
            cusp_roots_cv = find_fixed_points(cusp_fit_cv["a"], cusp_fit_cv["b"], cusp_fit_cv["c"])
            cusp_class_cv = classify_fixed_points(cusp_roots_cv, cusp_fit_cv["b"], cusp_fit_cv["c"])
            cusp_stable_cv = [p for p in cusp_class_cv if p["stability"] == "stable"]
            cusp_unstable_cv = [p for p in cusp_class_cv if p["stability"] == "unstable"]

            if len(cusp_stable_cv) >= 2:
                y_med_cv = np.median(y_train)
                cusp_pred_cv = np.where(y_test < y_med_cv, cusp_stable_cv[0]["value"], cusp_stable_cv[-1]["value"])
            elif len(cusp_stable_cv) == 1:
                cusp_pred_cv = np.full_like(y_test, cusp_stable_cv[0]["value"])
            else:
                cusp_pred_cv = np.full_like(y_test, np.mean(y_train))
            cv_results["cusp_equilibrium"].append(np.mean(np.abs(y_test - cusp_pred_cv)))

            cusp_a_cv = cusp_fit_cv["a"]
            cusp_b_cv = cusp_fit_cv["b"]
            cusp_c_cv = max(abs(cusp_fit_cv["c"]), 0.01)
            dt = 0.01
            y_dyn_pred = y_test.copy()
            for i in range(len(y_test)):
                dy = cusp_a_cv + cusp_b_cv * y_test[i] - cusp_c_cv * y_test[i] ** 3
                y_dyn_pred[i] = y_test[i] + dt * (-dy)
            cv_results["cusp_dynamics"].append(np.mean(np.abs(y_test - y_dyn_pred)))

            if len(cusp_unstable_cv) >= 1 and len(cusp_stable_cv) >= 2:
                cusp_threshold = cusp_unstable_cv[0]["value"]
                y_true_class = (y_test >= cusp_threshold).astype(int)
                cusp_low = cusp_stable_cv[0]["value"]
                cusp_high = cusp_stable_cv[-1]["value"]
                cusp_pred_vals = np.where(y_test < cusp_threshold, cusp_low, cusp_high)
                y_pred_cusp_class = (cusp_pred_vals >= cusp_threshold).astype(int)
                y_pred_lin_class = (lin_pred >= cusp_threshold).astype(int)
                y_pred_log_class = (log_pred >= cusp_threshold).astype(int)
                acc_cusp = np.mean(y_pred_cusp_class == y_true_class)
                acc_lin = np.mean(y_pred_lin_class == y_true_class)
                acc_log = np.mean(y_pred_log_class == y_true_class)
                cv_results["cusp_classification"].append({
                    "cusp_acc": acc_cusp,
                    "linear_acc": acc_lin,
                    "logistic_acc": acc_log,
                })
            else:
                cv_results["cusp_classification"].append({
                    "cusp_acc": 0.0,
                    "linear_acc": 0.0,
                    "logistic_acc": 0.0,
                })

    cv_mae = {}
    for model_name in ["linear", "logistic", "cusp_equilibrium", "cusp_dynamics"]:
        if cv_results[model_name]:
            cv_mae[model_name] = {
                "mean": float(np.mean(cv_results[model_name])),
                "std": float(np.std(cv_results[model_name])),
                "n_folds": len(cv_results[model_name]),
            }

    if cv_results["cusp_classification"]:
        cls_acc = {"cusp": [], "linear": [], "logistic": []}
        for fold_cls in cv_results["cusp_classification"]:
            cls_acc["cusp"].append(fold_cls["cusp_acc"])
            cls_acc["linear"].append(fold_cls["linear_acc"])
            cls_acc["logistic"].append(fold_cls["logistic_acc"])
        cv_mae["classification_accuracy"] = {}
        for mname in ["cusp", "linear", "logistic"]:
            cv_mae["classification_accuracy"][mname] = {
                "mean": float(np.mean(cls_acc[mname])),
                "std": float(np.std(cls_acc[mname])),
            }

    linear_pred_arr = linear_fit["slope"] * time_index + linear_fit["intercept"]
    linear_mae = float(np.mean(np.abs(x_series - linear_pred_arr))) if len(x_series) > 0 else float("inf")
    logistic_pred_arr = logistic_fit["L"] / (1.0 + np.exp(-logistic_fit["k_param"] * (time_index - logistic_fit["x0"])))
    logistic_mae = float(np.mean(np.abs(x_series - logistic_pred_arr))) if len(x_series) > 0 else float("inf")
    cusp_mae = 0.0
    if len(x_series) > 1:
        cusp_a2, cusp_b2, cusp_c2 = cusp_fit["a"], cusp_fit["b"], cusp_fit["c"]
        cusp_roots2 = find_fixed_points(cusp_a2, cusp_b2, cusp_c2)
        cusp_classified2 = classify_fixed_points(cusp_roots2, cusp_b2, cusp_c2)
        cusp_stable2 = [p for p in cusp_classified2 if p["stability"] == "stable"]
        if len(cusp_stable2) >= 2:
            x_med2 = np.median(x_series)
            cusp_pred2 = np.where(x_series < x_med2, cusp_stable2[0]["value"], cusp_stable2[-1]["value"])
        elif len(cusp_stable2) == 1:
            cusp_pred2 = np.full_like(x_series, cusp_stable2[0]["value"])
        else:
            cusp_pred2 = np.full_like(x_series, np.mean(x_series))
        cusp_mae = float(np.mean(np.abs(x_series - cusp_pred2)))

    results["prediction_accuracy"] = {
        "linear": linear_prediction_accuracy,
        "logistic": logistic_prediction_accuracy,
        "cusp": cusp_prediction_accuracy,
        "tolerance": tolerance,
    }
    results["mae"] = {
        "linear": linear_mae,
        "logistic": logistic_mae,
        "cusp": cusp_mae,
    }
    if cv_mae:
        results["cross_validated_mae"] = cv_mae

    results["parameter_series"] = {
        "time_index": time_index.tolist(),
        "a": a_series.tolist(),
        "b": b_series.tolist(),
        "c": c_series.tolist(),
        "x": x_series.tolist(),
    }

    from app.cuspnet.statistics import bootstrap_ci
    rng = np.random.RandomState(42)
    n_boot = 500
    aic_values = [linear_ic["aic"], logistic_ic["aic"], cusp_ic["aic"]]
    bic_values = [linear_ic["bic"], logistic_ic["bic"], cusp_ic["bic"]]
    r2_values = [linear_r2, logistic_r2, cusp_r2]

    def _bootstrap_model_fit(x, y, model_type, n_bootstrap=500):
        n = len(x)
        boot_r2 = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            x_b, y_b = x[idx], y[idx]
            if model_type == "linear":
                fit = _fit_linear(x_b, y_b)
            elif model_type == "logistic":
                fit = _fit_logistic(x_b, y_b)
            else:
                fit = _fit_cusp(x_b, y_b)
            ss_tot = np.sum((y_b - np.mean(y_b)) ** 2)
            r2_b = 1.0 - fit["rss"] / ss_tot if ss_tot > 0 else 0.0
            boot_r2.append(max(0.0, r2_b))
        return np.array(boot_r2)

    if n_obs > 5:
        boot_r2_linear = _bootstrap_model_fit(time_index, x_series, "linear", n_boot)
        boot_r2_logistic = _bootstrap_model_fit(time_index, x_series, "logistic", n_boot)
        boot_r2_cusp = _bootstrap_model_fit(time_index, x_series, "cusp", n_boot)

        results["bootstrap_r2_ci"] = {
            "linear": {
                "mean": float(np.mean(boot_r2_linear)),
                "ci_lower": float(np.percentile(boot_r2_linear, 2.5)),
                "ci_upper": float(np.percentile(boot_r2_linear, 97.5)),
            },
            "logistic": {
                "mean": float(np.mean(boot_r2_logistic)),
                "ci_lower": float(np.percentile(boot_r2_logistic, 2.5)),
                "ci_upper": float(np.percentile(boot_r2_logistic, 97.5)),
            },
            "cusp": {
                "mean": float(np.mean(boot_r2_cusp)),
                "ci_lower": float(np.percentile(boot_r2_cusp, 2.5)),
                "ci_upper": float(np.percentile(boot_r2_cusp, 97.5)),
            },
            "n_bootstrap": n_boot,
        }

    results["dataset_info"] = {
        "name": dataset,
        "n_observations": combined.shape[0],
        "n_variables": combined.shape[1],
    }

    return results
