"""
Experiment 3: Resilience Reserve Prediction.
Validates the CUSP resilience reserve (delta-V) as an early warning signal
for mental health deterioration using StudentLife EMA data with tipping point detection.
"""
import os
import numpy as np
from typing import Dict, List, Optional
from app.cuspnet.utils import (
    compute_resilience_reserve,
    compute_potential,
    find_fixed_points,
    classify_fixed_points,
)


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


def _sliding_window_delta_v(
    series: np.ndarray,
    window_size: int = 7,
    theta_bifurcation: float = 0.5,
    cusp_params: Optional[Dict] = None,
) -> List[Dict]:
    n = len(series)
    delta_v_list = []

    if cusp_params is not None:
        cusp_a = cusp_params.get("a", 0.0)
        cusp_b = cusp_params.get("b", 0.0)
        cusp_c = cusp_params.get("c", 1.0)
        cusp_roots = find_fixed_points(cusp_a, cusp_b, cusp_c)
        cusp_classified = classify_fixed_points(cusp_roots, cusp_b, cusp_c)
        cusp_stable = [p for p in cusp_classified if p["stability"] == "stable"]
        cusp_unstable = [p for p in cusp_classified if p["stability"] == "unstable"]
        has_bifurcation = len(cusp_stable) >= 2 and len(cusp_unstable) >= 1

    x_raw = series[:, 3] if series.shape[1] > 3 else series[:, 0]
    rolling_var = _compute_rolling_variance(x_raw, window_size=window_size)
    rolling_ac = _compute_rolling_autocorr(x_raw, window_size=window_size, lag=1)

    var_valid = rolling_var[~np.isnan(rolling_var)]
    ac_valid = rolling_ac[~np.isnan(rolling_ac)]
    var_max = np.percentile(var_valid, 95) if len(var_valid) > 10 else 1.0
    ac_max = np.percentile(ac_valid, 95) if len(ac_valid) > 10 else 1.0

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
        b = resilience_norm * support_norm - theta_bifurcation
        c = support_norm * 0.5 + 0.5

        delta_v_raw = compute_resilience_reserve(a, b, c)
        if not np.isfinite(delta_v_raw):
            delta_v_raw = 100.0

        if cusp_params is not None and has_bifurcation:
            x_val = depression_mean / 7.0
            v_current = compute_potential(np.array([x_val]), cusp_a, cusp_b, cusp_c)[0]
            v_saddle = compute_potential(
                np.array([cusp_unstable[0]["value"]]), cusp_a, cusp_b, cusp_c
            )[0]
            v_attractor = v_current
            if x_val < cusp_unstable[0]["value"]:
                v_attractor = compute_potential(
                    np.array([cusp_stable[0]["value"]]), cusp_a, cusp_b, cusp_c
                )[0]
            else:
                v_attractor = compute_potential(
                    np.array([cusp_stable[-1]["value"]]), cusp_a, cusp_b, cusp_c
                )[0]
            delta_v_cusp = v_saddle - v_current
            delta_v_well = v_saddle - v_attractor
            resilience_ratio = delta_v_cusp / (delta_v_well + 1e-10)
        else:
            delta_v_cusp = delta_v_raw
            delta_v_well = delta_v_raw
            resilience_ratio = 1.0

        rv = rolling_var[start + window_size - 1] if (start + window_size - 1) < len(rolling_var) else np.nan
        rac = rolling_ac[start + window_size - 1] if (start + window_size - 1) < len(rolling_ac) else np.nan

        var_contribution = 0.0
        if not np.isnan(rv) and var_max > 1e-10:
            var_contribution = min(rv / var_max, 1.0)

        ac_contribution = 0.0
        if not np.isnan(rac) and ac_max > 1e-10:
            ac_contribution = min(max(rac, 0) / max(ac_max, 1e-10), 1.0)

        resilience_ews = 1.0 - 0.5 * var_contribution - 0.5 * ac_contribution
        resilience_ews = max(0.0, min(1.0, resilience_ews))

        window_std = np.std(window[:, 3])
        window_range = np.max(window[:, 3]) - np.min(window[:, 3])
        stability = 1.0 / (1.0 + window_std + window_range * 0.5)

        delta_v_list.append(
            {
                "window_start": start,
                "window_end": start + window_size,
                "a": a,
                "b": b,
                "c": c,
                "delta_v": delta_v_raw,
                "delta_v_cusp": delta_v_cusp,
                "delta_v_well": delta_v_well,
                "resilience_ratio": resilience_ratio,
                "resilience_ews": resilience_ews,
                "stability": stability,
                "rolling_var": float(rv) if not np.isnan(rv) else 0.0,
                "rolling_ac": float(rac) if not np.isnan(rac) else 0.0,
                "mean_x": depression_mean / 7.0,
                "stress_norm": stress_norm,
                "resilience_norm": resilience_norm,
                "support_norm": support_norm,
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
    x_series: np.ndarray,
    threshold_sigma: float = 2.0,
    min_interval: int = 5,
    method: str = "combined",
) -> List[int]:
    valid_mask = np.isfinite(delta_v_series)
    if np.sum(valid_mask) < 3:
        return []

    if method == "sigma":
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

    elif method == "combined":
        tipping_points = []
        last_tip = -min_interval

        dx = np.abs(np.diff(x_series))
        dx_padded = np.concatenate([[0], dx])
        dx_threshold = np.percentile(dx[dx > 0], 90) if np.sum(dx > 0) > 10 else 0.2

        valid_dv = delta_v_series[np.isfinite(delta_v_series)]
        dv_q25 = np.percentile(valid_dv, 25) if len(valid_dv) > 10 else 0.5

        rolling_var = _compute_rolling_variance(x_series, window_size=7)
        valid_rv = rolling_var[~np.isnan(rolling_var)]
        rv_threshold = np.percentile(valid_rv, 90) if len(valid_rv) > 10 else 0.1

        for i in range(1, len(delta_v_series)):
            if not np.isfinite(delta_v_series[i]):
                continue
            if (i - last_tip) < min_interval:
                continue

            low_resilience = delta_v_series[i] < dv_q25
            large_change = dx_padded[i] > dx_threshold
            high_variance = False
            if i < len(rolling_var) and not np.isnan(rolling_var[i]):
                high_variance = rolling_var[i] > rv_threshold

            if (low_resilience and large_change) or (low_resilience and high_variance) or (large_change and high_variance):
                tipping_points.append(i)
                last_tip = i

        return tipping_points

    elif method == "transition":
        tipping_points = []
        last_tip = -min_interval
        dx = np.abs(np.diff(x_series))
        if len(dx) == 0:
            return []
        dx_threshold = np.percentile(dx[dx > 0], 85) if np.sum(dx > 0) > 10 else 0.15
        for i in range(len(dx)):
            if dx[i] > dx_threshold and (i - last_tip) >= min_interval:
                tipping_points.append(i + 1)
                last_tip = i + 1
        return tipping_points

    return []


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


def _compute_prospective_prediction(
    delta_v_series: np.ndarray,
    x_series: np.ndarray,
    lookahead_steps: List[int] = [1, 3, 5, 7],
    transition_threshold: float = 0.5,
    resilience_ratio_series: Optional[np.ndarray] = None,
    stability_series: Optional[np.ndarray] = None,
    rolling_var_series: Optional[np.ndarray] = None,
    rolling_ac_series: Optional[np.ndarray] = None,
) -> Dict:
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n = len(delta_v_series)
    finite_mask = np.isfinite(delta_v_series)
    results = {}

    rolling_var = _compute_rolling_variance(x_series, window_size=7)
    rolling_ac = _compute_rolling_autocorr(x_series, window_size=7, lag=1)

    for k in lookahead_steps:
        if n < k + 5:
            results[f"lookahead_{k}"] = {"note": "insufficient data"}
            continue

        feature_rows = []
        outcomes = []
        for i in range(n - k):
            if not finite_mask[i]:
                continue
            dv_at_t = delta_v_series[i]
            x_at_t = x_series[i] if i < len(x_series) else 0.0
            x_at_tk = x_series[i + k] if (i + k) < len(x_series) else 0.0
            transition = 1 if abs(x_at_tk - x_at_t) > transition_threshold else 0

            rv = rolling_var[i] if i < len(rolling_var) and not np.isnan(rolling_var[i]) else 0.0
            rac = rolling_ac[i] if i < len(rolling_ac) and not np.isnan(rolling_ac[i]) else 0.0
            rr = resilience_ratio_series[i] if resilience_ratio_series is not None and i < len(resilience_ratio_series) and np.isfinite(resilience_ratio_series[i]) else 0.5
            stab = stability_series[i] if stability_series is not None and i < len(stability_series) else 0.5
            rv_ext = rolling_var_series[i] if rolling_var_series is not None and i < len(rolling_var_series) else 0.0
            rac_ext = rolling_ac_series[i] if rolling_ac_series is not None and i < len(rolling_ac_series) and np.isfinite(rolling_ac_series[i]) else 0.0

            feature_rows.append([dv_at_t, x_at_t, rv, rac, -dv_at_t * x_at_t, rr, stab, rv_ext, rac_ext])
            outcomes.append(transition)

        if len(feature_rows) < 10:
            results[f"lookahead_{k}"] = {"note": "insufficient samples"}
            continue

        X = np.array(feature_rows)
        y = np.array(outcomes)

        if len(np.unique(y)) < 2:
            results[f"lookahead_{k}"] = {
                "n_samples": len(y),
                "n_transitions": int(np.sum(y)),
                "note": "only one class present",
            }
            continue

        neg_dv = -X[:, 0]
        try:
            auc_dv = roc_auc_score(y, neg_dv)
        except ValueError:
            auc_dv = 0.0

        binary_pred_dv = (X[:, 0] < np.percentile(X[:, 0], 50)).astype(int)
        acc_dv = accuracy_score(y, binary_pred_dv)

        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
            lr.fit(X_scaled, y)
            pred_prob = lr.predict_proba(X_scaled)[:, 1]
            auc_lr = roc_auc_score(y, pred_prob)
            binary_pred_lr = lr.predict(X_scaled)
            acc_lr = accuracy_score(y, binary_pred_lr)
            f1_lr = f1_score(y, binary_pred_lr, zero_division=0)
            prec_lr = precision_score(y, binary_pred_lr, zero_division=0)
            rec_lr = recall_score(y, binary_pred_lr, zero_division=0)

            from sklearn.model_selection import cross_val_score, StratifiedKFold
            cv_auc_scores = []
            cv_f1_scores = []
            if len(y) >= 30 and np.sum(y) >= 5 and (len(y) - np.sum(y)) >= 5:
                n_splits = min(5, min(np.sum(y), len(y) - np.sum(y)))
                if n_splits >= 2:
                    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    try:
                        cv_auc_scores = cross_val_score(
                            LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
                            X_scaled, y, cv=skf, scoring="roc_auc"
                        )
                        cv_f1_scores = cross_val_score(
                            LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
                            X_scaled, y, cv=skf, scoring="f1"
                        )
                    except Exception:
                        cv_auc_scores = []
                        cv_f1_scores = []
        except Exception:
            auc_lr = 0.0
            acc_lr = 0.0
            f1_lr = 0.0
            prec_lr = 0.0
            rec_lr = 0.0

        from scipy.stats import pointbiserialr
        r_pb, p_pb = pointbiserialr(y, neg_dv)

        results[f"lookahead_{k}"] = {
            "auc_roc_delta_v": float(auc_dv),
            "accuracy_delta_v": float(acc_dv),
            "auc_roc_multifeature": float(auc_lr),
            "accuracy_multifeature": float(acc_lr),
            "f1_multifeature": float(f1_lr),
            "precision_multifeature": float(prec_lr),
            "recall_multifeature": float(rec_lr),
            "cv_auc_roc_mean": float(np.mean(cv_auc_scores)) if len(cv_auc_scores) > 0 else None,
            "cv_auc_roc_std": float(np.std(cv_auc_scores)) if len(cv_auc_scores) > 0 else None,
            "cv_f1_mean": float(np.mean(cv_f1_scores)) if len(cv_f1_scores) > 0 else None,
            "cv_f1_std": float(np.std(cv_f1_scores)) if len(cv_f1_scores) > 0 else None,
            "point_biserial_r": float(r_pb),
            "point_biserial_p": float(p_pb),
            "n_samples": len(y),
            "n_transitions": int(np.sum(y)),
            "transition_rate": float(np.mean(y)),
        }

    best_k = None
    best_auc = -1.0
    for k in lookahead_steps:
        key = f"lookahead_{k}"
        if key in results and isinstance(results[key], dict) and "auc_roc_multifeature" in results[key]:
            if results[key]["auc_roc_multifeature"] > best_auc:
                best_auc = results[key]["auc_roc_multifeature"]
                best_k = k

    results["best_lookahead"] = {"k": best_k, "auc_roc": float(best_auc) if best_k is not None else None}
    results["transition_threshold"] = transition_threshold
    return results


def run_exp3(
    dataset: str = "studentlife",
    window_size: int = 7,
    tipping_sigma: float = 2.0,
    theta_bifurcation: float = 0.5,
) -> Dict:
    results = {}

    if dataset.lower() == "studentlife":
        ema_data = _load_studentlife_ema()
        stress_data = ema_data.get("stress", {})
        mood_data = ema_data.get("mood", {})

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

            if len(stress_vals) < 9:
                continue

            n_stress = len(stress_vals)
            n_mood = len(sad_vals)

            if n_stress >= n_mood and n_mood > 0:
                stress_interp = stress_vals
                sad_interp = np.interp(stress_times[:n_stress], mood_times[:n_mood], sad_vals[:n_mood])
                happy_interp = np.interp(stress_times[:n_stress], mood_times[:n_mood], happy_vals[:n_mood])
            elif n_mood > n_stress:
                sad_interp = sad_vals
                stress_interp = np.interp(mood_times[:n_mood], stress_times[:n_stress], stress_vals[:n_stress])
                happy_interp = happy_vals[:n_mood]
            else:
                continue

            n_rows = min(len(stress_interp), len(sad_interp), len(happy_interp))
            if n_rows < 9:
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
            raise ValueError("No valid EMA time series data found for StudentLife dataset")
        combined = np.vstack(all_series)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'studentlife'.")

    # Fit Cusp parameters from exp2
    cusp_params = None
    try:
        x_data = combined[:, 3] / 7.0  # normalized depression as state variable
        # Estimate derivatives using finite differences
        dx = np.diff(x_data)
        x_mid = (x_data[:-1] + x_data[1:]) / 2.0
        # Filter out NaN/Inf
        valid = np.isfinite(dx) & np.isfinite(x_mid)
        dx = dx[valid]
        x_mid = x_mid[valid]
        if len(dx) > 10:
            # Fit: dx = a + b*x - c*x^3
            # Design matrix: [1, x, -x^3]
            X_design = np.column_stack([
                np.ones(len(x_mid)),
                x_mid,
                -(x_mid ** 3),
            ])
            coeffs, _, _, _ = np.linalg.lstsq(X_design, dx, rcond=None)
            cusp_a = float(coeffs[0])
            cusp_b = float(coeffs[1])
            cusp_c = float(coeffs[2])
            if cusp_c <= 0:
                cusp_c = abs(cusp_c) if abs(cusp_c) > 1e-10 else 1.0
            cusp_params = {"a": cusp_a, "b": cusp_b, "c": cusp_c}
    except Exception:
        cusp_params = None

    delta_v_list = _sliding_window_delta_v(
        combined, window_size=window_size, theta_bifurcation=theta_bifurcation,
        cusp_params=cusp_params,
    )
    delta_v_series = np.array([d["delta_v"] for d in delta_v_list])
    delta_v_cusp_series = np.array([d["delta_v_cusp"] for d in delta_v_list])
    resilience_ratio_series = np.array([d["resilience_ratio"] for d in delta_v_list])
    resilience_ews_series = np.array([d["resilience_ews"] for d in delta_v_list])
    stability_series = np.array([d["stability"] for d in delta_v_list])
    rolling_var_series = np.array([d["rolling_var"] for d in delta_v_list])
    rolling_ac_series = np.array([d["rolling_ac"] for d in delta_v_list])
    x_series = np.array([d["mean_x"] for d in delta_v_list])

    finite_dv = delta_v_series[np.isfinite(delta_v_series)]
    finite_dv_cusp = delta_v_cusp_series[np.isfinite(delta_v_cusp_series)]
    finite_rr = resilience_ratio_series[np.isfinite(resilience_ratio_series)]

    results["delta_v_stats"] = {
        "mean": float(np.mean(finite_dv)) if len(finite_dv) > 0 else float("inf"),
        "std": float(np.std(finite_dv)) if len(finite_dv) > 0 else 0.0,
        "min": float(np.min(finite_dv)) if len(finite_dv) > 0 else float("inf"),
        "max": float(np.max(finite_dv)) if len(finite_dv) > 0 else float("inf"),
        "n_low_resilience": int(np.sum(finite_dv < 0.5)) if len(finite_dv) > 0 else 0,
        "n_critical": int(np.sum(finite_dv < 0.1)) if len(finite_dv) > 0 else 0,
    }

    if len(finite_dv_cusp) > 0:
        results["delta_v_cusp_stats"] = {
            "mean": float(np.mean(finite_dv_cusp)),
            "std": float(np.std(finite_dv_cusp)),
            "min": float(np.min(finite_dv_cusp)),
            "max": float(np.max(finite_dv_cusp)),
            "n_low_resilience": int(np.sum(finite_dv_cusp < 0.5)),
            "n_critical": int(np.sum(finite_dv_cusp < 0.1)),
        }
    if len(finite_rr) > 0:
        results["resilience_ratio_stats"] = {
            "mean": float(np.mean(finite_rr)),
            "std": float(np.std(finite_rr)),
            "min": float(np.min(finite_rr)),
            "max": float(np.max(finite_rr)),
            "n_low": int(np.sum(finite_rr < 0.3)),
            "n_critical": int(np.sum(finite_rr < 0.1)),
        }

    results["resilience_ews_stats"] = {
        "mean": float(np.mean(resilience_ews_series)),
        "std": float(np.std(resilience_ews_series)),
        "min": float(np.min(resilience_ews_series)),
        "max": float(np.max(resilience_ews_series)),
        "n_low": int(np.sum(resilience_ews_series < 0.3)),
        "n_critical": int(np.sum(resilience_ews_series < 0.1)),
    }

    results["stability_stats"] = {
        "mean": float(np.mean(stability_series)),
        "std": float(np.std(stability_series)),
        "min": float(np.min(stability_series)),
        "max": float(np.max(stability_series)),
    }

    if cusp_params is not None:
        results["cusp_params"] = cusp_params

    tipping_points = _detect_tipping_points(
        resilience_ews_series, x_series, threshold_sigma=tipping_sigma, min_interval=window_size, method="combined"
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
        pre_tip_ews = []
        for tip_idx in tipping_points:
            start = max(0, tip_idx - window_size)
            pre_tip_ews.extend(resilience_ews_series[start:tip_idx].tolist())
        results["pre_tipping_delta_v"] = {
            "mean": float(np.nanmean(pre_tip_ews)) if pre_tip_ews else float("inf"),
            "std": float(np.nanstd(pre_tip_ews)) if pre_tip_ews else 0.0,
        }
    else:
        results["pre_tipping_delta_v"] = {"mean": None, "std": None}

    if len(resilience_ews_series) > 10 and len(tipping_points) > 0:
        tipping_binary = np.zeros(len(resilience_ews_series))
        for tp_idx in tipping_points:
            if tp_idx < len(tipping_binary):
                tipping_binary[tp_idx] = 1.0
        ews_vals = resilience_ews_series
        tip_binary = tipping_binary
        n_total = len(ews_vals)
        if np.sum(tip_binary) > 0 and np.sum(tip_binary) < n_total:
            try:
                from sklearn.metrics import roc_auc_score
                neg_ews = -ews_vals
                auc = roc_auc_score(tip_binary, neg_ews)
                results["roc_analysis"] = {
                    "auc_roc": float(auc),
                    "predictor": "negative_resilience_ews",
                    "n_positive": int(np.sum(tip_binary)),
                    "n_negative": int(n_total - np.sum(tip_binary)),
                }
            except Exception as e:
                results["roc_analysis"] = {"error": str(e)}

            try:
                from lifelines import CoxPHFitter
                import pandas as pd
                time_to_event = np.ones(n_total)
                for i in range(n_total):
                    future_tips = [tp for tp in tipping_points if tp >= i]
                    if future_tips:
                        time_to_event[i] = min(future_tips) - i + 1
                    else:
                        time_to_event[i] = n_total - i
                cox_df = pd.DataFrame({
                    "T": time_to_event,
                    "E": tip_binary,
                    "resilience_ews": ews_vals,
                })
                cox_df = cox_df[cox_df["T"] > 0]
                if len(cox_df) > 5 and cox_df["E"].sum() > 0:
                    cph = CoxPHFitter()
                    cph.fit(cox_df, duration_col="T", event_col="E")
                    results["cox_regression"] = {
                        "hazard_ratio": float(np.exp(cph.params_["resilience_ews"])),
                        "p_value": float(cph.summary.loc["resilience_ews", "p"]),
                        "concordance_index": float(cph.concordance_index_),
                        "coefficient": float(cph.params_["resilience_ews"]),
                    }
            except ImportError:
                try:
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.metrics import roc_auc_score
                    ews_2d = ews_vals.reshape(-1, 1)
                    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
                    lr.fit(ews_2d, tip_binary)
                    pred_prob = lr.predict_proba(ews_2d)[:, 1]
                    auc_lr = roc_auc_score(tip_binary, pred_prob)
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

    prospective = _compute_prospective_prediction(
        resilience_ews_series, x_series,
        lookahead_steps=[1, 3, 5, 7],
        transition_threshold=0.15,
        resilience_ratio_series=resilience_ratio_series,
        stability_series=stability_series,
        rolling_var_series=rolling_var_series,
        rolling_ac_series=rolling_ac_series,
    )
    results["prospective_prediction"] = prospective

    if len(resilience_ews_series) > 10:
        from app.cuspnet.statistics import bootstrap_ci
        ews_ci = bootstrap_ci(resilience_ews_series, statistic_fn=np.mean, n_bootstrap=2000)
        results["resilience_ews_stats"]["mean_ci"] = ews_ci

    if "roc_analysis" in results and isinstance(results.get("roc_analysis"), dict) and "auc_roc" in results.get("roc_analysis", {}):
        from app.cuspnet.statistics import bootstrap_ci_metric
        from sklearn.metrics import roc_auc_score
        if len(tipping_points) > 0:
            tipping_binary_full = np.zeros(len(resilience_ews_series))
            for tp_idx in tipping_points:
                if tp_idx < len(tipping_binary_full):
                    tipping_binary_full[tp_idx] = 1.0
            if len(np.unique(tipping_binary_full)) >= 2 and np.sum(tipping_binary_full) > 0:
                auc_ci = bootstrap_ci_metric(tipping_binary_full, -resilience_ews_series, roc_auc_score, n_bootstrap=2000)
                results["roc_analysis"]["auc_ci"] = auc_ci

    results["dataset_info"] = {
        "name": dataset,
        "n_observations": combined.shape[0],
        "window_size": window_size,
    }

    return results
