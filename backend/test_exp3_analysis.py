import numpy as np
import sys
sys.path.insert(0, '.')
from app.experiments.exp3_resilience_prediction import (
    _load_studentlife_ema, _sliding_window_delta_v,
    _compute_rolling_variance, _compute_rolling_autocorr,
)

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
    combined_pid = np.column_stack([stress_interp[:n_rows], resilience[:n_rows], support[:n_rows], depression[:n_rows]])
    all_series.append(combined_pid)

combined = np.vstack(all_series)
delta_v_list = _sliding_window_delta_v(combined, window_size=7)
delta_v_series = np.array([d["delta_v"] for d in delta_v_list])
x_series = np.array([d["mean_x"] for d in delta_v_list])

print("=== Delta V Analysis ===")
print(f"Total windows: {len(delta_v_series)}")
print(f"Delta V stats: mean={np.mean(delta_v_series):.4f}, std={np.std(delta_v_series):.4f}")
print(f"  min={np.min(delta_v_series):.4f}, max={np.max(delta_v_series):.4f}")
print(f"  median={np.median(delta_v_series):.4f}")
print(f"  percentiles: 5%={np.percentile(delta_v_series, 5):.4f}, 10%={np.percentile(delta_v_series, 10):.4f}, 25%={np.percentile(delta_v_series, 25):.4f}")

n_low = np.sum(delta_v_series < 0.5)
n_very_low = np.sum(delta_v_series < 0.1)
n_zero = np.sum(delta_v_series < 0.01)
print(f"  N < 0.5: {n_low}, N < 0.1: {n_very_low}, N < 0.01: {n_zero}")

print("\n=== X Series (Depression) Analysis ===")
print(f"X stats: mean={np.mean(x_series):.4f}, std={np.std(x_series):.4f}")
print(f"  min={np.min(x_series):.4f}, max={np.max(x_series):.4f}")

dx = np.diff(x_series)
print(f"\n=== Changes in X ===")
print(f"dX stats: mean={np.mean(dx):.6f}, std={np.std(dx):.4f}")
print(f"  min={np.min(dx):.4f}, max={np.max(dx):.4f}")
print(f"  |dX| > 0.1: {np.sum(np.abs(dx) > 0.1)} ({np.mean(np.abs(dx) > 0.1)*100:.1f}%)")
print(f"  |dX| > 0.15: {np.sum(np.abs(dx) > 0.15)} ({np.mean(np.abs(dx) > 0.15)*100:.1f}%)")
print(f"  |dX| > 0.2: {np.sum(np.abs(dx) > 0.2)} ({np.mean(np.abs(dx) > 0.2)*100:.1f}%)")

pct_10 = np.percentile(np.abs(dx), 90)
pct_5 = np.percentile(np.abs(dx), 95)
print(f"  |dX| 90th percentile: {pct_10:.4f}")
print(f"  |dX| 95th percentile: {pct_5:.4f}")

print("\n=== Delta V vs X Change Correlation ===")
dv_short = delta_v_series[:-1]
dx_short = dx[:len(dv_short)]
corr = np.corrcoef(dv_short, np.abs(dx_short))[0, 1]
print(f"Correlation(delta_v, |dX|): {corr:.4f}")

low_dv_mask = delta_v_series < np.percentile(delta_v_series, 25)
high_dv_mask = delta_v_series >= np.percentile(delta_v_series, 75)
if np.sum(low_dv_mask) > 5 and np.sum(high_dv_mask) > 5:
    low_dv_x = x_series[low_dv_mask]
    high_dv_x = x_series[high_dv_mask]
    print(f"X when low delta_v (Q1): mean={np.mean(low_dv_x):.4f}, std={np.std(low_dv_x):.4f}")
    print(f"X when high delta_v (Q4): mean={np.mean(high_dv_x):.4f}, std={np.std(high_dv_x):.4f}")

print("\n=== Rolling Variance/Autocorrelation ===")
rolling_var = _compute_rolling_variance(x_series, window_size=7)
rolling_ac = _compute_rolling_autocorr(x_series, window_size=7, lag=1)
valid_var = rolling_var[~np.isnan(rolling_var)]
valid_ac = rolling_ac[~np.isnan(rolling_ac)]
print(f"Rolling variance: mean={np.mean(valid_var):.6f}, std={np.std(valid_var):.6f}")
print(f"Rolling autocorrelation: mean={np.mean(valid_ac):.4f}, std={np.std(valid_ac):.4f}")

print("\n=== Transition Analysis ===")
for thresh in [0.1, 0.15, 0.2, 0.25]:
    n_trans = np.sum(np.abs(dx) > thresh)
    rate = n_trans / len(dx) * 100
    print(f"  |dX| > {thresh}: {n_trans} transitions ({rate:.1f}%)")
