import numpy as np
from typing import Dict, Optional
from scipy.integrate import solve_ivp
from scipy.stats import norm
from .utils import (
    find_fixed_points,
    classify_fixed_points,
    compute_potential,
    compute_resilience_reserve,
    compute_critical_distance,
    sigmoid,
)


POPULATION_NORMS = {
    "pss10": {
        "general": {"mean": 19.6, "sd": 6.4, "n": 2387,
                     "citation": "Cohen et al., 1983; Lee, 2012"},
        "clinical": {"mean": 28.3, "sd": 5.8, "n": 450,
                      "citation": "Cohen et al., 1983"},
        "student": {"mean": 23.2, "sd": 5.9, "n": 1240,
                     "citation": "Roberti et al., 2006"},
    },
    "cdrisc": {
        "general": {"mean": 30.5, "sd": 6.3, "n": 1632,
                     "citation": "Connor & Davidson, 2003"},
        "clinical": {"mean": 21.7, "sd": 7.8, "n": 576,
                      "citation": "Connor & Davidson, 2003"},
        "student": {"mean": 27.1, "sd": 6.9, "n": 890,
                     "citation": "Campbell-Sills & Stein, 2007"},
    },
    "mspss": {
        "general": {"mean": 60.2, "sd": 14.8, "n": 1365,
                     "citation": "Zimet et al., 1988"},
        "clinical": {"mean": 42.5, "sd": 17.3, "n": 380,
                      "citation": "Zimet et al., 1988"},
        "student": {"mean": 55.8, "sd": 13.2, "n": 720,
                     "citation": "Zimet et al., 1988"},
    },
}


class NormProvider:
    def __init__(self, norm_group: str = "general", custom_norms: Dict = None):
        self.norm_group = norm_group
        self.norms = POPULATION_NORMS.copy()
        if custom_norms:
            for scale, groups in custom_norms.items():
                if scale not in self.norms:
                    self.norms[scale] = {}
                self.norms[scale][norm_group] = groups

    def zscore(self, scale: str, raw_score: float) -> float:
        norm_data = self.norms.get(scale, {}).get(self.norm_group)
        if norm_data is None:
            fallback = {"pss10": (19.6, 6.4), "cdrisc": (30.5, 6.3), "mspss": (60.2, 14.8)}
            mu, sigma = fallback.get(scale, (0.0, 1.0))
        else:
            mu, sigma = norm_data["mean"], norm_data["sd"]
        return (raw_score - mu) / (sigma + 1e-10)

    def percentile_rank(self, scale: str, raw_score: float) -> float:
        z = self.zscore(scale, raw_score)
        return float(norm.cdf(z)) * 100.0

    def normalize(self, scale: str, raw_score: float, method: str = "zscore_to_01") -> float:
        if method == "zscore_to_01":
            z = self.zscore(scale, raw_score)
            return float(norm.cdf(z))
        elif method == "percentile":
            return self.percentile_rank(scale, raw_score) / 100.0
        elif method == "minmax":
            ranges = {"pss10": (0, 50), "cdrisc": (0, 40), "mspss": (0, 84),
                      "cognitive_reappraisal": (0, 1)}
            lo, hi = ranges.get(scale, (0, 100))
            return (raw_score - lo) / (hi - lo + 1e-10)
        else:
            z = self.zscore(scale, raw_score)
            return float(norm.cdf(z))


class CuspDynamicsLayer:
    def __init__(
        self,
        theta_bifurcation: float = 0.5,
        lambda_a: float = 0.3,
        lambda_b: float = 0.3,
        lambda_c: float = 0.2,
        drift_eta: float = 0.1,
        tau_threshold: float = 0.5,
        coupling_k: float = 10.0,
        norm_method: str = "zscore_to_01",
        norm_group: str = "general",
    ):
        self.theta_bifurcation = theta_bifurcation
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.lambda_c = lambda_c
        self.drift_eta = drift_eta
        self.tau_threshold = tau_threshold
        self.coupling_k = coupling_k
        self.norm_method = norm_method
        self.norm_provider = NormProvider(norm_group=norm_group)

    def full_analysis(
        self,
        pss10: float,
        cdrisc: float,
        mspss: float,
        cognitive_reappraisal: float,
        centrality: np.ndarray,
        bridge: np.ndarray,
        causal_adjacency: np.ndarray,
        x0: Optional[np.ndarray] = None,
        t_span: Optional[np.ndarray] = None,
    ) -> Dict:
        a, b, c = self.estimate_global_params(pss10, cdrisc, mspss, cognitive_reappraisal)
        a_local, b_local, c_local = self.allocate_params(a, b, c, centrality, bridge)
        attractors = self.analyze_attractors(a, b, c)
        resilience = compute_resilience_reserve(a, b, c)
        tipping = self.compute_tipping_warning(b)

        drift_prediction = None
        if resilience != float("inf") and resilience < 1.0:
            drifted_a = self.state_dependent_drift(a, resilience)
            drift_prediction = {
                "current_a": float(a),
                "drifted_a_next": float(drifted_a),
                "delta_a": float(drifted_a - a),
                "resilience_at_drift": float(resilience),
            }
        elif resilience == float("inf"):
            drift_prediction = {
                "current_a": float(a),
                "drifted_a_next": float(a),
                "delta_a": 0.0,
                "resilience_at_drift": float(resilience),
                "note": "monostable_system_no_drift",
            }
        else:
            drifted_a = self.state_dependent_drift(a, resilience)
            drift_prediction = {
                "current_a": float(a),
                "drifted_a_next": float(drifted_a),
                "delta_a": float(drifted_a - a),
                "resilience_at_drift": float(resilience),
            }

        simulation = None
        if x0 is not None and t_span is not None:
            try:
                simulation = self.simulate_network_ode(
                    x0, a_local, b_local, c_local, causal_adjacency, t_span
                )
            except Exception:
                simulation = None

        x_range = np.linspace(-2, 2, 200)
        potential_data = {
            "a": a,
            "b": b,
            "c": c,
            "x_range": x_range.tolist(),
            "V": compute_potential(x_range, a, b, c).tolist(),
        }

        if attractors.get("is_bistable", False) and attractors.get("fixed_points"):
            stable_points = [
                fp for fp in attractors["fixed_points"] if fp["stability"] == "stable"
            ]
            if stable_points:
                potential_data["stable_fixed_points"] = [
                    {"x": fp["value"], "V": float(compute_potential(np.array([fp["value"]]), a, b, c)[0])}
                    for fp in stable_points
                ]
            unstable_points = [
                fp for fp in attractors["fixed_points"] if fp["stability"] == "unstable"
            ]
            if unstable_points:
                potential_data["unstable_fixed_points"] = [
                    {"x": fp["value"], "V": float(compute_potential(np.array([fp["value"]]), a, b, c)[0])}
                    for fp in unstable_points
                ]

        return {
            "global_a": a,
            "global_b": b,
            "global_c": c,
            "local_params": [
                {
                    "name": f"V{i}",
                    "a": float(a_local[i]),
                    "b": float(b_local[i]),
                    "c": float(c_local[i]),
                }
                for i in range(len(a_local))
            ],
            "attractor_states": attractors,
            "resilience_reserve": resilience,
            "critical_distance": tipping["critical_distance"],
            "tipping_point_warning": tipping["warning"],
            "potential_function": potential_data,
            "drift_prediction": drift_prediction,
            "simulation": simulation.tolist() if simulation is not None else None,
        }

    def estimate_global_params(
        self, pss10: float, cdrisc: float, mspss: float, cognitive_reappraisal: float
    ) -> tuple:
        pss_norm = self.norm_provider.normalize("pss10", pss10, method=self.norm_method)
        cdrisc_norm = self.norm_provider.normalize("cdrisc", cdrisc, method=self.norm_method)
        mspss_norm = self.norm_provider.normalize("mspss", mspss, method=self.norm_method)

        a = pss_norm - cdrisc_norm
        b = cdrisc_norm * mspss_norm - self.theta_bifurcation
        c = mspss_norm * cognitive_reappraisal

        return a, b, c

    def allocate_params(
        self, a: float, b: float, c: float, centrality: np.ndarray, bridge: np.ndarray
    ) -> tuple:
        cent_norm = centrality / (np.max(centrality) + 1e-10)
        bridge_norm = bridge / (np.max(bridge) + 1e-10)
        a_local = a * (1 + self.lambda_a * cent_norm)
        b_local = b * (1 - self.lambda_b * cent_norm)
        c_local = c * (1 + self.lambda_c * bridge_norm)
        return a_local, b_local, c_local

    def analyze_attractors(self, a: float, b: float, c: float) -> Dict:
        roots = find_fixed_points(a, b, c)
        classified = classify_fixed_points(roots, b, c)
        stable = [p for p in classified if p["stability"] == "stable"]
        unstable = [p for p in classified if p["stability"] == "unstable"]
        return {
            "fixed_points": classified,
            "is_bistable": len(stable) >= 2,
            "num_stable": len(stable),
            "num_unstable": len(unstable),
        }

    def simulate_network_ode(
        self,
        x0: np.ndarray,
        a_local: np.ndarray,
        b_local: np.ndarray,
        c_local: np.ndarray,
        A: np.ndarray,
        t_span: np.ndarray,
        tau: float = None,
        k: float = None,
    ) -> np.ndarray:
        if tau is None or k is None:
            tau_est, k_est = self.estimate_coupling_params(
                x0, a_local, b_local, c_local, A
            )
            if tau is None:
                tau = tau_est
            if k is None:
                k = k_est

        def derivative(t, x):
            coupling = A @ sigmoid(x - tau, k=k)
            return a_local + b_local * x - c_local * x**3 + coupling

        result = solve_ivp(
            derivative, [t_span[0], t_span[-1]], x0, method="BDF",
            t_eval=t_span, rtol=1e-6, atol=1e-8,
        )
        if not result.success:
            def derivative_decoupled(t, x):
                return a_local + b_local * x - c_local * x**3
            result = solve_ivp(
                derivative_decoupled, [t_span[0], t_span[-1]], x0,
                method="BDF", t_eval=t_span, rtol=1e-6, atol=1e-8,
            )
        return result.y.T

    def estimate_coupling_params(
        self,
        x0: np.ndarray,
        a_local: np.ndarray,
        b_local: np.ndarray,
        c_local: np.ndarray,
        A: np.ndarray,
    ) -> tuple:
        n = len(x0)
        abs_weights = np.abs(A[A != 0]) if np.any(A != 0) else np.array([0.1])

        tau = float(np.median(x0))

        mean_weight = float(np.mean(abs_weights))
        if mean_weight > 1e-10:
            k = np.pi / (2.0 * mean_weight)
        else:
            k = 10.0
        k = max(2.0, min(k, 50.0))

        if n > 1:
            x_range = float(np.max(x0) - np.min(x0))
            if x_range > 1e-10:
                k = max(k, np.pi / x_range)

        return tau, k

    def compute_tipping_warning(self, b: float) -> Dict:
        cd = compute_critical_distance(b)
        warning = cd < 0.15
        return {"critical_distance": cd, "warning": warning}

    def state_dependent_drift(self, a_current: float, delta_v: float, eta: float = None) -> float:
        if eta is None:
            eta = self.drift_eta
        if abs(delta_v) < 1e-10:
            return a_current + eta * np.sign(delta_v + 1e-20)
        drift_magnitude = eta / abs(delta_v)
        drift_magnitude = min(drift_magnitude, abs(a_current) * 0.5 + eta * 10.0)
        if delta_v > 0:
            return a_current + drift_magnitude
        else:
            return a_current - drift_magnitude
