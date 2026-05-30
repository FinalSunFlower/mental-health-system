import numpy as np
from typing import Dict, Optional, List
from scipy.integrate import solve_ivp
from .utils import (
    find_fixed_points,
    classify_fixed_points,
    compute_potential,
    compute_resilience_reserve,
    compute_critical_distance,
    sigmoid,
)


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
    ):
        self.theta_bifurcation = theta_bifurcation
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.lambda_c = lambda_c
        self.drift_eta = drift_eta
        self.tau_threshold = tau_threshold
        self.coupling_k = coupling_k

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

        simulation = None
        if x0 is not None and t_span is not None:
            simulation = self.simulate_network_ode(
                x0, a_local, b_local, c_local, causal_adjacency, t_span
            )

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
        pss_norm = pss10 / 50.0
        cdrisc_norm = cdrisc / 40.0
        mspss_norm = mspss / 84.0
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
    ) -> np.ndarray:
        def derivative(t, x):
            coupling = A @ sigmoid(x - self.tau_threshold, k=self.coupling_k)
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

    def compute_tipping_warning(self, b: float) -> Dict:
        cd = compute_critical_distance(b)
        warning = cd < 0.15
        return {"critical_distance": cd, "warning": warning}

    def state_dependent_drift(self, a_current: float, delta_v: float, eta: float = None) -> float:
        if eta is None:
            eta = self.drift_eta
        if abs(delta_v) < 1e-10:
            return a_current + eta * 1e10
        drift_magnitude = eta / abs(delta_v)
        if delta_v > 0:
            return a_current + drift_magnitude
        else:
            return a_current - drift_magnitude
