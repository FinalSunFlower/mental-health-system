import numpy as np
from typing import Dict, Optional, List
from .layer1_causal import CausalDiscoveryLayer
from .layer2_dynamics import CuspDynamicsLayer
from .layer3_llm import LazarusAppraisalChain
from .utils import compute_resilience_reserve, find_fixed_points, classify_fixed_points


PARAM_RANGES = {
    "a": {"min": -1.0, "max": 1.0},
    "b": {"min": -0.5, "max": 1.5},
    "c": {"min": 0.0, "max": 2.0},
}

PROXY_RANGES = {
    "a": {"min": -1.0, "max": 1.0},
    "b": {"min": -1.0, "max": 0.5},
    "c": {"min": 0.0, "max": 1.1},
}


class CuspNetEngine:
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.layer1 = CausalDiscoveryLayer(
            ebic_gamma=config.get("ebic_gamma", 0.5),
            score_threshold=config.get("score_threshold", 0.01),
        )
        self.layer2 = CuspDynamicsLayer(
            theta_bifurcation=config.get("theta_bifurcation", 0.5),
            lambda_a=config.get("lambda_a", 0.3),
            lambda_b=config.get("lambda_b", 0.3),
            lambda_c=config.get("lambda_c", 0.2),
            drift_eta=config.get("drift_eta", 0.1),
        )
        self.layer3 = LazarusAppraisalChain(
            model_name=config.get("llm_model_name", r"D:\Models\huggingface\Qwen3.5-2B"),
            device=config.get("llm_device", "cuda"),
            load_in_4bit=config.get("llm_load_in_4bit", False),
            max_new_tokens=config.get("llm_max_new_tokens", 2048),
            temperature=config.get("llm_temperature", 0.3),
        )
        self.config = config
        self._history_a: List[float] = []
        self._history_b: List[float] = []
        self._history_c: List[float] = []
        self._proxy_history_a: List[float] = []
        self._proxy_history_b: List[float] = []
        self._proxy_history_c: List[float] = []
        self._last_result: Optional[Dict] = None

    def assess(
        self,
        questionnaire_data: np.ndarray,
        variable_names: list,
        pss10: float,
        cdrisc: float,
        mspss: float,
        cognitive_reappraisal: float = 0.5,
        text_input: Optional[str] = None,
    ) -> Dict:
        l1_result = self.layer1.fit(questionnaire_data, variable_names)

        centrality = np.array(
            [l1_result["centrality_ranking"][i] for i in range(len(variable_names))]
        )
        bridge_centrality = l1_result.get("bridge_centrality", np.zeros(len(variable_names)))
        if isinstance(bridge_centrality, list):
            bridge_centrality = np.array(bridge_centrality)
        if bridge_centrality.shape[0] != len(variable_names):
            bridge_centrality = np.zeros(len(variable_names))

        p = len(variable_names)
        x0 = np.zeros(p)
        t_span = np.linspace(0, 10, 200)

        l2_result = self.layer2.full_analysis(
            pss10, cdrisc, mspss, cognitive_reappraisal,
            centrality, bridge_centrality, l1_result["causal_adjacency"],
            x0=x0, t_span=t_span,
        )

        l3_result = None
        if text_input:
            causal_info = {
                "central_symptoms": l1_result.get("bridge_symptoms", []),
                "top_loops": l1_result.get("positive_feedback_loops", []),
            }
            dynamics_info = {
                "resilience_reserve": l2_result["resilience_reserve"],
                "critical_distance": l2_result["critical_distance"],
                "tipping_point_warning": l2_result["tipping_point_warning"],
            }
            l3_result = self.layer3.full_chain(text_input, causal_info, dynamics_info)
            proxies = l3_result["cusp_proxies"]

            self._proxy_history_a.append(proxies["a_proxy"])
            self._proxy_history_b.append(proxies["b_proxy"])
            self._proxy_history_c.append(proxies["c_proxy"])

            l2_result["global_a"] = self._fuse_param(
                l2_result["global_a"], proxies["a_proxy"], "a"
            )
            l2_result["global_b"] = self._fuse_param(
                l2_result["global_b"], proxies["b_proxy"], "b"
            )
            l2_result["global_c"] = self._fuse_param(
                l2_result["global_c"], proxies["c_proxy"], "c"
            )

            a_fused = l2_result["global_a"]
            b_fused = l2_result["global_b"]
            c_fused = l2_result["global_c"]

            updated_attractors = self.layer2.analyze_attractors(a_fused, b_fused, c_fused)
            updated_resilience = compute_resilience_reserve(a_fused, b_fused, c_fused)
            updated_tipping = self.layer2.compute_tipping_warning(b_fused)

            l2_result["attractor_states"] = updated_attractors
            l2_result["resilience_reserve"] = updated_resilience
            l2_result["critical_distance"] = updated_tipping["critical_distance"]
            l2_result["tipping_point_warning"] = updated_tipping["warning"]

            if updated_resilience != float("inf") and updated_resilience < 1.0:
                drifted_a = self.layer2.state_dependent_drift(a_fused, updated_resilience)
                l2_result["drift_prediction"] = {
                    "current_a": float(a_fused),
                    "drifted_a_next": float(drifted_a),
                    "delta_a": float(drifted_a - a_fused),
                    "resilience_at_drift": float(updated_resilience),
                }
            else:
                l2_result["drift_prediction"] = None

            x_range = np.linspace(-2, 2, 200)
            from .utils import compute_potential
            l2_result["potential_function"] = {
                "a": a_fused,
                "b": b_fused,
                "c": c_fused,
                "x_range": x_range.tolist(),
                "V": compute_potential(x_range, a_fused, b_fused, c_fused).tolist(),
            }

            a_local, b_local, c_local = self.layer2.allocate_params(
                a_fused, b_fused, c_fused, centrality, bridge_centrality
            )
            l2_result["local_params"] = [
                {
                    "name": variable_names[i] if i < len(variable_names) else f"V{i}",
                    "a": float(a_local[i]),
                    "b": float(b_local[i]),
                    "c": float(c_local[i]),
                }
                for i in range(len(a_local))
            ]

            if l2_result.get("simulation") is not None:
                try:
                    sim = self.layer2.simulate_network_ode(
                        x0, a_local, b_local, c_local,
                        l1_result["causal_adjacency"], t_span,
                    )
                    l2_result["simulation"] = sim.tolist()
                except Exception:
                    l2_result["simulation"] = None

        if l2_result.get("drift_prediction") is not None:
            l2_result["drifted_a_next"] = l2_result["drift_prediction"]["drifted_a_next"]
        elif l2_result["resilience_reserve"] != float("inf") and l2_result["resilience_reserve"] < 1.0:
            drifted_a = self.layer2.state_dependent_drift(
                l2_result["global_a"], l2_result["resilience_reserve"]
            )
            l2_result["drift_prediction"] = {
                "current_a": float(l2_result["global_a"]),
                "drifted_a_next": float(drifted_a),
                "delta_a": float(drifted_a - l2_result["global_a"]),
                "resilience_at_drift": float(l2_result["resilience_reserve"]),
            }
            l2_result["drifted_a_next"] = drifted_a
        else:
            l2_result["drifted_a_next"] = l2_result["global_a"]

        self._history_a.append(l2_result["global_a"])
        self._history_b.append(l2_result["global_b"])
        self._history_c.append(l2_result["global_c"])

        risk_score = self._compute_risk_score(l2_result)
        risk_level = self._classify_risk(risk_score, l2_result["tipping_point_warning"])

        result = {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "causal_network": l1_result,
            "dynamics": l2_result,
            "llm_appraisal": l3_result,
            "model_version": "CuspNet-1.0",
        }
        self._last_result = result
        return result

    def _fuse_param(self, questionnaire_val: float, proxy_val: float, param_name: str) -> float:
        q_range = PARAM_RANGES[param_name]
        q_min, q_max = q_range["min"], q_range["max"]
        p_range = PROXY_RANGES[param_name]
        p_min, p_max = p_range["min"], p_range["max"]

        q_history = {"a": self._history_a, "b": self._history_b, "c": self._history_c}[param_name]
        p_history = {"a": self._proxy_history_a, "b": self._proxy_history_b, "c": self._proxy_history_c}[param_name]

        if len(q_history) >= 3 and len(p_history) >= 3:
            q_arr = np.array(q_history)
            q_mu = np.mean(q_arr)
            q_sigma = np.std(q_arr)
            if q_sigma < 1e-10:
                q_sigma = 1e-10

            p_arr = np.array(p_history)
            p_mu = np.mean(p_arr)
            p_sigma = np.std(p_arr)
            if p_sigma < 1e-10:
                p_sigma = 1e-10

            z_q = (questionnaire_val - q_mu) / q_sigma
            z_p = (proxy_val - p_mu) / p_sigma
            z_fused = 0.5 * z_q + 0.5 * z_p

            fused_val = z_fused * q_sigma + q_mu
            return float(fused_val)
        else:
            q_norm = (questionnaire_val - q_min) / (q_max - q_min + 1e-10)
            p_norm = (proxy_val - p_min) / (p_max - p_min + 1e-10)
            q_norm = max(0.0, min(1.0, q_norm))
            p_norm = max(0.0, min(1.0, p_norm))
            fused_norm = 0.5 * q_norm + 0.5 * p_norm
            return float(fused_norm * (q_max - q_min) + q_min)

    def _compute_risk_score(self, dynamics: Dict) -> float:
        score = 0.0
        if dynamics.get("attractor_states", {}).get("is_bistable", False):
            score += 0.3
        rv = dynamics.get("resilience_reserve", float("inf"))
        if rv < 0.1:
            score += 0.4
        elif rv < 0.5:
            score += 0.2
        cd = dynamics.get("critical_distance", 1.0)
        if cd < 0.15:
            score += 0.3
        elif cd < 0.3:
            score += 0.1
        return min(1.0, score)

    def _classify_risk(self, score: float, tipping_warning: bool) -> str:
        if tipping_warning:
            return "critical"
        if score >= 0.7:
            return "high"
        if score >= 0.4:
            return "medium"
        return "low"
