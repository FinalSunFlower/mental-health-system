"""
CUSPNet Engine - Main orchestrator for the three-layer mental health assessment system.
Integrates causal discovery (Layer 1), CUSP catastrophe dynamics (Layer 2), and
Lazarus LLM appraisal (Layer 3) with parameter fusion and risk classification.
"""
import numpy as np
from typing import Dict, Optional, List
from .layer1_causal import CausalDiscoveryLayer, TheoryConstraintEngine
from .layer2_dynamics import CuspDynamicsLayer
from .layer3_llm import LazarusAppraisalChain
from .utils import compute_resilience_reserve, compute_potential


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

QUESTIONNAIRE_RELIABILITY = {
    "pss10": {"alpha": 0.86, "test_retest": 0.72},
    "cdrisc": {"alpha": 0.89, "test_retest": 0.80},
    "mspss": {"alpha": 0.88, "test_retest": 0.85},
}

LLM_PROXY_BASE_RELIABILITY = {"alpha": 0.70, "test_retest": 0.55}


class CuspNetEngine:
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        constraint_theories = config.get("constraint_theories", ["borsboom_network"])
        constraint_config = config.get("constraint_config_path", None)
        constraint_engine = TheoryConstraintEngine(
            theories=constraint_theories,
            config_path=constraint_config,
            min_confidence=config.get("constraint_min_confidence", 0.5),
        )
        self.layer1 = CausalDiscoveryLayer(
            ebic_gamma=config.get("ebic_gamma", 0.5),
            score_threshold=config.get("score_threshold", 0.01),
            constraint_engine=constraint_engine,
        )
        self.layer2 = CuspDynamicsLayer(
            theta_bifurcation=config.get("theta_bifurcation", 0.5),
            lambda_a=config.get("lambda_a", 0.3),
            lambda_b=config.get("lambda_b", 0.3),
            lambda_c=config.get("lambda_c", 0.2),
            drift_eta=config.get("drift_eta", 0.1),
            norm_method=config.get("norm_method", "zscore_to_01"),
            norm_group=config.get("norm_group", "general"),
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
            elif updated_resilience == float("inf"):
                l2_result["drift_prediction"] = {
                    "current_a": float(a_fused),
                    "drifted_a_next": float(a_fused),
                    "delta_a": 0.0,
                    "resilience_at_drift": float(updated_resilience),
                    "note": "monostable_system_no_drift",
                }
            else:
                drifted_a = self.layer2.state_dependent_drift(a_fused, updated_resilience)
                l2_result["drift_prediction"] = {
                    "current_a": float(a_fused),
                    "drifted_a_next": float(drifted_a),
                    "delta_a": float(drifted_a - a_fused),
                    "resilience_at_drift": float(updated_resilience),
                }

            x_range = np.linspace(-2, 2, 200)
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

        if l2_result.get("simulation") is None:
            try:
                a_local, b_local, c_local = self.layer2.allocate_params(
                    l2_result["global_a"], l2_result["global_b"], l2_result["global_c"],
                    centrality, bridge_centrality,
                )
                sim = self.layer2.simulate_network_ode(
                    x0, a_local, b_local, c_local,
                    l1_result["causal_adjacency"], t_span,
                )
                l2_result["simulation"] = sim.tolist()
                if "local_params" not in l2_result or l2_result["local_params"][0].get("a") == l2_result["global_a"]:
                    l2_result["local_params"] = [
                        {
                            "name": variable_names[i] if i < len(variable_names) else f"V{i}",
                            "a": float(a_local[i]),
                            "b": float(b_local[i]),
                            "c": float(c_local[i]),
                        }
                        for i in range(len(a_local))
                    ]
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
        risk_level = self._classify_risk(risk_score, l2_result["tipping_point_warning"], l2_result)

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

        q_reliability = self._estimate_questionnaire_reliability(param_name)
        p_reliability = self._estimate_proxy_reliability(param_name, p_history)

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

            w_q = q_reliability / (q_sigma ** 2)
            w_p = p_reliability / (p_sigma ** 2)
            w_total = w_q + w_p
            if w_total < 1e-20:
                w_total = 1e-20

            z_fused = (w_q * z_q + w_p * z_p) / w_total
            fused_val = z_fused * q_sigma + q_mu
            return float(fused_val)
        else:
            q_norm = (questionnaire_val - q_min) / (q_max - q_min + 1e-10)
            p_norm = (proxy_val - p_min) / (p_max - p_min + 1e-10)
            q_norm = max(0.0, min(1.0, q_norm))
            p_norm = max(0.0, min(1.0, p_norm))

            w_q = q_reliability
            w_p = p_reliability
            w_total = w_q + w_p
            if w_total < 1e-20:
                w_total = 1e-20

            fused_norm = (w_q * q_norm + w_p * p_norm) / w_total
            return float(fused_norm * (q_max - q_min) + q_min)

    def _estimate_questionnaire_reliability(self, param_name: str) -> float:
        param_to_scale = {"a": "pss10", "b": "cdrisc", "c": "mspss"}
        scale = param_to_scale.get(param_name, "pss10")
        rel = QUESTIONNAIRE_RELIABILITY.get(scale, {"alpha": 0.80, "test_retest": 0.70})
        alpha = rel.get("alpha", 0.80)
        test_retest = rel.get("test_retest", 0.70)
        return (alpha + test_retest) / 2.0

    def _estimate_proxy_reliability(self, param_name: str, proxy_history: List[float]) -> float:
        base = (LLM_PROXY_BASE_RELIABILITY["alpha"] + LLM_PROXY_BASE_RELIABILITY["test_retest"]) / 2.0
        if len(proxy_history) >= 3:
            arr = np.array(proxy_history)
            coeff_var = np.std(arr) / (np.abs(np.mean(arr)) + 1e-10)
            consistency_bonus = max(0.0, 0.1 * (1.0 - min(coeff_var, 1.0)))
            return min(base + consistency_bonus, 0.95)
        return base

    def _compute_risk_score(self, dynamics: Dict) -> float:
        a = dynamics.get("global_a", 0.0)
        b = dynamics.get("global_b", 0.0)
        c = dynamics.get("global_c", 1.0)

        is_bistable = dynamics.get("attractor_states", {}).get("is_bistable", False)
        rv = dynamics.get("resilience_reserve", float("inf"))
        cd = dynamics.get("critical_distance", 1.0)

        if is_bistable and rv != float("inf") and rv > 0:
            R_bistable = 1.0 - np.tanh(rv)
        else:
            R_bistable = 0.0

        if is_bistable and cd > 1e-10:
            R_critical = np.exp(-2.0 * cd)
        elif is_bistable and cd <= 1e-10:
            R_critical = 1.0
        else:
            R_critical = 0.0

        fixed_points = dynamics.get("attractor_states", {}).get("fixed_points", [])
        stable_fps = [fp for fp in fixed_points if fp.get("stability") == "stable"]
        unstable_fps = [fp for fp in fixed_points if fp.get("stability") == "unstable"]

        R_attractor = 0.0
        if len(stable_fps) >= 2 and len(unstable_fps) >= 1:
            healthy_fp = min(stable_fps, key=lambda fp: abs(fp.get("value", 0)))
            pathological_fp = max(stable_fps, key=lambda fp: abs(fp.get("value", 0)))
            V_healthy = float(compute_potential(np.array([healthy_fp["value"]]), a, b, c)[0])
            V_pathological = float(compute_potential(np.array([pathological_fp["value"]]), a, b, c)[0])
            V_unstable = float(compute_potential(np.array([unstable_fps[0]["value"]]), a, b, c)[0])

            depth_healthy = V_unstable - V_healthy
            depth_pathological = V_unstable - V_pathological

            if depth_healthy > 1e-10:
                depth_pathological = max(depth_pathological, 0.0)
                attractor_ratio = depth_pathological / (depth_healthy + depth_pathological + 1e-10)
            else:
                attractor_ratio = 1.0
            R_attractor = attractor_ratio
        elif is_bistable and rv != float("inf"):
            R_attractor = 1.0 - np.tanh(rv)

        w_bistable = 0.35
        w_critical = 0.35
        w_attractor = 0.30

        risk = w_bistable * R_bistable + w_critical * R_critical + w_attractor * R_attractor

        risk = max(0.0, min(1.0, risk))
        return float(risk)

    def _classify_risk(self, score: float, tipping_warning: bool, dynamics: Dict = None) -> str:
        if tipping_warning:
            return "critical"

        if dynamics is not None:
            rv = dynamics.get("resilience_reserve", float("inf"))
            cd = dynamics.get("critical_distance", 1.0)
            is_bistable = dynamics.get("attractor_states", {}).get("is_bistable", False)

            if is_bistable and rv < 0.1 and cd < 0.15:
                return "critical"
            if is_bistable and rv < 0.3:
                return "high"
            if is_bistable and score >= 0.4:
                return "high"

        if score >= 0.7:
            return "high"
        if score >= 0.4:
            return "medium"
        return "low"
