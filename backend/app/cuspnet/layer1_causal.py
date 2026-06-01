"""
Layer 1: Causal Discovery Module.
Implements EBIC-Glasso for precision matrix estimation, Score-based DAG search,
and theory-constrained causal orientation using psychological domain knowledge
(Borsboom network theory, DSM-5, Sachs pathway).
"""
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
import networkx as nx
from .utils import detect_positive_feedback_loops, compute_loop_strength


def _ensure_r_env():
    if os.environ.get("R_HOME"):
        return
    import winreg
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\R-core\R64", 0, winreg.KEY_READ
        )
        r_home, _ = winreg.QueryValueEx(key, "InstallPath")
        winreg.CloseKey(key)
        os.environ["R_HOME"] = r_home
    except Exception:
        for candidate in [
            r"C:\Program Files\R\R-4.6.0",
            r"C:\Program Files\R\R-4.5.0",
            r"C:\Program Files\R\R-4.4.0",
        ]:
            if os.path.isdir(candidate):
                os.environ["R_HOME"] = candidate
                break
    r_home = os.environ.get("R_HOME", "")
    bin_dir = os.path.join(r_home, "bin", "x64")
    if bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_dir + ";" + os.environ.get("PATH", "")


class TheoryConstraint:
    def __init__(
        self,
        cause: str,
        effect: str,
        theory: str = "borsboom_network",
        confidence: float = 1.0,
        citation: str = "",
    ):
        self.cause = cause
        self.effect = effect
        self.theory = theory
        self.confidence = max(0.0, min(1.0, confidence))
        self.citation = citation

    def to_dict(self) -> Dict:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "theory": self.theory,
            "confidence": self.confidence,
            "citation": self.citation,
        }


class TheoryConstraintEngine:
    BUILTIN_BORBOOM = [
        TheoryConstraint("insomnia", "fatigue", "borsboom_network", 0.85,
                         "Borsboom, 2017; Robinaugh et al., 2020"),
        TheoryConstraint("fatigue", "concentration", "borsboom_network", 0.80,
                         "Borsboom, 2017; Cramer et al., 2016"),
        TheoryConstraint("anhedonia", "depressed_mood", "borsboom_network", 0.90,
                         "Borsboom, 2017; Fried et al., 2016"),
        TheoryConstraint("worry", "restlessness", "borsboom_network", 0.75,
                         "Borsboom, 2017"),
        TheoryConstraint("stress", "anxiety", "borsboom_network", 0.80,
                         "Borsboom, 2017; Cramer et al., 2016"),
        TheoryConstraint("anxiety", "insomnia", "borsboom_network", 0.75,
                         "Borsboom, 2017; Blanken et al., 2021"),
        TheoryConstraint("social_isolation", "depressed_mood", "borsboom_network", 0.70,
                         "Borsboom, 2017; Fried et al., 2016"),
    ]

    BUILTIN_DSM5 = [
        TheoryConstraint("depressed_mood", "suicidal_ideation", "dsm5", 0.70,
                         "APA DSM-5, 2013"),
        TheoryConstraint("insomnia", "fatigue", "dsm5", 0.85,
                         "APA DSM-5, 2013"),
        TheoryConstraint("appetite_change", "weight_change", "dsm5", 0.90,
                         "APA DSM-5, 2013"),
        TheoryConstraint("psychomotor_agitation", "restlessness", "dsm5", 0.75,
                         "APA DSM-5, 2013"),
        TheoryConstraint("concentration", "indecisiveness", "dsm5", 0.80,
                         "APA DSM-5, 2013"),
        TheoryConstraint("worthlessness", "depressed_mood", "dsm5", 0.75,
                         "APA DSM-5, 2013"),
    ]

    BUILTIN_NETWORK_THEORY = [
        TheoryConstraint("rumination", "depressed_mood", "network_theory", 0.80,
                         "Robinaugh et al., 2020; Fried et al., 2016"),
        TheoryConstraint("avoidance", "anxiety", "network_theory", 0.75,
                         "Robinaugh et al., 2020"),
        TheoryConstraint("negative_affect", "rumination", "network_theory", 0.70,
                         "Robinaugh et al., 2020"),
    ]

    BUILTIN_SACHS = [
        TheoryConstraint("PKC", "PKA", "sachs_pathway", 0.90,
                         "Sachs et al., 2005; known signaling cascade"),
        TheoryConstraint("PKC", "Raf", "sachs_pathway", 0.90,
                         "Sachs et al., 2005; known signaling cascade"),
        TheoryConstraint("PKA", "Raf", "sachs_pathway", 0.85,
                         "Sachs et al., 2005; known signaling cascade"),
        TheoryConstraint("PKA", "Mek", "sachs_pathway", 0.85,
                         "Sachs et al., 2005; known signaling cascade"),
        TheoryConstraint("PKA", "Akt", "sachs_pathway", 0.80,
                         "Sachs et al., 2005; known signaling cascade"),
        TheoryConstraint("Raf", "Mek", "sachs_pathway", 0.90,
                         "Sachs et al., 2005; known signaling cascade"),
        TheoryConstraint("Mek", "Erk", "sachs_pathway", 0.90,
                         "Sachs et al., 2005; known signaling cascade"),
        TheoryConstraint("Akt", "PIP3", "sachs_pathway", 0.80,
                         "Sachs et al., 2005; known signaling cascade"),
        TheoryConstraint("Jnk", "P38", "sachs_pathway", 0.80,
                         "Sachs et al., 2005; known signaling cascade"),
        TheoryConstraint("Plcg", "PIP2", "sachs_pathway", 0.85,
                         "Sachs et al., 2005; known signaling cascade"),
        TheoryConstraint("PIP3", "PIP2", "sachs_pathway", 0.80,
                         "Sachs et al., 2005; known signaling cascade"),
        TheoryConstraint("PIP2", "Plcg", "sachs_pathway", 0.75,
                         "Sachs et al., 2005; known feedback loop"),
    ]

    AVAILABLE_THEORIES = {
        "borsboom_network": BUILTIN_BORBOOM,
        "dsm5": BUILTIN_DSM5,
        "network_theory": BUILTIN_NETWORK_THEORY,
        "sachs_pathway": BUILTIN_SACHS,
    }

    def __init__(
        self,
        theories: List[str] = None,
        config_path: str = None,
        min_confidence: float = 0.5,
    ):
        self.constraints: List[TheoryConstraint] = []
        self.min_confidence = min_confidence
        self._name_aliases: Dict[str, List[str]] = {}
        self._load_builtin(theories or ["borsboom_network"])
        if config_path:
            self._load_from_config(config_path)
        self._build_name_aliases()

    def _load_builtin(self, theories: List[str]):
        for theory_name in theories:
            builtin = self.AVAILABLE_THEORIES.get(theory_name)
            if builtin is not None:
                self.constraints.extend([
                    TheoryConstraint(c.cause, c.effect, c.theory, c.confidence, c.citation)
                    for c in builtin
                ])

    def _load_from_config(self, config_path: str):
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        for item in config.get("constraints", []):
            self.constraints.append(TheoryConstraint(
                cause=item["cause"],
                effect=item["effect"],
                theory=item.get("theory", "custom"),
                confidence=item.get("confidence", 1.0),
                citation=item.get("citation", ""),
            ))

    def _build_name_aliases(self):
        self._name_aliases = {
            "insomnia": ["sleep", "sleep_problems", "insomnia_severity", "sleep_disturbance"],
            "fatigue": ["tiredness", "low_energy", "exhaustion", "energy"],
            "concentration": ["focus", "attention", "concentration_problems"],
            "anhedonia": ["loss_of_interest", "interest", "pleasure_loss"],
            "depressed_mood": ["depression", "sadness", "low_mood", "mood"],
            "worry": ["rumination", "repetitive_thinking"],
            "restlessness": ["agitation", "psychomotor_agitation", "fidgeting"],
            "stress": ["perceived_stress", "distress"],
            "anxiety": ["anxious", "nervousness", "tension"],
            "social_isolation": ["loneliness", "social_withdrawal", "social_support_low"],
            "suicidal_ideation": ["suicidal", "suicide", "self_harm"],
            "appetite_change": ["appetite", "appetite_loss"],
            "weight_change": ["weight", "weight_loss", "weight_gain"],
            "worthlessness": ["worth", "self_worth", "guilt"],
            "avoidance": ["avoidant", "avoidance_behavior"],
            "rumination": ["repetitive_thinking", "brooding"],
            "negative_affect": ["negative_emotion", "neg_affect"],
            "indecisiveness": ["indecision", "decision_problems"],
            "psychomotor_agitation": ["agitation", "restlessness", "motor_agitation"],
        }

    def _resolve_name(self, name: str, variable_names: List[str]) -> str:
        name_norm = name.lower().replace("_", "").replace("-", "")
        for vname in variable_names:
            if vname.lower().replace("_", "").replace("-", "") == name_norm:
                return vname
        aliases = self._name_aliases.get(name, [])
        for alias in aliases:
            alias_norm = alias.lower().replace("_", "").replace("-", "")
            for vname in variable_names:
                vname_norm = vname.lower().replace("_", "").replace("-", "")
                if alias_norm == vname_norm or alias_norm in vname_norm or vname_norm in alias_norm:
                    return vname
        for vname in variable_names:
            vname_norm = vname.lower().replace("_", "").replace("-", "")
            if name_norm in vname_norm or vname_norm in name_norm:
                return vname
        return None

    def add_constraint(self, cause: str, effect: str, theory: str = "custom",
                       confidence: float = 1.0, citation: str = ""):
        self.constraints.append(TheoryConstraint(cause, effect, theory, confidence, citation))

    def remove_constraint(self, cause: str, effect: str):
        self.constraints = [
            c for c in self.constraints
            if not (c.cause == cause and c.effect == effect)
        ]

    def get_applicable_constraints(self, variable_names: List[str]) -> List[Dict]:
        result = []
        seen_edges = {}
        for c in self.constraints:
            if c.confidence < self.min_confidence:
                continue
            resolved_cause = self._resolve_name(c.cause, variable_names)
            resolved_effect = self._resolve_name(c.effect, variable_names)
            if resolved_cause is not None and resolved_effect is not None:
                edge_key = (resolved_cause, resolved_effect)
                entry = {
                    "cause": resolved_cause,
                    "effect": resolved_effect,
                    "original_cause": c.cause,
                    "original_effect": c.effect,
                    "theory": c.theory,
                    "confidence": c.confidence,
                    "citation": c.citation,
                }
                if edge_key in seen_edges:
                    existing = seen_edges[edge_key]
                    if c.confidence > existing["confidence"]:
                        seen_edges[edge_key] = entry
                else:
                    seen_edges[edge_key] = entry
        result = list(seen_edges.values())
        return result

    def apply_constraints(
        self,
        A: np.ndarray,
        partial_corr: np.ndarray,
        names: List[str],
        fallback_weight: float = 0.1,
        min_partial_corr: float = 0.05,
    ) -> np.ndarray:
        A = A.copy()
        applicable = self.get_applicable_constraints(names)
        name_to_idx = {name: idx for idx, name in enumerate(names)}
        for constraint in applicable:
            cause_idx = name_to_idx.get(constraint["cause"])
            effect_idx = name_to_idx.get(constraint["effect"])
            if cause_idx is None or effect_idx is None:
                continue
            pc_val = partial_corr[cause_idx, effect_idx]
            if abs(pc_val) > min_partial_corr:
                A[cause_idx, effect_idx] = pc_val * constraint["confidence"]
                A[effect_idx, cause_idx] = 0.0
            elif A[cause_idx, effect_idx] != 0 or A[effect_idx, cause_idx] != 0:
                if A[cause_idx, effect_idx] != 0:
                    A[cause_idx, effect_idx] = abs(A[cause_idx, effect_idx]) * constraint["confidence"]
                    A[effect_idx, cause_idx] = 0.0
                else:
                    A[cause_idx, effect_idx] = abs(A[effect_idx, cause_idx]) * constraint["confidence"]
                    A[effect_idx, cause_idx] = 0.0
            else:
                A[cause_idx, effect_idx] = fallback_weight * constraint["confidence"]
                A[effect_idx, cause_idx] = 0.0
        return A


class CausalDiscoveryLayer:
    def __init__(self, ebic_gamma: float = 0.5, score_threshold: float = 0.1,
                 constraint_engine: TheoryConstraintEngine = None):
        self.ebic_gamma = ebic_gamma
        self.score_threshold = score_threshold
        self.constraint_engine = constraint_engine or TheoryConstraintEngine()

    def fit(self, X: np.ndarray, variable_names: List[str] = None) -> Dict:
        if variable_names is None:
            variable_names = [f"V{i}" for i in range(X.shape[1])]
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        precision_matrix, partial_corr = self._ebic_glasso(X_std, variable_names)
        topological_order, score_adj, score_skeleton = self._score_algorithm(X_std, variable_names)
        causal_adjacency = self._theory_constrained_orientation(
            partial_corr, topological_order, variable_names, score_adj, score_skeleton
        )
        centrality_ranking, bridge_centrality = self._compute_centrality(
            causal_adjacency, variable_names
        )
        loops = detect_positive_feedback_loops(causal_adjacency)
        loop_strengths = [
            {"loop": loop, "strength": compute_loop_strength(causal_adjacency, loop)}
            for loop in loops
        ]
        return {
            "precision_matrix": precision_matrix,
            "partial_correlation": partial_corr,
            "causal_adjacency": causal_adjacency,
            "topological_order": topological_order,
            "centrality_ranking": centrality_ranking,
            "bridge_centrality": bridge_centrality,
            "bridge_symptoms": [
                v
                for v, b in sorted(
                    zip(variable_names, bridge_centrality),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
            ],
            "positive_feedback_loops": loop_strengths,
            "variable_names": variable_names,
        }

    def _ebic_glasso(self, X: np.ndarray, variable_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        n, p = X.shape
        if variable_names is None:
            variable_names = [f"V{i}" for i in range(p)]
        if n < 2 or p < 2:
            return self._theory_prior_precision_corr(p, None, variable_names)
        S = np.cov(X, rowvar=False)
        try:
            partial_corr = self._ebic_glasso_rpy2(S, n, p)
            precision_matrix = self._partial_corr_to_precision(partial_corr, S)
        except Exception:
            try:
                precision_matrix = self._ebic_glasso_python(X, S, n, p)
                partial_corr = self._precision_to_partial_corr(precision_matrix)
            except Exception:
                return self._theory_prior_precision_corr(p, X, variable_names)
        return precision_matrix, partial_corr

    def _ebic_glasso_rpy2(self, S: np.ndarray, n: int, p: int) -> np.ndarray:
        _ensure_r_env()
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr

        with localconverter(ro.default_converter + numpy2ri.converter):
            qgraph = importr("qgraph")
            if np.allclose(np.diag(S), 1.0, atol=1e-6):
                R = S
            else:
                diag_std = np.sqrt(np.diag(S))
                R = S / np.outer(diag_std, diag_std)
                np.fill_diagonal(R, 1.0)
            eigvals = np.linalg.eigvalsh(R)
            if eigvals.min() < 1e-6:
                R = R + (1e-6 - eigvals.min()) * np.eye(p)
            result = qgraph.EBICglasso(
                R, n, gamma=self.ebic_gamma, nlambda=100,
                lambda_min_ratio=0.01
            )
            partial_corr = np.array(result)
        return partial_corr

    def _ebic_glasso_python(
        self, X: np.ndarray, S: np.ndarray, n: int, p: int
    ) -> np.ndarray:
        from sklearn.covariance import GraphicalLasso

        R = S.copy()
        diag_std = np.sqrt(np.diag(R))
        denom = np.outer(diag_std, diag_std)
        denom = np.where(denom < 1e-10, 1e-10, denom)
        R = R / denom
        np.fill_diagonal(R, 1.0)

        lambda_max = np.max(np.abs(R - np.eye(p)))
        lambda_min = 0.01 * lambda_max
        nlambda = 100
        if lambda_max < 1e-10:
            return np.linalg.pinv(S)
        lambdas = np.exp(
            np.linspace(np.log(lambda_min), np.log(lambda_max), nlambda)
        )

        n_subsample = 20
        subsample_frac = 0.8
        rng = np.random.RandomState(42)
        subsample_indices = [
            rng.choice(n, size=int(n * subsample_frac), replace=False)
            for _ in range(n_subsample)
        ]

        stability_path = np.zeros((nlambda, p, p))
        for s_idx, s_inds in enumerate(subsample_indices):
            X_sub = X[s_inds]
            S_sub = np.cov(X_sub, rowvar=False)
            for lam_idx, lam in enumerate(lambdas):
                try:
                    gl = GraphicalLasso(alpha=lam, max_iter=500, tol=1e-4)
                    gl.fit(X_sub)
                    Theta_sub = gl.precision_
                    diag_sqrt = np.sqrt(np.abs(np.diag(Theta_sub)))
                    diag_sqrt = np.where(diag_sqrt < 1e-10, 1e-10, diag_sqrt)
                    pc_sub = -Theta_sub / np.outer(diag_sqrt, diag_sqrt)
                    np.fill_diagonal(pc_sub, 0)
                    edge_sub = (np.abs(pc_sub) > 1e-6).astype(float)
                    stability_path[lam_idx] += edge_sub
                except Exception:
                    pass
        stability_path /= n_subsample

        stars_beta = 0.05
        stars_D = np.zeros(nlambda)
        for lam_idx in range(nlambda):
            for i in range(p):
                for j in range(i + 1, p):
                    p_hat = stability_path[lam_idx, i, j]
                    stars_D[lam_idx] += 2 * p_hat * (1 - p_hat)
        stars_D /= (p * (p - 1) / 2)

        best_lam_idx = nlambda - 1
        for lam_idx in range(nlambda):
            if stars_D[lam_idx] <= stars_beta:
                best_lam_idx = lam_idx
                break

        if best_lam_idx > 0 and stars_D[best_lam_idx - 1] <= stars_beta:
            for lam_idx in range(best_lam_idx, -1, -1):
                if stars_D[lam_idx] > stars_beta:
                    best_lam_idx = lam_idx + 1
                    break

        best_lam_stars = lambdas[best_lam_idx]

        best_ebic = np.inf
        best_precision = None
        for lam in lambdas:
            if lam < best_lam_stars * 0.5:
                continue
            try:
                gl = GraphicalLasso(alpha=lam, max_iter=1000, tol=1e-6)
                gl.fit(X)
                Theta = gl.precision_
                sign, logdet = np.linalg.slogdet(Theta)
                if sign <= 0:
                    continue
                trace_ST = np.trace(S @ Theta)
                E_mask = np.abs(np.triu(Theta, k=1)) > 1e-8
                E_count = int(np.sum(E_mask))
                ebic = (
                    -n * logdet
                    + n * trace_ST
                    + E_count * np.log(n)
                    + 4 * self.ebic_gamma * E_count * np.log(p)
                )
                if ebic < best_ebic:
                    best_ebic = ebic
                    best_precision = Theta.copy()
            except Exception:
                continue
        if best_precision is None:
            best_precision = np.linalg.pinv(S)
        return best_precision

    def _precision_to_partial_corr(self, precision: np.ndarray) -> np.ndarray:
        diag_sqrt = np.sqrt(np.diag(precision))
        denom = np.outer(diag_sqrt, diag_sqrt)
        denom = np.where(denom < 1e-10, 1e-10, denom)
        partial_corr = -precision / denom
        np.fill_diagonal(partial_corr, 0.0)
        return partial_corr

    def _partial_corr_to_precision(
        self, partial_corr: np.ndarray, S: np.ndarray
    ) -> np.ndarray:
        R = S.copy()
        diag_std = np.sqrt(np.diag(R))
        denom = np.outer(diag_std, diag_std)
        denom = np.where(denom < 1e-10, 1e-10, denom)
        R = R / denom
        np.fill_diagonal(R, 1.0)

        omega = -partial_corr.copy()
        np.fill_diagonal(omega, 1.0)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(S)))
        precision = D_inv_sqrt @ omega @ D_inv_sqrt
        return precision

    def _score_algorithm(self, X: np.ndarray, variable_names: List[str] = None) -> Tuple[List[int], np.ndarray, np.ndarray]:
        try:
            return self._score_via_causal_learn(X)
        except Exception:
            return self._score_fallback_pc_ges(X, variable_names)

    def _score_via_causal_learn(self, X: np.ndarray) -> Tuple[List[int], np.ndarray, np.ndarray]:
        p = X.shape[1]
        adj = np.zeros((p, p))
        skeleton = np.zeros((p, p))
        cpdag_graph = None
        try:
            from causallearn.search.ScoreBased.GES import ges
            Record = ges(X)
            cpdag_graph = Record["G"]
            for i in range(p):
                for j in range(p):
                    if i != j:
                        if cpdag_graph.graph[j, i] == 1 and cpdag_graph.graph[i, j] == -1:
                            adj[i, j] = 1.0
                            skeleton[i, j] = 1.0
                            skeleton[j, i] = 1.0
                        elif cpdag_graph.graph[i, j] == -1 and cpdag_graph.graph[j, i] == -1:
                            skeleton[i, j] = 1.0
                            skeleton[j, i] = 1.0
        except Exception:
            pass
        if cpdag_graph is not None:
            topo = self._extract_topo_from_cgm(cpdag_graph, p)
            return topo, adj, skeleton
        try:
            from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
            result = bic_exact_search(X, search_method="astar")
            if isinstance(result, tuple) and len(result) >= 1:
                bic_adj = np.array(result[0])
                adj = bic_adj.copy()
                for i in range(p):
                    for j in range(p):
                        if adj[i, j] != 0:
                            skeleton[i, j] = 1.0
                            skeleton[j, i] = 1.0
            elif hasattr(result, "G"):
                G = result.G
                for i in range(p):
                    for j in range(p):
                        if i != j and G.graph[j, i] == 1 and G.graph[i, j] == -1:
                            adj[i, j] = 1.0
                            skeleton[i, j] = 1.0
                            skeleton[j, i] = 1.0
            if np.any(adj != 0):
                return self._extract_topo_from_adj(adj, p), adj, skeleton
        except (ImportError, TypeError, Exception):
            pass
        raise RuntimeError("score_via_causal_learn: no causal structure discovered")

    def _extract_topo_from_adj(self, adj: np.ndarray, p: int) -> List[int]:
        dag = nx.DiGraph()
        dag.add_nodes_from(range(p))
        for i in range(p):
            for j in range(p):
                if i != j and adj[i, j] != 0:
                    dag.add_edge(i, j)
        try:
            topo_order = list(nx.topological_sort(dag))
        except nx.NetworkXUnfeasible:
            topo_order = list(range(p))
        existing = set(topo_order)
        missing = [i for i in range(p) if i not in existing]
        topo_order.extend(missing)
        return topo_order

    def _extract_topo_from_cgm(self, G, p: int) -> List[int]:
        dag = nx.DiGraph()
        dag.add_nodes_from(range(p))
        for i in range(p):
            for j in range(p):
                if i != j and G.graph[j, i] == 1 and G.graph[i, j] == -1:
                    dag.add_edge(i, j)
        try:
            topo_order = list(nx.topological_sort(dag))
        except nx.NetworkXUnfeasible:
            topo_order = list(range(p))
        existing = set(topo_order)
        missing = [i for i in range(p) if i not in existing]
        topo_order.extend(missing)
        return topo_order

    def _score_fallback_pc_ges(self, X: np.ndarray, variable_names: List[str] = None) -> Tuple[List[int], np.ndarray, np.ndarray]:
        p = X.shape[1]
        if variable_names is None:
            variable_names = [f"V{i}" for i in range(p)]
        G = None
        adj = np.zeros((p, p))
        skeleton = np.zeros((p, p))
        try:
            from causallearn.search.ScoreBased.GES import ges

            Record = ges(X)
            G = Record["G"]
        except Exception:
            try:
                from causallearn.search.ConstraintBased.PC import pc

                cg = pc(X, alpha=self.score_threshold)
                G = cg.G
            except Exception:
                return self._theory_prior_score(p, variable_names)
        dag = nx.DiGraph()
        dag.add_nodes_from(range(p))
        for i in range(p):
            for j in range(p):
                if i != j and G.graph[j, i] == 1 and G.graph[i, j] == -1:
                    dag.add_edge(i, j)
                    adj[i, j] = 1.0
                    skeleton[i, j] = 1.0
                    skeleton[j, i] = 1.0
        for i in range(p):
            for j in range(i + 1, p):
                if G.graph[i, j] == -1 and G.graph[j, i] == -1:
                    skeleton[i, j] = 1.0
                    skeleton[j, i] = 1.0
                    if not dag.has_edge(j, i) and not nx.has_path(dag, j, i):
                        dag.add_edge(i, j)
                        adj[i, j] = 1.0
                    elif not dag.has_edge(i, j) and not nx.has_path(dag, i, j):
                        dag.add_edge(j, i)
                        adj[j, i] = 1.0
        try:
            topo_order = list(nx.topological_sort(dag))
        except nx.NetworkXUnfeasible:
            topo_order = list(range(p))
        existing = set(topo_order)
        missing = [i for i in range(p) if i not in existing]
        topo_order.extend(missing)
        return topo_order, adj, skeleton

    def _theory_constrained_orientation(
        self, partial_corr: np.ndarray, topo_order: List[int], names: List[str],
        score_adj: np.ndarray = None, score_skeleton: np.ndarray = None
    ) -> np.ndarray:
        p = len(names)
        A = np.zeros((p, p))

        combined_skeleton = np.zeros((p, p))
        if score_skeleton is not None:
            combined_skeleton = score_skeleton.copy()

        for i in range(p):
            for j in range(i + 1, p):
                if combined_skeleton[i, j] == 0:
                    if abs(partial_corr[i, j]) > self.score_threshold:
                        combined_skeleton[i, j] = 1.0
                        combined_skeleton[j, i] = 1.0

        topo_rank = {node: rank for rank, node in enumerate(topo_order)}

        dag = nx.DiGraph()
        dag.add_nodes_from(range(p))

        if score_adj is not None:
            for i in range(p):
                for j in range(p):
                    if i == j:
                        continue
                    if score_adj[i, j] != 0 and combined_skeleton[i, j] > 0:
                        A[i, j] = partial_corr[i, j] if abs(partial_corr[i, j]) > 1e-10 else 0.1
                        dag.add_edge(i, j)

        for i in range(p):
            for j in range(i + 1, p):
                if combined_skeleton[i, j] > 0:
                    if A[i, j] == 0 and A[j, i] == 0:
                        pc_val = partial_corr[i, j]
                        if abs(pc_val) < 1e-10:
                            continue
                        i_before_j = topo_rank.get(i, p) < topo_rank.get(j, p)
                        can_i_to_j = i_before_j and not nx.has_path(dag, j, i)
                        can_j_to_i = not i_before_j and not nx.has_path(dag, i, j)
                        if can_i_to_j and not can_j_to_i:
                            A[i, j] = pc_val
                            dag.add_edge(i, j)
                        elif can_j_to_i and not can_i_to_j:
                            A[j, i] = pc_val
                            dag.add_edge(j, i)
                        elif can_i_to_j and can_j_to_i:
                            if abs(pc_val) > 1e-10:
                                A[i, j] = pc_val
                                dag.add_edge(i, j)
                        else:
                            if not nx.has_path(dag, j, i):
                                A[i, j] = pc_val
                                dag.add_edge(i, j)
                            elif not nx.has_path(dag, i, j):
                                A[j, i] = pc_val
                                dag.add_edge(j, i)

        nonzero_edges = A[A != 0]
        if len(nonzero_edges) > 0:
            fallback_weight = float(np.percentile(np.abs(nonzero_edges), 25))
        else:
            fallback_weight = 0.05
        A = self.constraint_engine.apply_constraints(A, partial_corr, names, fallback_weight)
        A = self._prune_indirect_effects(A, partial_corr, X_std=None)
        return A

    def _prune_indirect_effects(
        self, A: np.ndarray, partial_corr: np.ndarray, X_std: np.ndarray = None
    ) -> np.ndarray:
        p = A.shape[0]
        A = A.copy()
        edges_to_check = []
        for i in range(p):
            for j in range(p):
                if A[i, j] != 0:
                    edges_to_check.append((i, j))

        for i, j in edges_to_check:
            intermediates = []
            for k in range(p):
                if k == i or k == j:
                    continue
                if A[i, k] != 0 and A[k, j] != 0:
                    intermediates.append(k)

            if intermediates:
                direct_pc = abs(partial_corr[i, j])
                for k in intermediates:
                    indirect_pc = min(abs(partial_corr[i, k]), abs(partial_corr[k, j]))
                    if direct_pc < indirect_pc and direct_pc < 0.15:
                        A[i, j] = 0.0
                        break
                    if direct_pc < 0.15 and indirect_pc > 0.1 and direct_pc < indirect_pc * 2:
                        A[i, j] = 0.0
                        break

        return A

    def _compute_centrality(
        self, A: np.ndarray, names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        ei = np.sum(np.abs(A), axis=1)
        p = A.shape[0]
        bridge = np.zeros(p)
        if p < 4:
            return ei, bridge
        G = nx.Graph()
        G.add_nodes_from(range(p))
        for i in range(p):
            for j in range(i + 1, p):
                w = abs(A[i, j]) + abs(A[j, i])
                if w > 1e-10:
                    G.add_edge(i, j, weight=w)
        if G.number_of_edges() == 0:
            return ei, bridge
        try:
            communities = list(
                nx.community.greedy_modularity_communities(G, weight="weight")
            )
        except Exception:
            return ei, bridge
        if len(communities) < 2:
            return ei, bridge
        node_community = {}
        for comm_idx, comm in enumerate(communities):
            for node in comm:
                node_community[node] = comm_idx
        for i in range(p):
            for j in range(p):
                if i != j and node_community.get(i) != node_community.get(j):
                    bridge[i] += abs(A[i, j]) + abs(A[j, i])
        return ei, bridge

    def _theory_prior_precision_corr(self, p: int, X: Optional[np.ndarray], variable_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        partial_corr = np.zeros((p, p))
        if variable_names is None:
            variable_names = [f"V{i}" for i in range(p)]
        if X is not None and X.shape[0] >= 2:
            S = np.cov(X, rowvar=False)
            if S.shape == (p, p):
                diag_sqrt = np.sqrt(np.diag(S))
                diag_sqrt[diag_sqrt < 1e-10] = 1e-10
                empirical_corr = S / np.outer(diag_sqrt, diag_sqrt)
                np.fill_diagonal(empirical_corr, 1.0)
                partial_corr = empirical_corr.copy()
        applicable = self.constraint_engine.get_applicable_constraints(variable_names)
        name_to_idx = {name: i for i, name in enumerate(variable_names)}
        for constraint in applicable:
            cause_idx = name_to_idx.get(constraint["cause"])
            effect_idx = name_to_idx.get(constraint["effect"])
            if cause_idx is not None and effect_idx is not None:
                w = constraint["confidence"] * 0.3
                partial_corr[cause_idx, effect_idx] = w
                partial_corr[effect_idx, cause_idx] = w
        np.fill_diagonal(partial_corr, 1.0)
        eigvals = np.linalg.eigvalsh(partial_corr)
        min_eig = np.min(eigvals)
        if min_eig < 1e-6:
            partial_corr += (1e-6 - min_eig) * np.eye(p)
        try:
            precision_matrix = np.linalg.inv(partial_corr)
        except np.linalg.LinAlgError:
            precision_matrix = np.eye(p)
        return precision_matrix, partial_corr

    def _theory_prior_score(self, p: int, variable_names: List[str] = None) -> Tuple[List[int], np.ndarray, np.ndarray]:
        adj = np.zeros((p, p))
        skeleton = np.zeros((p, p))
        if variable_names is None:
            variable_names = [f"V{i}" for i in range(p)]
        applicable = self.constraint_engine.get_applicable_constraints(variable_names)
        name_to_idx = {name: i for i, name in enumerate(variable_names)}
        dag = nx.DiGraph()
        dag.add_nodes_from(range(p))
        for constraint in applicable:
            cause_idx = name_to_idx.get(constraint["cause"])
            effect_idx = name_to_idx.get(constraint["effect"])
            if cause_idx is not None and effect_idx is not None:
                w = constraint["confidence"] * 0.3
                adj[cause_idx, effect_idx] = w
                skeleton[cause_idx, effect_idx] = 1.0
                skeleton[effect_idx, cause_idx] = 1.0
                dag.add_edge(cause_idx, effect_idx)
        try:
            topo_order = list(nx.topological_sort(dag))
        except nx.NetworkXUnfeasible:
            topo_order = list(range(p))
        existing = set(topo_order)
        missing = [i for i in range(p) if i not in existing]
        topo_order.extend(missing)
        return topo_order, adj, skeleton
