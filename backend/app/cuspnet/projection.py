"""Support-preserving causal orientation for RCEP."""
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
import networkx as nx
from .contracts import RiskControlledEvidenceGate


class TheoryConstraint:
    def __init__(
        self,
        cause: str,
        effect: str,
        theory: str = "external_source",
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

    def __init__(
        self,
        theories: List[str] = None,
        config_path: str = None,
        min_confidence: float = 0.5,
        constraints: Optional[List[TheoryConstraint]] = None,
        risk_gate: Optional[RiskControlledEvidenceGate] = None,
        require_risk_certificate: bool = False,
    ):
        self.constraints: List[TheoryConstraint] = []
        self.min_confidence = min_confidence
        self.risk_gate = risk_gate
        self.require_risk_certificate = bool(require_risk_certificate)
        self._name_aliases: Dict[str, List[str]] = {}
        selected_theories = [] if theories is None else theories
        self._load_builtin(selected_theories)
        if constraints:
            self.constraints.extend([
                TheoryConstraint(
                    constraint.cause,
                    constraint.effect,
                    constraint.theory,
                    constraint.confidence,
                    constraint.citation,
                )
                for constraint in constraints
            ])
        if config_path:
            self._load_from_config(config_path)
        self._build_name_aliases()

    def _load_builtin(self, theories: List[str]):
        if theories:
            raise ValueError(
                "Built-in domain constraints were removed; pass explicit, calibrated constraints."
            )

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
        self._name_aliases = {}

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
            if self.require_risk_certificate and self.risk_gate is None:
                continue
            if c.confidence < self.min_confidence:
                continue
            if self.risk_gate is not None and not self.risk_gate.accepts(c.confidence):
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

    def orientation_prior(self, names: List[str]) -> np.ndarray:
        """Return a soft directional-prior matrix for the observed variables.

        A prior expresses a proposed orientation only. It is intentionally kept
        separate from adjacency construction so external knowledge cannot create
        an edge that has no statistical or score-based candidate support.
        """
        prior = np.zeros((len(names), len(names)), dtype=float)
        name_to_idx = {name: idx for idx, name in enumerate(names)}
        for constraint in self.get_applicable_constraints(names):
            cause_idx = name_to_idx[constraint["cause"]]
            effect_idx = name_to_idx[constraint["effect"]]
            prior[cause_idx, effect_idx] = max(
                prior[cause_idx, effect_idx],
                constraint["confidence"],
            )
        return prior


class CausalDiscoveryLayer:
    def __init__(
        self,
        ebic_gamma: float = 0.5,
        score_threshold: float = 0.1,
        constraint_engine: TheoryConstraintEngine = None,
        orientation_weights: Optional[Dict[str, float]] = None,
        fixed_glasso_alpha: Optional[float] = None,
        orientation_solver: str = "exact_order",
        max_exact_variables: int = 16,
    ):
        self.ebic_gamma = ebic_gamma
        self.score_threshold = score_threshold
        self.constraint_engine = constraint_engine or TheoryConstraintEngine()
        self.fixed_glasso_alpha = fixed_glasso_alpha
        if orientation_solver not in {"exact_order", "refined_order", "score_sort"}:
            raise ValueError(
                "orientation_solver must be 'exact_order', 'refined_order', or 'score_sort'"
            )
        self.orientation_solver = orientation_solver
        self.max_exact_variables = int(max_exact_variables)
        self._last_orientation_audit: Dict[str, object] = {}
        default_weights = {
            "partial_correlation": 1.0,
            "score_direction": 0.5,
            "theory_prior": 0.35,
            "topological_compatibility": 0.15,
        }
        self.orientation_weights = {
            **default_weights,
            **(orientation_weights or {}),
        }

    def fit(self, X: np.ndarray, variable_names: List[str] = None) -> Dict:
        if variable_names is None:
            variable_names = [f"V{i}" for i in range(X.shape[1])]
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        precision_matrix, partial_corr = self._ebic_glasso(X_std, variable_names)
        topological_order, score_adj, score_skeleton = self._score_algorithm(X_std, variable_names)
        causal_adjacency = self._theory_constrained_orientation(
            partial_corr, topological_order, variable_names, score_adj, score_skeleton
        )
        constraint_report = self._constraint_report(variable_names)
        return {
            "precision_matrix": precision_matrix,
            "partial_correlation": partial_corr,
            "causal_adjacency": causal_adjacency,
            "topological_order": topological_order,
            "constraint_report": constraint_report,
            "orientation_audit": self._last_orientation_audit,
            "variable_names": variable_names,
        }

    def _constraint_report(self, variable_names: List[str]) -> Dict:
        applicable = self.constraint_engine.get_applicable_constraints(variable_names)
        unique_edges = {
            (item["cause"], item["effect"])
            for item in applicable
        }
        theory_counts: Dict[str, int] = {}
        for item in applicable:
            theory_counts[item["theory"]] = theory_counts.get(item["theory"], 0) + 1

        total_possible = max(len(variable_names) * max(len(variable_names) - 1, 1), 1)
        coverage = len(unique_edges) / total_possible
        avg_confidence = (
            float(np.mean([item["confidence"] for item in applicable]))
            if applicable else 0.0
        )
        return {
            "available_constraints": len(self.constraint_engine.constraints),
            "available_theories": sorted(
                {constraint.theory for constraint in self.constraint_engine.constraints}
            ),
            "applicable_constraints": len(applicable),
            "unique_applicable_edges": len(unique_edges),
            "coverage": float(coverage),
            "average_confidence": avg_confidence,
            "theory_counts": theory_counts,
            "risk_control": (
                self.constraint_engine.risk_gate.certificate_
                if self.constraint_engine.risk_gate is not None
                else {
                    "status": "abstain_missing_calibration"
                    if self.constraint_engine.require_risk_certificate
                    else "not_required"
                }
            ),
        }

    def _ebic_glasso(self, X: np.ndarray, variable_names: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        n, p = X.shape
        if n < 2 or p < 2:
            return np.eye(p), np.zeros((p, p), dtype=float)
        S = np.cov(X, rowvar=False)
        try:
            if self.fixed_glasso_alpha is not None:
                precision_matrix = self._fit_graphical_lasso(
                    X,
                    alpha=self.fixed_glasso_alpha,
                    max_iter=1000,
                    tol=1e-4,
                ).precision_
            else:
                precision_matrix = self._ebic_glasso_python(X, S, n, p)
            partial_corr = self._precision_to_partial_corr(precision_matrix)
        except Exception:
            covariance = np.asarray(S, dtype=float)
            covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
            covariance += 1e-6 * np.eye(p)
            precision_matrix = np.linalg.pinv(covariance)
            partial_corr = self._precision_to_partial_corr(precision_matrix)
        return precision_matrix, partial_corr

    def _ebic_glasso_python(
        self, X: np.ndarray, S: np.ndarray, n: int, p: int
    ) -> np.ndarray:
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
        for s_inds in subsample_indices:
            X_sub = X[s_inds]
            for lam_idx, lam in enumerate(lambdas):
                try:
                    gl = self._fit_graphical_lasso(X_sub, lam, max_iter=1000, tol=1e-4)
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
                gl = self._fit_graphical_lasso(X, lam, max_iter=3000, tol=1e-5)
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

    def _fit_graphical_lasso(self, X: np.ndarray, alpha: float, max_iter: int, tol: float):
        from sklearn.covariance import GraphicalLasso
        from sklearn.exceptions import ConvergenceWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gl = GraphicalLasso(alpha=alpha, max_iter=max_iter, tol=tol)
            gl.fit(X)
        return gl

    def _precision_to_partial_corr(self, precision: np.ndarray) -> np.ndarray:
        diag_sqrt = np.sqrt(np.diag(precision))
        denom = np.outer(diag_sqrt, diag_sqrt)
        denom = np.where(denom < 1e-10, 1e-10, denom)
        partial_corr = -precision / denom
        np.fill_diagonal(partial_corr, 0.0)
        return partial_corr

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
                return list(range(p)), adj, skeleton
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
        combined_skeleton = np.zeros((p, p))
        if score_skeleton is not None:
            combined_skeleton = score_skeleton.copy()

        for i in range(p):
            for j in range(i + 1, p):
                if combined_skeleton[i, j] == 0:
                    if abs(partial_corr[i, j]) > self.score_threshold:
                        combined_skeleton[i, j] = 1.0
                        combined_skeleton[j, i] = 1.0

        theory_prior = self.constraint_engine.orientation_prior(names)
        fused_scores = self._orientation_score_matrix(
            partial_corr, topo_order, score_adj, theory_prior
        )
        data_scores = self._orientation_score_matrix(
            partial_corr, topo_order, score_adj, np.zeros_like(theory_prior)
        )
        fused, fused_meta = self._project_acyclic_orientation(
            combined_skeleton, fused_scores, partial_corr
        )
        data_only, data_meta = self._project_acyclic_orientation(
            combined_skeleton, data_scores, partial_corr
        )

        changed_pairs = []
        for i in range(p):
            for j in range(i + 1, p):
                if combined_skeleton[i, j] and (
                    (fused[i, j] != 0) != (data_only[i, j] != 0)
                ):
                    changed_pairs.append([names[i], names[j]])

        output = self._prune_indirect_effects(fused, partial_corr, X_std=None)
        output_support = (np.abs(output) > 1e-10).astype(int)
        support_violations = int(np.sum(output_support * (combined_skeleton == 0)))
        graph = nx.DiGraph(output_support)
        prior_on_support = int(np.sum((theory_prior > 0) * (combined_skeleton > 0)))
        prior_off_support = int(np.sum((theory_prior > 0) * (combined_skeleton == 0)))
        self._last_orientation_audit = {
            "contract": "support_preserving_global_acyclic_projection",
            "solver": fused_meta["solver"],
            "candidate_edges": int(np.sum(np.triu(combined_skeleton, 1))),
            "output_edges": int(np.sum(output_support)),
            "support_violations": support_violations,
            "is_acyclic": bool(nx.is_directed_acyclic_graph(graph)),
            "prior_statements_on_support": prior_on_support,
            "prior_statements_rejected_off_support": prior_off_support,
            "prior_induced_orientation_changes": len(changed_pairs),
            "changed_pairs": changed_pairs,
            "fused_objective": fused_meta["objective"],
            "fused_objective_upper_bound": fused_meta["objective_upper_bound"],
            "fused_instancewise_approximation_lower_bound": fused_meta[
                "instancewise_approximation_lower_bound"
            ],
            "data_only_objective_under_data_scores": data_meta["objective"],
            "data_only_objective_upper_bound": data_meta["objective_upper_bound"],
            "fused_order": [names[index] for index in fused_meta["order"]],
            "data_only_order": [names[index] for index in data_meta["order"]],
        }
        return output

    def _orientation_score_matrix(
        self,
        partial_corr: np.ndarray,
        topo_order: List[int],
        score_adj: Optional[np.ndarray],
        theory_prior: np.ndarray,
    ) -> np.ndarray:
        """Combine directional evidence without changing candidate support."""
        p = partial_corr.shape[0]
        topo_rank = {node: rank for rank, node in enumerate(topo_order)}
        weights = self.orientation_weights
        scores = np.zeros((p, p), dtype=float)
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                score_direction = 0.0 if score_adj is None else float(score_adj[i, j] != 0)
                topo_compatibility = float(topo_rank.get(i, p) < topo_rank.get(j, p))
                scores[i, j] = (
                    weights["partial_correlation"] * abs(partial_corr[i, j])
                    + weights["score_direction"] * score_direction
                    + weights["theory_prior"] * theory_prior[i, j]
                    + weights["topological_compatibility"] * topo_compatibility
                )
        return scores

    def _project_acyclic_orientation(
        self,
        skeleton: np.ndarray,
        scores: np.ndarray,
        partial_corr: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """Find the maximum-evidence orientation induced by a total order."""
        p = skeleton.shape[0]
        if self.orientation_solver == "exact_order" and p <= self.max_exact_variables:
            order, objective = self._exact_maximum_evidence_order(skeleton, scores)
            solver = "exact_subset_dynamic_programming"
        elif self.orientation_solver in {"exact_order", "refined_order"}:
            order, objective, passes = self._refined_score_sort_order(skeleton, scores)
            solver = "deterministic_relocation_search"
        else:
            order = self._score_sort_order(skeleton, scores)
            objective = self._order_objective(order, skeleton, scores)
            solver = "deterministic_score_sort"

        rank = {node: position for position, node in enumerate(order)}
        adjacency = np.zeros_like(partial_corr, dtype=float)
        for i in range(p):
            for j in range(i + 1, p):
                if skeleton[i, j] == 0 or abs(partial_corr[i, j]) < 1e-10:
                    continue
                if rank[i] < rank[j]:
                    adjacency[i, j] = partial_corr[i, j]
                else:
                    adjacency[j, i] = partial_corr[i, j]
        metadata = {
            "solver": solver,
            "order": order,
            "objective": float(objective),
        }
        # For nonnegative directional scores, the sum of the larger direction
        # on each candidate pair is a valid instance-wise upper bound on every
        # total-order objective.  Reporting it makes fallback quality
        # auditable without claiming a uniform approximation theorem.
        upper_bound = self._order_objective_upper_bound(skeleton, scores)
        metadata["objective_upper_bound"] = float(upper_bound)
        metadata["instancewise_approximation_lower_bound"] = float(
            1.0 if upper_bound <= 1e-12 else max(0.0, min(1.0, objective / upper_bound))
        )
        metadata["objective_gap_upper_bound"] = float(max(0.0, upper_bound - objective))
        if solver == "deterministic_relocation_search":
            metadata["refinement_passes"] = passes
        return adjacency, metadata

    @staticmethod
    def _order_objective(order: List[int], skeleton: np.ndarray, scores: np.ndarray) -> float:
        rank = {node: position for position, node in enumerate(order)}
        objective = 0.0
        p = len(order)
        for i in range(p):
            for j in range(i + 1, p):
                if skeleton[i, j] == 0:
                    continue
                objective += scores[i, j] if rank[i] < rank[j] else scores[j, i]
        return float(objective)

    @staticmethod
    def _order_objective_upper_bound(skeleton: np.ndarray, scores: np.ndarray) -> float:
        """Upper bound obtained by independently choosing each edge direction."""
        p = skeleton.shape[0]
        upper = 0.0
        for i in range(p):
            for j in range(i + 1, p):
                if skeleton[i, j] != 0:
                    upper += max(float(scores[i, j]), float(scores[j, i]))
        return float(max(0.0, upper))

    @staticmethod
    def _score_sort_order(skeleton: np.ndarray, scores: np.ndarray) -> List[int]:
        preference = np.sum((scores - scores.T) * (skeleton > 0), axis=1)
        return sorted(range(len(preference)), key=lambda node: (-preference[node], node))

    @classmethod
    def _refined_score_sort_order(
        cls,
        skeleton: np.ndarray,
        scores: np.ndarray,
    ) -> Tuple[List[int], float, int]:
        """Deterministically improve score-sort by best node relocations.

        Each pass evaluates every single-node relocation in O(p^2) time using
        only the pair preferences crossed by that move. The accepted objective
        is monotone, so this fallback can never be worse than score-sort.
        """
        order = cls._score_sort_order(skeleton, scores)
        objective = cls._order_objective(order, skeleton, scores)
        tolerance = 1e-12
        passes = 0
        max_passes = max(1, len(order))

        while passes < max_passes:
            best_delta = 0.0
            best_move = None
            for old_position, node in enumerate(order):
                delta = 0.0
                for new_position in range(old_position + 1, len(order)):
                    crossed = order[new_position]
                    if skeleton[node, crossed] != 0:
                        delta += scores[crossed, node] - scores[node, crossed]
                    move = (old_position, new_position)
                    if delta > best_delta + tolerance or (
                        abs(delta - best_delta) <= tolerance
                        and delta > tolerance
                        and (best_move is None or move < best_move)
                    ):
                        best_delta = float(delta)
                        best_move = move

                delta = 0.0
                for new_position in range(old_position - 1, -1, -1):
                    crossed = order[new_position]
                    if skeleton[node, crossed] != 0:
                        delta += scores[node, crossed] - scores[crossed, node]
                    move = (old_position, new_position)
                    if delta > best_delta + tolerance or (
                        abs(delta - best_delta) <= tolerance
                        and delta > tolerance
                        and (best_move is None or move < best_move)
                    ):
                        best_delta = float(delta)
                        best_move = move

            if best_move is None:
                break
            old_position, new_position = best_move
            node = order.pop(old_position)
            order.insert(new_position, node)
            objective += best_delta
            passes += 1

        verified = cls._order_objective(order, skeleton, scores)
        if abs(verified - objective) > 1e-8:
            raise RuntimeError("relocation objective accounting mismatch")
        return order, float(verified), passes

    @staticmethod
    def _exact_maximum_evidence_order(
        skeleton: np.ndarray,
        scores: np.ndarray,
    ) -> Tuple[List[int], float]:
        """Exact O(p 2^p) maximum-weight order for small candidate graphs."""
        p = skeleton.shape[0]
        n_states = 1 << p
        best = np.full(n_states, -np.inf, dtype=float)
        last = np.full(n_states, -1, dtype=np.int16)
        best[0] = 0.0
        tolerance = 1e-12
        for mask in range(1, n_states):
            remaining = mask
            while remaining:
                bit = remaining & -remaining
                node = bit.bit_length() - 1
                previous = mask ^ bit
                gain = 0.0
                parents = previous
                while parents:
                    parent_bit = parents & -parents
                    parent = parent_bit.bit_length() - 1
                    if skeleton[parent, node] != 0:
                        gain += scores[parent, node]
                    parents ^= parent_bit
                value = best[previous] + gain
                if value > best[mask] + tolerance or (
                    abs(value - best[mask]) <= tolerance
                    and (last[mask] < 0 or node < last[mask])
                ):
                    best[mask] = value
                    last[mask] = node
                remaining ^= bit

        reverse_order = []
        mask = n_states - 1
        while mask:
            node = int(last[mask])
            if node < 0:
                raise RuntimeError("exact orientation projection failed to reconstruct an order")
            reverse_order.append(node)
            mask ^= 1 << node
        order = list(reversed(reverse_order))
        return order, float(best[-1])

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
