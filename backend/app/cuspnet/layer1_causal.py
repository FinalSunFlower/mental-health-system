import os
import numpy as np
from typing import List, Dict, Tuple
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


BORBOOM_CONSTRAINTS = {
    "insomnia": ["fatigue"],
    "fatigue": ["concentration"],
    "anhedonia": ["depressed_mood"],
    "worry": ["restlessness"],
    "stress": ["anxiety"],
    "anxiety": ["insomnia"],
    "social_isolation": ["depressed_mood"],
}


class CausalDiscoveryLayer:
    def __init__(self, ebic_gamma: float = 0.5, score_threshold: float = 0.1):
        self.ebic_gamma = ebic_gamma
        self.score_threshold = score_threshold

    def fit(self, X: np.ndarray, variable_names: List[str] = None) -> Dict:
        if variable_names is None:
            variable_names = [f"V{i}" for i in range(X.shape[1])]
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        precision_matrix, partial_corr = self._ebic_glasso(X_std)
        topological_order, score_adj, score_skeleton = self._score_algorithm(X_std)
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

    def _ebic_glasso(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n, p = X.shape
        S = np.cov(X, rowvar=False)
        if n < 2 or p < 2:
            return np.eye(p), np.zeros((p, p))
        try:
            partial_corr = self._ebic_glasso_rpy2(S, n, p)
            precision_matrix = self._partial_corr_to_precision(partial_corr, S)
        except Exception:
            precision_matrix = self._ebic_glasso_python(X, S, n, p)
            partial_corr = self._precision_to_partial_corr(precision_matrix)
        return precision_matrix, partial_corr

    def _ebic_glasso_rpy2(self, S: np.ndarray, n: int, p: int) -> np.ndarray:
        _ensure_r_env()
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr

        with localconverter(ro.default_converter + numpy2ri.converter):
            qgraph = importr("qgraph")
            R = np.corrcoef(S) if S.shape[0] == S.shape[1] and not np.allclose(
                np.diag(S), 1.0
            ) else S
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

        lambda_max = max(
            np.max(R - np.eye(p)), -np.min(R - np.eye(p))
        )
        lambda_min = 0.01 * lambda_max
        nlambda = 100
        lambdas = np.exp(
            np.linspace(np.log(lambda_min), np.log(lambda_max), nlambda)
        )

        best_ebic = np.inf
        best_precision = None
        for lam in lambdas:
            try:
                gl = GraphicalLasso(alpha=lam, max_iter=500, tol=1e-4)
                gl.fit(X)
                Theta = gl.precision_
                sign, logdet = np.linalg.slogdet(Theta)
                if sign <= 0:
                    continue
                trace_ST = np.trace(R @ Theta)
                E_mask = np.abs(np.triu(Theta, k=1)) > 1e-8
                E_count = int(np.sum(E_mask))
                ebic = (
                    -2 * logdet
                    + 2 * trace_ST
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

    def _score_algorithm(self, X: np.ndarray) -> Tuple[List[int], np.ndarray, np.ndarray]:
        try:
            return self._score_via_causal_learn(X)
        except Exception:
            return self._score_fallback_pc_ges(X)

    def _score_via_causal_learn(self, X: np.ndarray) -> Tuple[List[int], np.ndarray]:
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
        return list(range(p)), np.zeros((p, p)), np.zeros((p, p))

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

    def _score_fallback_pc_ges(self, X: np.ndarray) -> Tuple[List[int], np.ndarray, np.ndarray]:
        p = X.shape[1]
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
                return list(range(p)), np.zeros((p, p)), np.zeros((p, p))
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

        off_diag = np.abs(partial_corr)[np.triu_indices(p, k=1)]
        off_diag_nonzero = off_diag[off_diag > 1e-10]
        if len(off_diag_nonzero) > 0:
            ebic_strong_threshold = float(np.percentile(off_diag_nonzero, 80))
            ebic_strong_threshold = max(ebic_strong_threshold, self.score_threshold)
        else:
            ebic_strong_threshold = self.score_threshold

        combined_skeleton = np.zeros((p, p))
        if score_skeleton is not None:
            combined_skeleton = score_skeleton.copy()

        for i in range(p):
            for j in range(i + 1, p):
                if combined_skeleton[i, j] == 0:
                    if abs(partial_corr[i, j]) > ebic_strong_threshold:
                        combined_skeleton[i, j] = 1.0
                        combined_skeleton[j, i] = 1.0

        dag = nx.DiGraph()
        dag.add_nodes_from(range(p))

        if score_adj is not None:
            for i in range(p):
                for j in range(p):
                    if i == j:
                        continue
                    if score_adj[i, j] != 0 and combined_skeleton[i, j] > 0:
                        if abs(partial_corr[i, j]) > self.score_threshold:
                            A[i, j] = partial_corr[i, j]
                            dag.add_edge(i, j)

        for i in range(p):
            for j in range(i + 1, p):
                if combined_skeleton[i, j] > 0:
                    if A[i, j] == 0 and A[j, i] == 0:
                        if abs(partial_corr[i, j]) > self.score_threshold:
                            if not nx.has_path(dag, j, i):
                                A[i, j] = partial_corr[i, j]
                                dag.add_edge(i, j)
                            elif not nx.has_path(dag, i, j):
                                A[j, i] = partial_corr[i, j]
                                dag.add_edge(j, i)

        name_to_idx = {name: idx for idx, name in enumerate(names)}
        nonzero_edges = A[A != 0]
        if len(nonzero_edges) > 0:
            fallback_weight = float(np.median(np.abs(nonzero_edges)))
        else:
            fallback_weight = 0.1
        for cause_name, effect_names in BORBOOM_CONSTRAINTS.items():
            if cause_name not in name_to_idx:
                continue
            cause_idx = name_to_idx[cause_name]
            for effect_name in effect_names:
                if effect_name not in name_to_idx:
                    continue
                effect_idx = name_to_idx[effect_name]
                if abs(partial_corr[cause_idx, effect_idx]) > 1e-10:
                    A[cause_idx, effect_idx] = partial_corr[cause_idx, effect_idx]
                else:
                    A[cause_idx, effect_idx] = fallback_weight
                A[effect_idx, cause_idx] = 0.0
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
