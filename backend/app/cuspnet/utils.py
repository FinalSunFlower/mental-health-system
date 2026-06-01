"""
Utility functions for CUSP catastrophe model computations.
Includes sigmoid activation, cubic root solving, potential function evaluation,
fixed point analysis, resilience reserve calculation, and feedback loop detection.
"""
import numpy as np
from typing import List, Dict


def sigmoid(x: np.ndarray, tau: float = 0.5, k: float = 10.0) -> np.ndarray:
    z = np.clip(-k * (x - tau), -500, 500)
    return 1.0 / (1.0 + np.exp(z))


def solve_cubic(a: float, b: float, c: float) -> List[float]:
    roots = np.roots([c, 0, -b, -a])
    real_roots = sorted([r.real for r in roots if abs(r.imag) < 1e-10])
    deduped = []
    for r in real_roots:
        if not deduped or abs(r - deduped[-1]) > 1e-8:
            deduped.append(r)
    return deduped


def compute_potential(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return -a * x - (b / 2) * x**2 + (c / 4) * x**4


def find_fixed_points(a: float, b: float, c: float) -> List[float]:
    return solve_cubic(a, b, c)


def classify_fixed_points(roots: List[float], b: float, c: float) -> List[Dict]:
    results = []
    for x in roots:
        second_deriv = -b + 3 * c * x**2
        stability = "stable" if second_deriv > 0 else "unstable"
        results.append({"value": x, "stability": stability, "second_deriv": second_deriv})
    return results


def compute_resilience_reserve(a: float, b: float, c: float) -> float:
    roots = find_fixed_points(a, b, c)
    classified = classify_fixed_points(roots, b, c)
    stable = [p for p in classified if p["stability"] == "stable"]
    unstable = [p for p in classified if p["stability"] == "unstable"]
    if len(stable) < 2 or len(unstable) < 1:
        return float("inf")
    healthy = min(stable, key=lambda p: abs(p["value"]))
    saddle = unstable[0]
    v_healthy = compute_potential(np.array([healthy["value"]]), a, b, c)[0]
    v_saddle = compute_potential(np.array([saddle["value"]]), a, b, c)[0]
    return v_saddle - v_healthy


def compute_critical_distance(b: float, b_crit: float = 0.0) -> float:
    if b_crit == 0:
        return abs(b)
    return abs(b - b_crit) / abs(b_crit)


def detect_positive_feedback_loops(A: np.ndarray, max_depth: int = 5) -> List[List[int]]:
    p = A.shape[0]
    loops = []
    def dfs(node, path, visited):
        if len(path) > max_depth:
            return
        for neighbor in range(p):
            if abs(A[node, neighbor]) > 1e-10:
                if neighbor == path[0] and len(path) >= 2:
                    loops.append(path[:])
                elif neighbor not in visited:
                    visited.add(neighbor)
                    dfs(neighbor, path + [neighbor], visited)
                    visited.discard(neighbor)
    for start in range(p):
        dfs(start, [start], {start})
    return loops


def compute_loop_strength(A: np.ndarray, loop: List[int]) -> float:
    if len(loop) < 2:
        return 0.0
    product = 1.0
    for i in range(len(loop)):
        j = (i + 1) % len(loop)
        product *= abs(A[loop[i], loop[j]])
    return product ** (1.0 / len(loop))
