"""Scalability and objective-quality audit for DAG-order projection."""
from __future__ import annotations

from time import perf_counter
from typing import Dict, Sequence

import numpy as np

from app.cuspnet.projection import CausalDiscoveryLayer


def _orientation_accuracy(order, true_order, skeleton) -> float:
    rank = {node: position for position, node in enumerate(order)}
    truth = {node: position for position, node in enumerate(true_order)}
    correct = total = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if skeleton[i, j] == 0:
                continue
            total += 1
            correct += int((rank[i] < rank[j]) == (truth[i] < truth[j]))
    return correct / total if total else 1.0


def run_exp10(
    variable_counts: Sequence[int] = (10, 14, 25, 50, 100),
    n_repeats: int = 12,
    edge_probability: float = 0.15,
    seed: int = 20260816,
) -> Dict:
    rows = []
    for n_variables in variable_counts:
        for repeat in range(n_repeats):
            rng = np.random.RandomState(seed + n_variables * 1009 + repeat)
            true_order = list(rng.permutation(n_variables))
            true_rank = {node: position for position, node in enumerate(true_order)}
            skeleton = np.zeros((n_variables, n_variables), dtype=float)
            scores = np.zeros_like(skeleton)
            for i in range(n_variables):
                for j in range(i + 1, n_variables):
                    if rng.rand() >= edge_probability:
                        continue
                    skeleton[i, j] = skeleton[j, i] = 1.0
                    if true_rank[i] < true_rank[j]:
                        scores[i, j] = max(0.0, rng.normal(1.0, 0.45))
                        scores[j, i] = max(0.0, rng.normal(0.25, 0.45))
                    else:
                        scores[j, i] = max(0.0, rng.normal(1.0, 0.45))
                        scores[i, j] = max(0.0, rng.normal(0.25, 0.45))

            started = perf_counter()
            sorted_order = CausalDiscoveryLayer._score_sort_order(skeleton, scores)
            sort_runtime = perf_counter() - started
            sort_objective = CausalDiscoveryLayer._order_objective(
                sorted_order, skeleton, scores
            )
            independent_pair_upper_bound = (
                CausalDiscoveryLayer._order_objective_upper_bound(skeleton, scores)
            )

            started = perf_counter()
            refined_order, refined_objective, passes = (
                CausalDiscoveryLayer._refined_score_sort_order(skeleton, scores)
            )
            refined_runtime = perf_counter() - started
            row = {
                "n_variables": n_variables,
                "repeat": repeat,
                "candidate_edges": int(np.sum(np.triu(skeleton, 1))),
                "score_sort": {
                    "objective": sort_objective,
                    "orientation_accuracy": _orientation_accuracy(
                        sorted_order, true_order, skeleton
                    ),
                    "runtime_seconds": sort_runtime,
                },
                "refined_order": {
                    "objective": refined_objective,
                    "orientation_accuracy": _orientation_accuracy(
                        refined_order, true_order, skeleton
                    ),
                    "runtime_seconds": refined_runtime,
                    "relocation_passes": passes,
                    "independent_pair_upper_bound": independent_pair_upper_bound,
                    "instancewise_approximation_lower_bound": (
                        1.0
                        if independent_pair_upper_bound <= 1e-12
                        else float(refined_objective / independent_pair_upper_bound)
                    ),
                    "objective_gap_upper_bound": float(
                        max(0.0, independent_pair_upper_bound - refined_objective)
                    ),
                },
            }
            if n_variables <= 14:
                started = perf_counter()
                exact_order, exact_objective = (
                    CausalDiscoveryLayer._exact_maximum_evidence_order(skeleton, scores)
                )
                row["exact_order"] = {
                    "objective": exact_objective,
                    "orientation_accuracy": _orientation_accuracy(
                        exact_order, true_order, skeleton
                    ),
                    "runtime_seconds": perf_counter() - started,
                    "refined_relative_gap": (
                        (exact_objective - refined_objective) / max(abs(exact_objective), 1e-12)
                    ),
                    "independent_pair_bound_relative_slack": (
                        (independent_pair_upper_bound - exact_objective)
                        / max(abs(exact_objective), 1e-12)
                    ),
                }
            rows.append(row)

    settings = []
    for n_variables in variable_counts:
        group = [row for row in rows if row["n_variables"] == n_variables]
        settings.append({
            "n_variables": n_variables,
            "n_repeats": len(group),
            "mean_candidate_edges": float(np.mean([row["candidate_edges"] for row in group])),
            "score_sort_mean_accuracy": float(np.mean([
                row["score_sort"]["orientation_accuracy"] for row in group
            ])),
            "refined_mean_accuracy": float(np.mean([
                row["refined_order"]["orientation_accuracy"] for row in group
            ])),
            "mean_objective_gain": float(np.mean([
                row["refined_order"]["objective"] - row["score_sort"]["objective"]
                for row in group
            ])),
            "refined_mean_runtime_seconds": float(np.mean([
                row["refined_order"]["runtime_seconds"] for row in group
            ])),
            "refined_max_runtime_seconds": float(np.max([
                row["refined_order"]["runtime_seconds"] for row in group
            ])),
            "mean_exact_relative_gap": (
                float(np.mean([row["exact_order"]["refined_relative_gap"] for row in group]))
                if "exact_order" in group[0] else None
            ),
            "mean_instancewise_approximation_lower_bound": float(np.mean([
                row["refined_order"]["instancewise_approximation_lower_bound"]
                for row in group
            ])),
            "mean_objective_gap_upper_bound": float(np.mean([
                row["refined_order"]["objective_gap_upper_bound"]
                for row in group
            ])),
        })
    return {
        "protocol": {
            "variable_counts": list(variable_counts),
            "n_repeats": n_repeats,
            "edge_probability": edge_probability,
            "resource_policy": "CPU only; exact DP is restricted to p <= 14 in this audit",
        },
        "aggregate": {
            "refined_never_below_score_sort": bool(all(
                row["refined_order"]["objective"] + 1e-10
                >= row["score_sort"]["objective"] for row in rows
            )),
            "largest_graph": int(max(variable_counts)),
            "largest_graph_max_runtime_seconds": float(max(
                row["refined_order"]["runtime_seconds"]
                for row in rows if row["n_variables"] == max(variable_counts)
            )),
        },
        "settings": settings,
        "replicates": rows,
    }
