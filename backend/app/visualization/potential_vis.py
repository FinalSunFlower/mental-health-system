import numpy as np
from typing import Dict, Optional
from app.cuspnet.utils import compute_potential, find_fixed_points, classify_fixed_points


def plot_potential_surface(
    a: float,
    b: float,
    c: float,
    x_range: Optional[np.ndarray] = None,
    current_state: Optional[Dict] = None,
) -> Dict:
    if x_range is None:
        x_range = np.linspace(-2, 2, 200)
    V = compute_potential(x_range, a, b, c)
    roots = find_fixed_points(a, b, c)
    classified = classify_fixed_points(roots, b, c)
    fixed_points = [
        {
            "x": p["value"],
            "V": float(compute_potential(np.array([p["value"]]), a, b, c)[0]),
            "stability": p["stability"],
        }
        for p in classified
    ]

    result = {"x": x_range.tolist(), "V": V.tolist(), "fixed_points": fixed_points}

    if current_state is not None:
        result["current_state"] = current_state
        if "a" in current_state and "b" in current_state and "c" in current_state:
            cs_a = current_state["a"]
            cs_b = current_state["b"]
            cs_c = current_state["c"]
            cs_x_range = np.linspace(-2, 2, 200)
            cs_V = compute_potential(cs_x_range, cs_a, cs_b, cs_c)
            cs_roots = find_fixed_points(cs_a, cs_b, cs_c)
            cs_classified = classify_fixed_points(cs_roots, cs_b, cs_c)
            result["current_state_potential"] = {
                "x": cs_x_range.tolist(),
                "V": cs_V.tolist(),
                "fixed_points": [
                    {
                        "x": p["value"],
                        "V": float(compute_potential(np.array([p["value"]]), cs_a, cs_b, cs_c)[0]),
                        "stability": p["stability"],
                    }
                    for p in cs_classified
                ],
            }

    return result
