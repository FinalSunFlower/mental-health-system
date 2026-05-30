import numpy as np
from typing import Tuple


def sigmoid(x: np.ndarray, k: float = 10.0, tau: float = 0.5) -> np.ndarray:
    z = k * (x - tau)
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def generate_cusp_synthetic(
    n_samples: int = 500,
    n_variables: int = 9,
    a_range: Tuple[float, float] = (-0.5, 0.5),
    b_range: Tuple[float, float] = (0.1, 1.0),
    c_range: Tuple[float, float] = (0.5, 1.5),
    noise_std: float = 0.1,
    tau_threshold: float = 0.5,
    coupling_k: float = 10.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, list]:
    rng = np.random.RandomState(seed)
    A = np.zeros((n_variables, n_variables))
    for i in range(1, n_variables):
        for j in range(i):
            if rng.rand() < 0.3:
                A[i, j] = rng.uniform(0.1, 0.5)
    var_names = [f"V{i}" for i in range(n_variables)]
    a_params = rng.uniform(*a_range, size=n_variables)
    b_params = rng.uniform(*b_range, size=n_variables)
    c_params = rng.uniform(*c_range, size=n_variables)
    X = np.zeros((n_samples, n_variables))
    for sample in range(n_samples):
        x = rng.randn(n_variables) * 0.1
        for _ in range(100):
            coupling = A @ sigmoid(x, k=coupling_k, tau=tau_threshold)
            dx = a_params + b_params * x - c_params * x**3 + coupling
            x = x + 0.01 * dx
        X[sample] = x + rng.randn(n_variables) * noise_std
    return X, A, var_names
