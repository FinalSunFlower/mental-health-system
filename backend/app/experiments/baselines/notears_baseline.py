import numpy as np
from typing import Optional


def run_notears(X: np.ndarray, lambda1: float = 0.01, max_iter: int = 100) -> np.ndarray:
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    try:
        from notears.linear import notears_linear
        W = notears_linear(X_std, lambda1=lambda1, max_iter=max_iter, loss_type='l2')
        adj = (np.abs(W) > 0.1).astype(np.float64)
        np.fill_diagonal(adj, 0)
        return adj
    except (ImportError, AttributeError):
        pass
    return _notears_fallback(X_std, lambda1, max_iter)


def _notears_fallback(X: np.ndarray, lambda1: float = 0.1, max_iter: int = 100) -> np.ndarray:
    from scipy.optimize import minimize

    n, d = X.shape

    def _loss(W_flat):
        W = W_flat.reshape(d, d)
        M = np.eye(d) - W
        XM = X @ M.T
        return 0.5 / n * np.sum(XM ** 2)

    def _h(W_flat):
        W = W_flat.reshape(d, d)
        E = np.linalg.matrix_power(np.abs(W) + np.eye(d), d) - np.eye(d)
        return np.trace(E)

    def _grad_loss(W_flat):
        W = W_flat.reshape(d, d)
        M = np.eye(d) - W
        XM = X @ M.T
        G_loss = -1.0 / n * XM.T @ X
        return G_loss.flatten()

    def _grad_h(W_flat):
        W = W_flat.reshape(d, d)
        E = np.linalg.matrix_power(np.abs(W) + np.eye(d), d) - np.eye(d)
        G_h = E.T * np.sign(W + 1e-10)
        return G_h.flatten()

    def _objective(W_flat, rho, alpha_dual):
        loss = _loss(W_flat)
        h_val = _h(W_flat)
        return loss + rho / 2 * h_val ** 2 + alpha_dual * h_val

    def _grad_objective(W_flat, rho, alpha_dual):
        return _grad_loss(W_flat) + (rho * _h(W_flat) + alpha_dual) * _grad_h(W_flat)

    W_flat = np.zeros(d * d)
    rho = 1.0
    alpha_dual = 0.0
    h_tol = 1e-8
    for iteration in range(max_iter):
        result = minimize(
            _objective, W_flat, args=(rho, alpha_dual),
            method="L-BFGS-B", jac=_grad_objective,
            options={"maxiter": 100},
        )
        W_flat = result.x
        h_val = _h(W_flat)
        if h_val <= h_tol:
            break
        rho *= 10.0
        alpha_dual += rho * h_val

    W = W_flat.reshape(d, d)
    np.fill_diagonal(W, 0)
    W[np.abs(W) < lambda1] = 0
    adj = (np.abs(W) > 0).astype(np.float64)
    return adj
