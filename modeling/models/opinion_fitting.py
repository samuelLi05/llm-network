"""Homophily model fitting built on baseline adjacency utilities."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

from baseline_utils import build_dataset_from_run, build_row_normalized_adjacency
from plot_utils import (
    compute_mean_per_timestep,
    compute_variance_per_timestep,
    compute_wasserstein_distance_per_timestep,
)

Array = np.ndarray


def sanitize_array(values: Array) -> Array:
    return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def _pooled_blocks(run_traj_map: Dict[str, Array]) -> Tuple[Array, Array]:
    run_names = sorted(run_traj_map.keys())
    x_blocks, y_blocks = [], []

    for run_name in run_names:
        traj = np.asarray(run_traj_map[run_name], dtype=float)
        x, y = build_dataset_from_run(traj)
        x_blocks.append(x)
        y_blocks.append(y)

    return np.vstack(x_blocks), np.vstack(y_blocks)


def fit_homophily(
    run_traj_map: Dict[str, Array],
    run_neighbors: Dict[str, Dict[int, List[int]]],
    gamma0: float = 1.0,
) -> Dict[str, object]:
    """Fit a single nonnegative gamma for the homophily kernel."""
    x_pool, y_pool = _pooled_blocks(run_traj_map)

    run_names = sorted(run_traj_map.keys())
    ref_neighbors = run_neighbors[run_names[0]]
    n = x_pool.shape[1]
    abar = build_row_normalized_adjacency(ref_neighbors, n)

    def objective(gamma_array: Array) -> float:
        gamma = float(gamma_array[0])
        pred_rows = []
        for t in range(x_pool.shape[0]):
            x_t = sanitize_array(x_pool[t]).ravel()
            diff = np.abs(x_t[:, None] - x_t[None, :])
            raw = abar * np.exp(-gamma * diff)
            row_sums = raw.sum(axis=1, keepdims=True)
            w_t = np.zeros_like(raw, dtype=float)
            valid = row_sums[:, 0] > 0
            w_t[valid] = raw[valid] / row_sums[valid]
            pred_rows.append(w_t @ x_t)
        pred_pool = np.asarray(pred_rows, dtype=float)
        return float(np.mean((y_pool - pred_pool) ** 2))

    result = minimize(
        objective,
        x0=np.asarray([max(float(gamma0), 0.0)], dtype=float),
        method="L-BFGS-B",
        bounds=[(0.0, None)],
    )

    gamma_hat = float(result.x[0])
    fitted_rows = []
    for t in range(x_pool.shape[0]):
        x_t = sanitize_array(x_pool[t]).ravel()
        diff = np.abs(x_t[:, None] - x_t[None, :])
        raw = abar * np.exp(-gamma_hat * diff)
        row_sums = raw.sum(axis=1, keepdims=True)
        w_t = np.zeros_like(raw, dtype=float)
        valid = row_sums[:, 0] > 0
        w_t[valid] = raw[valid] / row_sums[valid]
        fitted_rows.append(w_t @ x_t)
    fitted_pool = np.asarray(fitted_rows, dtype=float)
    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

    return {
        "name": "homophily",
        "gamma": gamma_hat,
        "Abar": abar,
        "X_pool": x_pool,
        "Y_pool": y_pool,
        "mse_pool": mse_pool,
        "status": result.message if isinstance(result.message, str) else str(result.message),
        "success": bool(result.success),
        "nit": int(result.nit),
        "objective": float(result.fun),
    }


def build_gamma_line_search_grid(gamma0: float, num_local_points: int = 100) -> Array:
    """Build a nontrivial positive gamma grid for outer line search."""
    local_count = max(int(num_local_points), 5)
    base = max(abs(float(gamma0)), 1e-3)

    # Explore multiple scales because W is row-normalized after the exponential kernel.
    local = np.geomspace(base / 50.0, base * 50.0, num=local_count)
    anchors = np.asarray([0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0], dtype=float)
    gamma_grid = np.unique(np.concatenate([anchors, local]))
    return gamma_grid[gamma_grid >= 0.0]

def fit_homophily_stubborness(
    run_traj_map: Dict[str, Array],
    run_neighbors: Dict[str, Dict[int, List[int]]],
    gamma0: float = 1.0,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    bias0: float = 0.0,
) -> Dict[str, object]:
    """
    Fit homophily + Friedkin-Johnsen stubbornness via outer gamma search.

    Dynamics used in pooled one-step fit:
        y_t = lambda1 * x_init + lambda2 * b + alpha * H_gamma(x_t)
        alpha = 1 - lambda1 - lambda2
        H_gamma(x_t) = W_gamma(x_t) @ x_t
    where b is a global scalar uniform bias.

    Optimization strategy:
        1) Fix gamma on a search grid.
        2) For each gamma, solve a convex OLS problem in
           (lambda1, lambda2, b_tilde=lambda2*bias) with CVXPY.
        3) Recover bias and keep the gamma/lambda/bias triple with smallest pooled MSE.
    """
    run_names = sorted(run_traj_map.keys())
    if not run_names:
        raise ValueError("run_traj_map is empty")

    ref_neighbors = run_neighbors[run_names[0]]
    for rn in run_names[1:]:
        if run_neighbors[rn] != ref_neighbors:
            raise ValueError("RUN_NEIGHBORS must be identical across runs for pooled fitting.")

    x_blocks, y_blocks, x0_blocks = [], [], []
    for run_name in run_names:
        traj = np.asarray(run_traj_map[run_name], dtype=float)
        x, y = build_dataset_from_run(traj)
        x_blocks.append(x)
        y_blocks.append(y)
        x0_blocks.append(np.repeat(traj[0].reshape(1, -1), x.shape[0], axis=0))

    x_pool = np.vstack(x_blocks)
    y_pool = np.vstack(y_blocks)
    x0_pool = np.vstack(x0_blocks)

    n = x_pool.shape[1]
    abar = build_row_normalized_adjacency(ref_neighbors, n)

    def _homophily_step(x_t: Array, gamma: float) -> Array:
        x_t = sanitize_array(x_t).ravel()
        diff = np.abs(x_t[:, None] - x_t[None, :])
        raw = abar * np.exp(-gamma * diff)
        row_sums = raw.sum(axis=1, keepdims=True)
        w_t = np.zeros_like(raw, dtype=float)
        valid = row_sums[:, 0] > 0
        w_t[valid] = raw[valid] / row_sums[valid]
        return w_t @ x_t

    gamma_candidates = build_gamma_line_search_grid(gamma0)
    best_result: Dict[str, object] | None = None
    eps = 1e-8

    for gamma_candidate in gamma_candidates:
        gamma_fixed = float(gamma_candidate)
        homo_pool = np.asarray([_homophily_step(x_pool[t], gamma_fixed) for t in range(x_pool.shape[0])], dtype=float)

        lambda1_var = cp.Variable(nonneg=True)
        lambda2_var = cp.Variable(nonneg=True)
        b_tilde_var = cp.Variable()
        alpha_expr = 1.0 - lambda1_var - lambda2_var

        pred_pool = lambda1_var * x0_pool + b_tilde_var + cp.multiply(alpha_expr, homo_pool)
        objective = cp.Minimize(cp.sum_squares(y_pool - pred_pool))
        constraints = [
            lambda1_var + lambda2_var <= 1.0,
            lambda1_var <= 1.0,
            lambda2_var <= 1.0,
            b_tilde_var <= lambda2_var,
            b_tilde_var >= -lambda2_var,
        ]
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.OSQP, warm_start=True)
        except Exception:
            problem.solve(warm_start=True)

        if lambda1_var.value is None or lambda2_var.value is None or b_tilde_var.value is None:
            continue

        l1_hat = float(np.clip(float(lambda1_var.value), 0.0, 1.0))
        l2_hat = float(np.clip(float(lambda2_var.value), 0.0, 1.0))
        b_tilde_hat = float(b_tilde_var.value)
        b_tilde_hat = float(np.clip(b_tilde_hat, -l2_hat, l2_hat))
        bias_hat = float(b_tilde_hat / l2_hat) if l2_hat > eps else 0.0
        alpha_hat = 1.0 - l1_hat - l2_hat

        fitted_pool = l1_hat * x0_pool + b_tilde_hat + alpha_hat * homo_pool
        mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

        solver_iters = -1
        if problem.solver_stats is not None and problem.solver_stats.num_iters is not None:
            solver_iters = int(problem.solver_stats.num_iters)

        candidate = {
            "gamma": gamma_fixed,
            "bias": bias_hat,
            "b_tilde": b_tilde_hat,
            "lambda1": l1_hat,
            "lambda2": l2_hat,
            "alpha": alpha_hat,
            "mse_pool": mse_pool,
            "status": str(problem.status),
            "success": bool(problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)),
            "nit": solver_iters,
            "objective": float(problem.value) if problem.value is not None else mse_pool,
        }

        if best_result is None or candidate["mse_pool"] < best_result["mse_pool"]:
            best_result = candidate

    if best_result is None:
        raise RuntimeError("Gamma line search did not produce any candidate result")

    gamma_hat = float(best_result["gamma"])
    lambda1_hat = float(best_result["lambda1"])
    lambda2_hat = float(best_result["lambda2"])
    bias_hat = float(best_result["bias"])
    alpha_hat = float(best_result["alpha"])

    fitted_rows = []
    for t in range(x_pool.shape[0]):
        homo_t = _homophily_step(x_pool[t], gamma_hat)
        fitted_t = lambda1_hat * x0_pool[t] + lambda2_hat * bias_hat + alpha_hat * homo_t
        fitted_rows.append(fitted_t)
    fitted_pool = np.asarray(fitted_rows, dtype=float)
    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

    return {
        "name": "homophily_stubborness",
        "gamma": gamma_hat,
        "bias": bias_hat,
        "lambda1": lambda1_hat,
        "lambda2": lambda2_hat,
        "alpha": alpha_hat,
        "gamma_grid": gamma_candidates,
        "Abar": abar,
        "X_pool": x_pool,
        "Y_pool": y_pool,
        "X0_pool": x0_pool,
        "mse_pool": mse_pool,
        "status": str(best_result["status"]),
        "success": bool(best_result["success"]),
        "nit": int(best_result["nit"]),
        "objective": float(best_result["objective"]),
    }


def rollout_with_homophily_stubborness(
    Abar: Array,
    gamma: float,
    bias: float,
    lambda1: float,
    lambda2: float,
    x0: Array,
    horizon: int,
) -> Array:
    """Roll out homophily + FJ stubbornness with fixed lambda1/lambda2."""
    if lambda1 < 0 or lambda2 < 0 or (lambda1 + lambda2) > 1:
        raise ValueError("lambda1 and lambda2 must be nonnegative and satisfy lambda1 + lambda2 <= 1")

    alpha = 1.0 - float(lambda1) - float(lambda2)
    Abar = np.asarray(Abar, dtype=float)

    x_init = sanitize_array(np.asarray(x0, dtype=float).ravel())
    current = x_init.copy()
    bias_vec = np.full_like(current, float(bias), dtype=float)

    predictions = [current.copy()]
    for _ in range(int(horizon)):
        diff = np.abs(current[:, None] - current[None, :])
        raw = Abar * np.exp(-float(gamma) * diff)
        row_sums = raw.sum(axis=1, keepdims=True)
        W = np.zeros_like(raw, dtype=float)
        valid = row_sums[:, 0] > 0
        W[valid] = raw[valid] / row_sums[valid]

        homophily_part = W @ current
        current = float(lambda1) * x_init + float(lambda2) * bias_vec + alpha * homophily_part
        predictions.append(current.copy())

    return np.asarray(predictions, dtype=float)


def rollout_with_homophily(
    Abar: Array,
    gamma: float,
    x0: Array,
    horizon: int,
) -> Array:
    """Roll out the homophily model with row-normalized kernel weights."""
    Abar = np.asarray(Abar, dtype=float)
    current = sanitize_array(np.asarray(x0, dtype=float).ravel())
    predictions = [current.copy()]

    for _ in range(int(horizon)):
        diff = np.abs(current[:, None] - current[None, :])
        raw = Abar * np.exp(-float(gamma) * diff)
        row_sums = raw.sum(axis=1, keepdims=True)
        W = np.zeros_like(raw, dtype=float)
        valid = row_sums[:, 0] > 0
        W[valid] = raw[valid] / row_sums[valid]
        current = W @ current
        predictions.append(current.copy())

    return np.asarray(predictions, dtype=float)


def evaluate_rollout_model(
    run_traj_map: Dict[str, Array],
    rollout_fn: Callable[[Array], Array],
) -> Dict[str, object]:
    run_names = sorted(run_traj_map.keys())
    per_run = {}

    mean_true_curves = []
    mean_pred_curves = []
    var_true_curves = []
    var_pred_curves = []
    wass_curves = []
    transition_mses = []

    for run_name in run_names:
        observed = np.asarray(run_traj_map[run_name], dtype=float)
        predicted = np.asarray(rollout_fn(observed), dtype=float)

        T = min(observed.shape[0], predicted.shape[0])
        observed = observed[:T]
        predicted = predicted[:T]

        mean_true, mean_pred = compute_mean_per_timestep(observed, predicted)
        var_true, var_pred = compute_variance_per_timestep(observed, predicted)
        wass = compute_wasserstein_distance_per_timestep(observed, predicted)
        mse = float(np.mean((observed - predicted) ** 2))

        per_run[run_name] = {
            "observed": observed,
            "predicted": predicted,
            "mean_true": mean_true,
            "mean_pred": mean_pred,
            "var_true": var_true,
            "var_pred": var_pred,
            "wasserstein": wass,
            "transition_mse": mse,
        }

        mean_true_curves.append(mean_true)
        mean_pred_curves.append(mean_pred)
        var_true_curves.append(var_true)
        var_pred_curves.append(var_pred)
        wass_curves.append(wass)
        transition_mses.append(mse)

    def _stack(curves: List[Array]) -> Array:
        curves = [np.asarray(curve, dtype=float).ravel() for curve in curves if len(curve) > 0]
        if not curves:
            return np.empty((0, 0), dtype=float)
        common_T = min(curve.shape[0] for curve in curves)
        return np.stack([curve[:common_T] for curve in curves], axis=0)

    mean_true_stack = _stack(mean_true_curves)
    mean_pred_stack = _stack(mean_pred_curves)
    var_true_stack = _stack(var_true_curves)
    var_pred_stack = _stack(var_pred_curves)
    wass_stack = _stack(wass_curves)

    return {
        "per_run": per_run,
        "mean_true_stack": mean_true_stack,
        "mean_pred_stack": mean_pred_stack,
        "var_true_stack": var_true_stack,
        "var_pred_stack": var_pred_stack,
        "wasserstein_stack": wass_stack,
        "transition_mse_mean": float(np.mean(transition_mses)),
    }
