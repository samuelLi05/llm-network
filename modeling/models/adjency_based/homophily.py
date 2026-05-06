"""Adjacency-based homophily model variants (plain, stubbornness, FJ)."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import cvxpy as cp
import numpy as np

from data_prep import build_dataset_from_run, build_row_normalized_adjacency, sanitize_array
from analysis_utils import (
    compute_mean_per_timestep,
    compute_variance_per_timestep,
    compute_wasserstein_distance_per_timestep,
)

Array = np.ndarray


def _pooled_blocks(run_traj_map: Dict[str, Array]) -> Tuple[Array, Array]:
    run_names = sorted(run_traj_map.keys())
    x_blocks, y_blocks = [], []

    for run_name in run_names:
        traj = np.asarray(run_traj_map[run_name], dtype=float)
        x, y = build_dataset_from_run(traj)
        x_blocks.append(x)
        y_blocks.append(y)

    return np.vstack(x_blocks), np.vstack(y_blocks)


def build_gamma_line_search_grid(gamma0: float, num_local_points: int = 100) -> Array:
    local_count = max(int(num_local_points), 5)
    base = max(abs(float(gamma0)), 1e-3)
    local = np.geomspace(base / 50.0, base * 50.0, num=local_count)
    anchors = np.asarray([0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0], dtype=float)
    gamma_grid = np.unique(np.concatenate([anchors, local]))
    return gamma_grid[gamma_grid >= 0.0]


def fit_homophily(
    run_traj_map: Dict[str, Array],
    run_neighbors: Dict[str, Dict[int, List[int]]],
    gamma0: float = 1.0,
) -> Dict[str, object]:
    x_pool, y_pool = _pooled_blocks(run_traj_map)

    run_names = sorted(run_traj_map.keys())
    ref_neighbors = run_neighbors[run_names[0]]
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

    for gamma_candidate in gamma_candidates:
        gamma_fixed = float(gamma_candidate)
        homo_pool = np.asarray([_homophily_step(x_pool[t], gamma_fixed) for t in range(x_pool.shape[0])], dtype=float)

        lambda_self_var = cp.Variable(nonneg=True)
        pred_pool = lambda_self_var * x_pool + (1.0 - lambda_self_var) * homo_pool
        objective = cp.Minimize(cp.sum_squares(y_pool - pred_pool))
        constraints = [lambda_self_var <= 1.0]
        problem = cp.Problem(objective, constraints)

        problem.solve(solver=cp.OSQP)

        if lambda_self_var.value is None:
            continue

        lambda_self_hat = float(np.clip(float(lambda_self_var.value), 0.0, 1.0))
        alpha_hat = 1.0 - lambda_self_hat
        fitted_pool = lambda_self_hat * x_pool + alpha_hat * homo_pool
        mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

        solver_iters = -1
        if problem.solver_stats is not None and problem.solver_stats.num_iters is not None:
            solver_iters = int(problem.solver_stats.num_iters)

        candidate = {
            "gamma": gamma_fixed,
            "lambda": lambda_self_hat,
            "lambda_self": lambda_self_hat,
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
    lambda_self_hat = float(best_result["lambda_self"])
    alpha_hat = float(best_result["alpha"])

    fitted_rows = []
    for t in range(x_pool.shape[0]):
        homo_t = _homophily_step(x_pool[t], gamma_hat)
        fitted_rows.append(lambda_self_hat * x_pool[t] + alpha_hat * homo_t)
    fitted_pool = np.asarray(fitted_rows, dtype=float)
    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

    return {
        "name": "homophily",
        "gamma": gamma_hat,
        "lambda": lambda_self_hat,
        "lambda_self": lambda_self_hat,
        "alpha": alpha_hat,
        "gamma_grid": gamma_candidates,
        "Abar": abar,
        "X_pool": x_pool,
        "Y_pool": y_pool,
        "mse_pool": mse_pool,
        "status": str(best_result["status"]),
        "success": bool(best_result["success"]),
        "nit": int(best_result["nit"]),
        "objective": float(best_result["objective"]),
    }


def fit_homophily_stubborness(
    run_traj_map: Dict[str, Array],
    run_neighbors: Dict[str, Dict[int, List[int]]],
    gamma0: float = 1.0,
) -> Dict[str, object]:
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

        lambda_self_var = cp.Variable(nonneg=True)
        lambda1_var = cp.Variable(nonneg=True)
        lambda2_var = cp.Variable(nonneg=True)
        b_tilde_var = cp.Variable()
        alpha_expr = 1.0 - lambda_self_var - lambda1_var - lambda2_var

        pred_pool = lambda_self_var * x_pool + lambda1_var * x0_pool + b_tilde_var + cp.multiply(alpha_expr, homo_pool)
        objective = cp.Minimize(cp.sum_squares(y_pool - pred_pool))
        constraints = [
            lambda_self_var + lambda1_var + lambda2_var <= 1.0,
            lambda_self_var <= 1.0,
            lambda1_var <= 1.0,
            lambda2_var <= 1.0,
            b_tilde_var <= lambda2_var,
            b_tilde_var >= -lambda2_var,
        ]
        problem = cp.Problem(objective, constraints)

        problem.solve(solver=cp.OSQP)

        if lambda_self_var.value is None or lambda1_var.value is None or lambda2_var.value is None or b_tilde_var.value is None:
            continue

        lambda_self_hat = float(np.clip(float(lambda_self_var.value), 0.0, 1.0))
        l1_hat = float(np.clip(float(lambda1_var.value), 0.0, 1.0))
        l2_hat = float(np.clip(float(lambda2_var.value), 0.0, 1.0))
        b_tilde_hat = float(b_tilde_var.value)
        b_tilde_hat = float(np.clip(b_tilde_hat, -l2_hat, l2_hat))
        bias_hat = float(b_tilde_hat / l2_hat) if l2_hat > eps else 0.0
        alpha_hat = 1.0 - lambda_self_hat - l1_hat - l2_hat

        fitted_pool = lambda_self_hat * x_pool + l1_hat * x0_pool + b_tilde_hat + alpha_hat * homo_pool
        mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

        solver_iters = -1
        if problem.solver_stats is not None and problem.solver_stats.num_iters is not None:
            solver_iters = int(problem.solver_stats.num_iters)

        candidate = {
            "gamma": gamma_fixed,
            "lambda_self": lambda_self_hat,
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
    lambda_self_hat = float(best_result["lambda_self"])
    lambda1_hat = float(best_result["lambda1"])
    lambda2_hat = float(best_result["lambda2"])
    bias_hat = float(best_result["bias"])
    alpha_hat = float(best_result["alpha"])

    fitted_rows = []
    for t in range(x_pool.shape[0]):
        homo_t = _homophily_step(x_pool[t], gamma_hat)
        fitted_t = lambda_self_hat * x_pool[t] + lambda1_hat * x0_pool[t] + lambda2_hat * bias_hat + alpha_hat * homo_t
        fitted_rows.append(fitted_t)
    fitted_pool = np.asarray(fitted_rows, dtype=float)
    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

    return {
        "name": "homophily_stubborness",
        "gamma": gamma_hat,
        "lambda_self": lambda_self_hat,
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
    *,
    lambda_self: float = 0.0,
) -> Array:
    if lambda_self < 0 or lambda1 < 0 or lambda2 < 0 or (lambda_self + lambda1 + lambda2) > 1:
        raise ValueError("lambda_self, lambda1 and lambda2 must be nonnegative and satisfy lambda_self + lambda1 + lambda2 <= 1")

    alpha = 1.0 - float(lambda_self) - float(lambda1) - float(lambda2)
    Abar = np.asarray(Abar, dtype=float)
    bias_term = float(lambda2) * float(bias)

    x_init = sanitize_array(np.asarray(x0, dtype=float).ravel())
    current = x_init.copy()

    predictions = [current.copy()]
    for _ in range(int(horizon)):
        diff = np.abs(current[:, None] - current[None, :])
        raw = Abar * np.exp(-float(gamma) * diff)
        row_sums = raw.sum(axis=1, keepdims=True)
        w = np.zeros_like(raw, dtype=float)
        valid = row_sums[:, 0] > 0
        w[valid] = raw[valid] / row_sums[valid]

        homophily_part = w @ current
        current = float(lambda_self) * current + float(lambda1) * x_init + bias_term + alpha * homophily_part
        predictions.append(current.copy())

    return np.asarray(predictions, dtype=float)


def rollout_with_homophily(
    Abar: Array,
    gamma: float,
    x0: Array,
    horizon: int,
    *,
    lambda_self: float = 0.0,
    lambda1: float = 0.0,
) -> Array:
    if lambda_self < 0 or lambda1 < 0 or (lambda_self + lambda1) > 1:
        raise ValueError("lambda_self and lambda1 must be nonnegative and satisfy lambda_self + lambda1 <= 1")

    Abar = np.asarray(Abar, dtype=float)
    x_init = sanitize_array(np.asarray(x0, dtype=float).ravel())
    current = x_init.copy()
    predictions = [current.copy()]
    self_weight = float(lambda_self)
    init_weight = float(lambda1)
    homo_weight = 1.0 - self_weight - init_weight

    for _ in range(int(horizon)):
        diff = np.abs(current[:, None] - current[None, :])
        raw = Abar * np.exp(-float(gamma) * diff)
        row_sums = raw.sum(axis=1, keepdims=True)
        w = np.zeros_like(raw, dtype=float)
        valid = row_sums[:, 0] > 0
        w[valid] = raw[valid] / row_sums[valid]
        homophily_part = w @ current
        current = self_weight * current + init_weight * x_init + homo_weight * homophily_part
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

        t_common = min(observed.shape[0], predicted.shape[0])
        observed = observed[:t_common]
        predicted = predicted[:t_common]

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
        common_t = min(curve.shape[0] for curve in curves)
        return np.stack([curve[:common_t] for curve in curves], axis=0)

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


def fit_homophily_friedkin_johnsen(
    run_traj_map: Dict[str, Array],
    run_neighbors: Dict[str, Dict[int, List[int]]],
    gamma0: float = 1.0,
) -> Dict[str, object]:
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

    for gamma_candidate in gamma_candidates:
        gamma_fixed = float(gamma_candidate)
        homo_pool = np.asarray([_homophily_step(x_pool[t], gamma_fixed) for t in range(x_pool.shape[0])], dtype=float)

        lambda_self_var = cp.Variable(nonneg=True)
        lambda1_var = cp.Variable(nonneg=True)
        alpha_expr = 1.0 - lambda_self_var - lambda1_var

        pred_pool = lambda_self_var * x_pool + lambda1_var * x0_pool + cp.multiply(alpha_expr, homo_pool)
        objective = cp.Minimize(cp.sum_squares(y_pool - pred_pool))
        constraints = [lambda_self_var + lambda1_var <= 1.0, lambda_self_var <= 1.0, lambda1_var <= 1.0]
        problem = cp.Problem(objective, constraints)

        problem.solve(solver=cp.OSQP)

        if lambda_self_var.value is None or lambda1_var.value is None:
            continue

        lambda_self_hat = float(np.clip(float(lambda_self_var.value), 0.0, 1.0))
        l1_hat = float(np.clip(float(lambda1_var.value), 0.0, 1.0))
        alpha_hat = 1.0 - lambda_self_hat - l1_hat

        fitted_pool = lambda_self_hat * x_pool + l1_hat * x0_pool + alpha_hat * homo_pool
        mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

        solver_iters = -1
        if problem.solver_stats is not None and problem.solver_stats.num_iters is not None:
            solver_iters = int(problem.solver_stats.num_iters)

        candidate = {
            "gamma": gamma_fixed,
            "lambda_self": lambda_self_hat,
            "lambda1": l1_hat,
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
    lambda_self_hat = float(best_result["lambda_self"])
    lambda1_hat = float(best_result["lambda1"])
    alpha_hat = float(best_result["alpha"])

    fitted_rows = []
    for t in range(x_pool.shape[0]):
        homo_t = _homophily_step(x_pool[t], gamma_hat)
        fitted_t = lambda_self_hat * x_pool[t] + lambda1_hat * x0_pool[t] + alpha_hat * homo_t
        fitted_rows.append(fitted_t)
    fitted_pool = np.asarray(fitted_rows, dtype=float)
    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

    return {
        "name": "homophily_friedkin_johnsen",
        "gamma": gamma_hat,
        "lambda_self": lambda_self_hat,
        "lambda1": lambda1_hat,
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


def rollout_with_homophily_friedkin_johnsen(
    Abar: Array,
    gamma: float,
    lambda1: float,
    x0: Array,
    horizon: int,
    *,
    lambda_self: float = 0.0,
) -> Array:
    if lambda_self < 0 or lambda1 < 0 or (lambda_self + lambda1) > 1:
        raise ValueError("lambda_self and lambda1 must be nonnegative and satisfy lambda_self + lambda1 <= 1")

    alpha = 1.0 - float(lambda_self) - float(lambda1)
    Abar = np.asarray(Abar, dtype=float)

    x_init = sanitize_array(np.asarray(x0, dtype=float).ravel())
    current = x_init.copy()

    predictions = [current.copy()]
    for _ in range(int(horizon)):
        diff = np.abs(current[:, None] - current[None, :])
        raw = Abar * np.exp(-float(gamma) * diff)
        row_sums = raw.sum(axis=1, keepdims=True)
        w = np.zeros_like(raw, dtype=float)
        valid = row_sums[:, 0] > 0
        w[valid] = raw[valid] / row_sums[valid]

        homophily_part = w @ current
        current = float(lambda_self) * current + float(lambda1) * x_init + alpha * homophily_part
        predictions.append(current.copy())

    return np.asarray(predictions, dtype=float)
