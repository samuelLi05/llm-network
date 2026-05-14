"""Adjacency-based homophily model variants (plain, stubbornness, FJ)."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import cvxpy as cp
import numpy as np

from data_prep import(
    build_dataset_from_run,
    build_expected_message_matrix,
    sanitize_array,
    _pooled_blocks,
    _refine_gamma_search,
    _make_homophily_step,
)

Array = np.ndarray


def fit_homophily(
    run_traj_map: Dict[str, Array],
    run_neighbors: Dict[str, Dict[int, List[int]]],
    gamma0: float = 1.0,
) -> Dict[str, object]:
    # Build per-run datasets so each run can have its own adjacency.
    run_names = sorted(run_traj_map.keys())
    x_blocks = []
    y_blocks = []
    for run_name in run_names:
        traj = np.asarray(run_traj_map[run_name], dtype=float)
        x, y = build_dataset_from_run(traj)
        x_blocks.append(x)
        y_blocks.append(y)

    x_pool = np.vstack(x_blocks)
    y_pool = np.vstack(y_blocks)
    n = x_pool.shape[1]

    # Build expected-message adjacency per run and create homophily_step per run
    abar_blocks = [build_expected_message_matrix(run_neighbors.get(rn, {}), n) for rn in run_names]
    homophily_steps = [_make_homophily_step(A) for A in abar_blocks]
    candidate_cache: Dict[float, Dict[str, object]] = {}

    def _solve_for_gamma(gamma_fixed: float) -> Dict[str, object] | None:
        gamma_fixed = float(gamma_fixed)
        if gamma_fixed in candidate_cache:
            return candidate_cache[gamma_fixed]

        # Compute homo_pool blockwise so each run uses its own adjacency
        homo_blocks = [
            np.asarray([homophily_steps[i](x_blocks[i][t], gamma_fixed) for t in range(x_blocks[i].shape[0])], dtype=float)
            for i in range(len(x_blocks))
        ]
        homo_pool = np.vstack(homo_blocks)

        lambda_self_var = cp.Variable(nonneg=True)
        pred_pool = lambda_self_var * x_pool + (1.0 - lambda_self_var) * homo_pool
        objective = cp.Minimize(cp.sum_squares(y_pool - pred_pool))
        constraints = [0.0 <= lambda_self_var, lambda_self_var <= 1.0]
        problem = cp.Problem(objective, constraints)

        problem.solve(solver=cp.OSQP)

        if lambda_self_var.value is None:
            return None

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
        candidate_cache[gamma_fixed] = candidate
        return candidate

    def _gamma_objective(gamma_fixed: float) -> float:
        candidate = _solve_for_gamma(gamma_fixed)
        return float(candidate["mse_pool"]) if candidate is not None else float("inf")

    gamma_hat, gamma_candidates, gamma_refined = _refine_gamma_search(_gamma_objective, gamma0)
    gamma_grid = np.unique(np.concatenate([gamma_candidates, gamma_refined]))

    best_result = _solve_for_gamma(gamma_hat)
    if best_result is None:
        raise RuntimeError("Gamma line search did not produce any candidate result")

    gamma_hat = float(best_result["gamma"])
    lambda_self_hat = float(best_result["lambda_self"])
    alpha_hat = float(best_result["alpha"])

    # reconstruct fitted blocks using homophily_steps with gamma_hat
    homo_blocks = [
        np.asarray([homophily_steps[i](x_blocks[i][t], gamma_hat) for t in range(x_blocks[i].shape[0])], dtype=float)
        for i in range(len(x_blocks))
    ]
    fitted_pool = np.vstack([
        lambda_self_hat * x_blocks[i] + alpha_hat * homo_blocks[i]
        for i in range(len(x_blocks))
    ])
    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

    # build per-run representative W (time-averaged row-stochastic matrices)
    W_blocks = {}
    for i, A in enumerate(abar_blocks):
        x_block = x_blocks[i]
        w_time = []
        for t in range(x_block.shape[0]):
            diff = np.abs(x_block[t][:, None] - x_block[t][None, :])
            raw = A * np.exp(-float(gamma_hat) * diff)
            row_sums = raw.sum(axis=1, keepdims=True)
            w_t = np.zeros_like(raw, dtype=float)
            valid = row_sums[:, 0] > 0
            w_t[valid] = raw[valid] / row_sums[valid]
            w_time.append(w_t)
        W_blocks[run_names[i]] = np.mean(np.asarray(w_time, dtype=float), axis=0)

    return {
        "name": "homophily",
        "gamma": gamma_hat,
        "lambda": lambda_self_hat,
        "lambda_self": lambda_self_hat,
        "alpha": alpha_hat,
        "gamma_grid": gamma_grid,
        "Abar_blocks": {run_names[i]: abar_blocks[i] for i in range(len(run_names))},
        "W_blocks": W_blocks,
        "X_pool": x_pool,
        "Y_pool": y_pool,
        "mse_pool": mse_pool,
        "status": str(best_result["status"]),
        "success": bool(best_result["success"]),
        "nit": int(best_result["nit"]),
        "objective": float(best_result["objective"]),
    }


def fit_homophily_friedkin_johnsen(
    run_traj_map: Dict[str, Array],
    run_neighbors: Dict[str, Dict[int, List[int]]],
    gamma0: float = 1.0,
) -> Dict[str, object]:
    run_names = sorted(run_traj_map.keys())
    if not run_names:
        raise ValueError("run_traj_map is empty")

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
    abar_blocks = [build_expected_message_matrix(run_neighbors.get(rn, {}), n) for rn in run_names]
    homophily_steps = [_make_homophily_step(A) for A in abar_blocks]
    candidate_cache: Dict[float, Dict[str, object]] = {}

    def _solve_for_gamma(gamma_fixed: float) -> Dict[str, object] | None:
        gamma_fixed = float(gamma_fixed)
        if gamma_fixed in candidate_cache:
            return candidate_cache[gamma_fixed]

        homo_blocks = [
            np.asarray([homophily_steps[i](x_blocks[i][t], gamma_fixed) for t in range(x_blocks[i].shape[0])], dtype=float)
            for i in range(len(x_blocks))
        ]
        homo_pool = np.vstack(homo_blocks)

        lambda_self_var = cp.Variable(nonneg=True)
        lambda1_var = cp.Variable(nonneg=True)
        alpha_expr = 1.0 - lambda_self_var - lambda1_var

        pred_pool = lambda_self_var * x_pool + lambda1_var * x0_pool + cp.multiply(alpha_expr, homo_pool)
        objective = cp.Minimize(cp.sum_squares(y_pool - pred_pool))
        constraints = [lambda_self_var + lambda1_var <= 1.0, lambda_self_var <= 1.0, lambda1_var <= 1.0, lambda_self_var >= 0, lambda1_var >= 0]
        problem = cp.Problem(objective, constraints)

        problem.solve(solver=cp.OSQP)

        if lambda_self_var.value is None or lambda1_var.value is None:
            return None

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
        candidate_cache[gamma_fixed] = candidate
        return candidate

    def _gamma_objective(gamma_fixed: float) -> float:
        candidate = _solve_for_gamma(gamma_fixed)
        return float(candidate["mse_pool"]) if candidate is not None else float("inf")

    gamma_hat, gamma_candidates, gamma_refined = _refine_gamma_search(_gamma_objective, gamma0)
    gamma_grid = np.unique(np.concatenate([gamma_candidates, gamma_refined]))

    best_result = _solve_for_gamma(gamma_hat)
    if best_result is None:
        raise RuntimeError("Gamma line search did not produce any candidate result")

    gamma_hat = float(best_result["gamma"])
    lambda_self_hat = float(best_result["lambda_self"])
    lambda1_hat = float(best_result["lambda1"])
    alpha_hat = float(best_result["alpha"])

    # reconstruct fitted blocks using homophily_steps with gamma_hat
    homo_blocks = [
        np.asarray([homophily_steps[i](x_blocks[i][t], gamma_hat) for t in range(x_blocks[i].shape[0])], dtype=float)
        for i in range(len(x_blocks))
    ]
    fitted_pool = np.vstack([
        lambda_self_hat * x_blocks[i] + lambda1_hat * x0_blocks[i] + alpha_hat * homo_blocks[i]
        for i in range(len(x_blocks))
    ])
    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

    # build per-run representative W (time-averaged row-stochastic matrices)
    W_blocks = {}
    for i, A in enumerate(abar_blocks):
        x_block = x_blocks[i]
        w_time = []
        for t in range(x_block.shape[0]):
            diff = np.abs(x_block[t][:, None] - x_block[t][None, :])
            raw = A * np.exp(-float(gamma_hat) * diff)
            row_sums = raw.sum(axis=1, keepdims=True)
            w_t = np.zeros_like(raw, dtype=float)
            valid = row_sums[:, 0] > 0
            w_t[valid] = raw[valid] / row_sums[valid]
            w_time.append(w_t)
        W_blocks[run_names[i]] = np.mean(np.asarray(w_time, dtype=float), axis=0)

    return {
        "name": "homophily_friedkin_johnsen",
        "gamma": gamma_hat,
        "lambda_self": lambda_self_hat,
        "lambda1": lambda1_hat,
        "alpha": alpha_hat,
        "gamma_grid": gamma_grid,
        "Abar_blocks": {run_names[i]: abar_blocks[i] for i in range(len(run_names))},
        "W_blocks": W_blocks,
        "X_pool": x_pool,
        "Y_pool": y_pool,
        "X0_pool": x0_pool,
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
    abar_blocks = [build_expected_message_matrix(run_neighbors.get(rn, {}), n) for rn in run_names]
    homophily_steps = [_make_homophily_step(A) for A in abar_blocks]
    candidate_cache: Dict[float, Dict[str, object]] = {}
    eps = 1e-8

    def _solve_for_gamma(gamma_fixed: float) -> Dict[str, object] | None:
        gamma_fixed = float(gamma_fixed)
        if gamma_fixed in candidate_cache:
            return candidate_cache[gamma_fixed]

        homo_blocks = [
            np.asarray([homophily_steps[i](x_blocks[i][t], gamma_fixed) for t in range(x_blocks[i].shape[0])], dtype=float)
            for i in range(len(x_blocks))
        ]
        homo_pool = np.vstack(homo_blocks)

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
            return None

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
        candidate_cache[gamma_fixed] = candidate
        return candidate

    def _gamma_objective(gamma_fixed: float) -> float:
        candidate = _solve_for_gamma(gamma_fixed)
        return float(candidate["mse_pool"]) if candidate is not None else float("inf")

    gamma_hat, gamma_candidates, gamma_refined = _refine_gamma_search(_gamma_objective, gamma0)
    gamma_grid = np.unique(np.concatenate([gamma_candidates, gamma_refined]))

    best_result = _solve_for_gamma(gamma_hat)
    if best_result is None:
        raise RuntimeError("Gamma line search did not produce any candidate result")

    gamma_hat = float(best_result["gamma"])
    lambda_self_hat = float(best_result["lambda_self"])
    lambda1_hat = float(best_result["lambda1"])
    lambda2_hat = float(best_result["lambda2"])
    bias_hat = float(best_result["bias"])
    alpha_hat = float(best_result["alpha"])

    homo_blocks = [
        np.asarray([homophily_steps[i](x_blocks[i][t], gamma_hat) for t in range(x_blocks[i].shape[0])], dtype=float)
        for i in range(len(x_blocks))
    ]
    fitted_pool = np.vstack([
        lambda_self_hat * x_blocks[i] + lambda1_hat * x0_blocks[i] + lambda2_hat * bias_hat + alpha_hat * homo_blocks[i]
        for i in range(len(x_blocks))
    ])
    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

    # build per-run representative W (time-averaged row-stochastic matrices)
    W_blocks = {}
    for i, A in enumerate(abar_blocks):
        x_block = x_blocks[i]
        w_time = []
        for t in range(x_block.shape[0]):
            diff = np.abs(x_block[t][:, None] - x_block[t][None, :])
            raw = A * np.exp(-float(gamma_hat) * diff)
            row_sums = raw.sum(axis=1, keepdims=True)
            w_t = np.zeros_like(raw, dtype=float)
            valid = row_sums[:, 0] > 0
            w_t[valid] = raw[valid] / row_sums[valid]
            w_time.append(w_t)
        W_blocks[run_names[i]] = np.mean(np.asarray(w_time, dtype=float), axis=0)

    return {
        "name": "homophily_stubborness",
        "gamma": gamma_hat,
        "lambda_self": lambda_self_hat,
        "bias": bias_hat,
        "lambda1": lambda1_hat,
        "lambda2": lambda2_hat,
        "alpha": alpha_hat,
        "gamma_grid": gamma_grid,
        "Abar_blocks": {run_names[i]: abar_blocks[i] for i in range(len(run_names))},
        "W_blocks": W_blocks,
        "X_pool": x_pool,
        "Y_pool": y_pool,
        "X0_pool": x0_pool,
        "mse_pool": mse_pool,
        "status": str(best_result["status"]),
        "success": bool(best_result["success"]),
        "nit": int(best_result["nit"]),
        "objective": float(best_result["objective"]),
    }


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

