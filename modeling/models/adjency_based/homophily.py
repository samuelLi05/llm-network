"""Adjacency-based homophily model variants (plain, stubbornness, FJ)."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import cvxpy as cp
import numpy as np

from data_prep import build_dataset_from_run, build_row_normalized_adjacency, sanitize_array

Array = np.ndarray


def _gamma_to_theta(gamma: float) -> float:
    return float(np.log(max(float(gamma), 1e-12)))


def _theta_to_gamma(theta: float) -> float:
    return float(np.exp(float(theta)))


def _make_homophily_step(abar: Array) -> Callable[[Array, float], Array]:
    def _homophily_step(x_t: Array, gamma: float) -> Array:
        x_t = sanitize_array(x_t).ravel()
        diff = np.abs(x_t[:, None] - x_t[None, :])
        raw = abar * np.exp(-gamma * diff)
        row_sums = raw.sum(axis=1, keepdims=True)
        w_t = np.zeros_like(raw, dtype=float)
        valid = row_sums[:, 0] > 0
        w_t[valid] = raw[valid] / row_sums[valid]
        return w_t @ x_t

    return _homophily_step


def _pooled_blocks(run_traj_map: Dict[str, Array]) -> Tuple[Array, Array]:
    run_names = sorted(run_traj_map.keys())
    x_blocks, y_blocks = [], []

    for run_name in run_names:
        traj = np.asarray(run_traj_map[run_name], dtype=float)
        x, y = build_dataset_from_run(traj)
        x_blocks.append(x)
        y_blocks.append(y)

    return np.vstack(x_blocks), np.vstack(y_blocks)


def build_gamma_line_search_grid(
    gamma0: float,
    local_decades: float = 1.0,
    num_local_points: int = 160,
) -> Array:
    base = max(abs(float(gamma0)), 1e-6)
    local_count = max(int(num_local_points), 5)
    span = 10.0 ** max(float(local_decades), 0.5)

    lo = max(base / span, 1e-8)
    hi = max(base * span, lo * 1.0001)
    local = np.geomspace(lo, hi, num=local_count)

    anchors = np.asarray(
        [0.0, base * 0.5, base * 0.8, base, base * 1.25, base * 1.5],
        dtype=float,
    )

    gamma_grid = np.unique(np.concatenate([anchors, local]))
    gamma_grid = gamma_grid[gamma_grid >= 0.0]
    return np.sort(gamma_grid)


def expand_search_region(best_gamma: float, expansion_factor: float = 1.5, points_per_side: int = 80) -> Array:
    best_gamma = max(float(best_gamma), 1e-8)
    expansion_factor = max(float(expansion_factor), 1.1)
    point_count = max(int(points_per_side), 2) * 2
    lower = best_gamma / expansion_factor
    upper = best_gamma * expansion_factor
    return np.geomspace(lower, upper, num=point_count)


def golden_section_search(
    objective: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float:
    left = max(float(min(a, b)), 1e-12)
    right = max(float(max(a, b)), left * 1.0001)

    phi = (1.0 + np.sqrt(5.0)) / 2.0
    invphi = 1.0 / phi

    c = right - (right - left) * invphi
    d = left + (right - left) * invphi
    fc = float(objective(c))
    fd = float(objective(d))

    for _ in range(int(max_iter)):
        if abs(right - left) < tol:
            break

        if fc < fd:
            right = d
            d = c
            fd = fc
            c = right - (right - left) * invphi
            fc = float(objective(c))
        else:
            left = c
            c = d
            fc = fd
            d = left + (right - left) * invphi
            fd = float(objective(d))

    return float((left + right) / 2.0)


def _refine_gamma_search(objective: Callable[[float], float], gamma0: float) -> Tuple[float, Array, Array]:
    coarse_grid = build_gamma_line_search_grid(gamma0)
    coarse_thetas = np.asarray([_gamma_to_theta(gamma) for gamma in coarse_grid], dtype=float)
    coarse_losses = np.asarray([float(objective(float(gamma))) for gamma in coarse_grid], dtype=float)
    coarse_best_idx = int(np.argmin(coarse_losses))
    coarse_best_theta = float(coarse_thetas[coarse_best_idx])

    refined_grid = expand_search_region(_theta_to_gamma(coarse_best_theta))
    refined_thetas = np.asarray([_gamma_to_theta(gamma) for gamma in refined_grid], dtype=float)
    refined_losses = np.asarray([float(objective(float(gamma))) for gamma in refined_grid], dtype=float)
    refined_best_idx = int(np.argmin(refined_losses))

    if len(refined_grid) >= 2:
        left_idx = max(refined_best_idx - 1, 0)
        right_idx = min(refined_best_idx + 1, len(refined_grid) - 1)
        left_theta = float(refined_thetas[left_idx])
        right_theta = float(refined_thetas[right_idx])
        if right_theta > left_theta:
            best_theta = golden_section_search(
                lambda theta: objective(_theta_to_gamma(theta)),
                left_theta,
                right_theta,
            )
            best_gamma = _theta_to_gamma(best_theta)
        else:
            best_gamma = float(refined_grid[refined_best_idx])
    else:
        best_gamma = float(refined_grid[refined_best_idx])

    return best_gamma, coarse_grid, refined_grid


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

    homophily_step = _make_homophily_step(abar)
    candidate_cache: Dict[float, Dict[str, object]] = {}

    def _solve_for_gamma(gamma_fixed: float) -> Dict[str, object] | None:
        gamma_fixed = float(gamma_fixed)
        if gamma_fixed in candidate_cache:
            return candidate_cache[gamma_fixed]

        homo_pool = np.asarray([homophily_step(x_pool[t], gamma_fixed) for t in range(x_pool.shape[0])], dtype=float)

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

    fitted_rows = []
    for t in range(x_pool.shape[0]):
        homo_t = homophily_step(x_pool[t], gamma_hat)
        fitted_rows.append(lambda_self_hat * x_pool[t] + alpha_hat * homo_t)
    fitted_pool = np.asarray(fitted_rows, dtype=float)
    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

    return {
        "name": "homophily",
        "gamma": gamma_hat,
        "lambda": lambda_self_hat,
        "lambda_self": lambda_self_hat,
        "alpha": alpha_hat,
        "gamma_grid": gamma_grid,
        "Abar": abar,
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

    homophily_step = _make_homophily_step(abar)
    candidate_cache: Dict[float, Dict[str, object]] = {}

    def _solve_for_gamma(gamma_fixed: float) -> Dict[str, object] | None:
        gamma_fixed = float(gamma_fixed)
        if gamma_fixed in candidate_cache:
            return candidate_cache[gamma_fixed]

        homo_pool = np.asarray([homophily_step(x_pool[t], gamma_fixed) for t in range(x_pool.shape[0])], dtype=float)

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

    fitted_rows = []
    for t in range(x_pool.shape[0]):
        homo_t = homophily_step(x_pool[t], gamma_hat)
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
        "gamma_grid": gamma_grid,
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

    homophily_step = _make_homophily_step(abar)
    candidate_cache: Dict[float, Dict[str, object]] = {}
    eps = 1e-8

    def _solve_for_gamma(gamma_fixed: float) -> Dict[str, object] | None:
        gamma_fixed = float(gamma_fixed)
        if gamma_fixed in candidate_cache:
            return candidate_cache[gamma_fixed]

        homo_pool = np.asarray([homophily_step(x_pool[t], gamma_fixed) for t in range(x_pool.shape[0])], dtype=float)

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

    fitted_rows = []
    for t in range(x_pool.shape[0]):
        homo_t = homophily_step(x_pool[t], gamma_hat)
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
        "gamma_grid": gamma_grid,
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

