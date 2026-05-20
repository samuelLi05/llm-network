"""Adjacency-based Friedkin-Johnsen fitting and rollout."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from modeling.models.data_prep import build_dataset_from_run, build_expected_message_matrix


def _row_normalize_matrix(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    row_sums = w.sum(axis=1, keepdims=True)
    out = np.zeros_like(w, dtype=float)
    valid = row_sums[:, 0] > 0.0
    out[valid] = w[valid] / row_sums[valid]
    zero_idx = np.where(~valid)[0]
    for i in zero_idx:
        out[i, i] = 1.0
    return out


def _prepare_pooled_blocks(run_traj_map, run_neighbors):
    run_names = sorted(run_traj_map.keys())
    x_blocks = []
    y_blocks = []
    x0_blocks = []

    for rn in run_names:
        traj = np.asarray(run_traj_map[rn], dtype=float)
        x, y = build_dataset_from_run(traj)
        x_blocks.append(x)
        y_blocks.append(y)
        x0_blocks.append(np.repeat(traj[0].reshape(1, -1), x.shape[0], axis=0))

    x_pool = np.vstack(x_blocks)
    y_pool = np.vstack(y_blocks)
    x0_pool = np.vstack(x0_blocks)

    n = x_pool.shape[1]
    abar_blocks = [build_expected_message_matrix(run_neighbors.get(rn, {}), n) for rn in run_names]
    xa_blocks = [x_blocks[i] @ abar_blocks[i].T for i in range(len(x_blocks))]

    return run_names, x_blocks, y_blocks, x0_blocks, x_pool, y_pool, x0_pool, abar_blocks, xa_blocks, n


def fit_base_friedkin_johnsen_adjacency_joint(run_traj_map, run_neighbors, eps=1e-9):
    (
        run_names,
        x_blocks,
        _,
        x0_blocks,
        x_pool,
        y_pool,
        x0_pool,
        abar_blocks,
        xa_blocks,
        n,
    ) = _prepare_pooled_blocks(run_traj_map, run_neighbors)

    lambda1_var = cp.Variable(nonneg=True)
    u_var = cp.Variable(nonneg=True)  # u = alpha
    v_var = cp.Variable(nonneg=True)  # v = alpha * gamma

    pred_blocks = [
        lambda1_var * x0_blocks[i] + u_var * x_blocks[i] + v_var * (xa_blocks[i] - x_blocks[i])
        for i in range(len(x_blocks))
    ]
    pred_pool = cp.vstack(pred_blocks)

    objective = cp.Minimize(cp.sum_squares(y_pool - pred_pool))
    constraints = [
        lambda1_var + u_var == 1.0,
        v_var <= u_var,
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9, verbose=False)

    if lambda1_var.value is None or u_var.value is None or v_var.value is None:
        raise RuntimeError("Adjacency-based base FJ joint optimization failed to produce a solution.")

    lambda1_hat = float(lambda1_var.value)
    alpha_hat = float(u_var.value)
    v_hat = float(v_var.value)

    # Numerical cleanup for tiny negative values.
    lambda1_hat = max(0.0, min(1.0, lambda1_hat))
    alpha_hat = max(0.0, min(1.0, alpha_hat))
    v_hat = max(0.0, min(alpha_hat, v_hat))

    gamma_hat = float(v_hat / alpha_hat) if alpha_hat > eps else 0.0
    gamma_hat = max(0.0, min(1.0, gamma_hat))

    w_hat_blocks = [
        _row_normalize_matrix(gamma_hat * abar_blocks[i] + (1.0 - gamma_hat) * np.eye(n, dtype=float))
        for i in range(len(abar_blocks))
    ]
    fitted_blocks = [lambda1_hat * x0_blocks[i] + alpha_hat * (x_blocks[i] @ w_hat_blocks[i].T) for i in range(len(x_blocks))]
    fitted_pool = np.vstack(fitted_blocks)
    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

    return {
        "lambda1": lambda1_hat,
        "alpha": alpha_hat,
        "gamma": gamma_hat,
        "Abar_blocks": {run_names[i]: abar_blocks[i] for i in range(len(run_names))},
        "W_blocks": {run_names[i]: w_hat_blocks[i] for i in range(len(run_names))},
        "X_pool": x_pool,
        "Y_pool": y_pool,
        "X0_pool": x0_pool,
        "mse_pool": mse_pool,
        "status": problem.status,
        "objective": float(problem.value) if problem.value is not None else np.nan,
    }


def fit_base_friedkin_johnson_adjency(run_traj_map, run_neighbors, lambda1):
    if lambda1 < 0:
        raise ValueError("lambda1 must be nonnegative")

    run_names = sorted(run_traj_map.keys())
    x_blocks = []
    y_blocks = []
    x0_blocks = []

    for rn in run_names:
        traj = np.asarray(run_traj_map[rn], dtype=float)
        x, y = build_dataset_from_run(traj)
        x_blocks.append(x)
        y_blocks.append(y)
        x0_blocks.append(np.repeat(traj[0].reshape(1, -1), x.shape[0], axis=0))

    x_pool = np.vstack(x_blocks)
    y_pool = np.vstack(y_blocks)
    x0_pool = np.vstack(x0_blocks)

    n = x_pool.shape[1]
    abar_blocks = [build_expected_message_matrix(run_neighbors.get(rn, {}), n) for rn in run_names]
    alpha = 1.0 - lambda1

    gamma = cp.Variable()

    # Build blockwise prediction so each run uses its own Abar
    pred_blocks = [
        lambda1 * x0_blocks[i] + alpha * (gamma * (x_blocks[i] @ abar_blocks[i].T) + (1.0 - gamma) * x_blocks[i])
        for i in range(len(x_blocks))
    ]
    pred_pool = cp.vstack(pred_blocks)

    objective = cp.Minimize(cp.sum_squares(y_pool - pred_pool))
    constraints = [gamma >= 0, gamma <= 1]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)

    if gamma.value is None:
        raise RuntimeError("Adjacency-based FJ optimization failed to produce a solution.")

    gamma_hat = float(gamma.value)
    w_hat_blocks = [
        _row_normalize_matrix(gamma_hat * abar_blocks[i] + (1.0 - gamma_hat) * np.eye(n, dtype=float))
        for i in range(len(abar_blocks))
    ]
    fitted_blocks = [lambda1 * x0_blocks[i] + alpha * (x_blocks[i] @ w_hat_blocks[i].T) for i in range(len(x_blocks))]
    fitted_pool = np.vstack(fitted_blocks)
    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

    return {
        "gamma": gamma_hat,
        "Abar_blocks": {run_names[i]: abar_blocks[i] for i in range(len(run_names))},
        "W_blocks": {run_names[i]: w_hat_blocks[i] for i in range(len(run_names))},
        "X_pool": x_pool,
        "Y_pool": y_pool,
        "X0_pool": x0_pool,
        "mse_pool": mse_pool,
        "status": problem.status,
        "objective": float(problem.value) if problem.value is not None else np.nan,
    }


def base_friedkin_johnsen_adjacency_rollout(w, x0, horizon, lambda1):
    alpha = 1.0 - lambda1
    x0 = np.asarray(x0, dtype=float)
    current_x = x0.copy()
    predictions = [current_x.copy()]

    for _ in range(horizon):
        current_x = lambda1 * x0 + alpha * (w @ current_x)
        predictions.append(current_x.copy())

    return predictions


def fit_friedkin_johnsen_adjacency(run_traj_map, run_neighbors, lambda1, lambda2):
    if lambda1 < 0 or lambda2 < 0 or lambda1 + lambda2 > 1:
        raise ValueError("lambda1 and lambda2 must be nonnegative and satisfy lambda1 + lambda2 <= 1")

    run_names = sorted(run_traj_map.keys())
    x_blocks = []
    y_blocks = []
    x0_blocks = []

    for rn in run_names:
        traj = np.asarray(run_traj_map[rn], dtype=float)
        x, y = build_dataset_from_run(traj)
        x_blocks.append(x)
        y_blocks.append(y)
        x0_blocks.append(np.repeat(traj[0].reshape(1, -1), x.shape[0], axis=0))

    x_pool = np.vstack(x_blocks)
    y_pool = np.vstack(y_blocks)
    x0_pool = np.vstack(x0_blocks)

    n = x_pool.shape[1]
    abar_blocks = [build_expected_message_matrix(run_neighbors.get(rn, {}), n) for rn in run_names]
    alpha = 1.0 - lambda1 - lambda2

    gamma = cp.Variable()
    bias = cp.Variable()

    # Build blockwise prediction with per-run Abar and shared scalar bias
    bias_vec = bias * np.ones((n,), dtype=float)
    pred_blocks = [
        lambda1 * x0_blocks[i] + lambda2 * bias_vec[None, :] + alpha * (gamma * (x_blocks[i] @ abar_blocks[i].T) + (1.0 - gamma) * x_blocks[i])
        for i in range(len(x_blocks))
    ]
    pred_pool = cp.vstack(pred_blocks)

    objective = cp.Minimize(cp.sum_squares(y_pool - pred_pool))
    # tighten bias bounds to avoid saturation that can cause multi-step drift
    constraints = [gamma >= 0, gamma <= 1, bias >= -0.5, bias <= 0.5]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)

    if gamma.value is None or bias.value is None:
        raise RuntimeError("Adjacency-based FJ optimization failed to produce a solution.")

    gamma_hat = float(gamma.value)
    bias_hat = float(bias.value)
    w_hat_blocks = [
        _row_normalize_matrix(gamma_hat * abar_blocks[i] + (1.0 - gamma_hat) * np.eye(n, dtype=float))
        for i in range(len(abar_blocks))
    ]
    fitted_blocks = [lambda1 * x0_blocks[i] + lambda2 * bias_hat * np.ones((1, n), dtype=float) + alpha * (x_blocks[i] @ w_hat_blocks[i].T) for i in range(len(x_blocks))]
    fitted_pool = np.vstack(fitted_blocks)
    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

    return {
        "gamma": gamma_hat,
        "bias": bias_hat,
        "Abar_blocks": {run_names[i]: abar_blocks[i] for i in range(len(run_names))},
        "W_blocks": {run_names[i]: w_hat_blocks[i] for i in range(len(run_names))},
        "X_pool": x_pool,
        "Y_pool": y_pool,
        "X0_pool": x0_pool,
        "mse_pool": mse_pool,
        "status": problem.status,
        "objective": float(problem.value) if problem.value is not None else np.nan,
    }


def fit_friedkin_johnsen_adjacency_joint(run_traj_map, run_neighbors, eps=1e-9):
    (
        run_names,
        x_blocks,
        _,
        x0_blocks,
        x_pool,
        y_pool,
        x0_pool,
        abar_blocks,
        xa_blocks,
        n,
    ) = _prepare_pooled_blocks(run_traj_map, run_neighbors)

    lambda1_var = cp.Variable(nonneg=True)
    lambda2_var = cp.Variable(nonneg=True)
    b_tilde_var = cp.Variable()  # b_tilde = lambda2 * bias
    u_var = cp.Variable(nonneg=True)  # u = alpha
    v_var = cp.Variable(nonneg=True)  # v = alpha * gamma

    pred_blocks = [
        lambda1_var * x0_blocks[i] + b_tilde_var + u_var * x_blocks[i] + v_var * (xa_blocks[i] - x_blocks[i])
        for i in range(len(x_blocks))
    ]
    pred_pool = cp.vstack(pred_blocks)

    objective = cp.Minimize(cp.sum_squares(y_pool - pred_pool))
    constraints = [
        lambda1_var + lambda2_var + u_var == 1.0,
        v_var <= u_var,
        b_tilde_var <= lambda2_var,
        b_tilde_var >= -lambda2_var,
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9, verbose=False)

    if (
        lambda1_var.value is None
        or lambda2_var.value is None
        or b_tilde_var.value is None
        or u_var.value is None
        or v_var.value is None
    ):
        raise RuntimeError("Adjacency-based FJ joint optimization failed to produce a solution.")

    lambda1_hat = float(lambda1_var.value)
    lambda2_hat = float(lambda2_var.value)
    alpha_hat = float(u_var.value)
    v_hat = float(v_var.value)
    b_tilde_hat = float(b_tilde_var.value)

    # Numerical cleanup for tiny negatives.
    lambda1_hat = max(0.0, min(1.0, lambda1_hat))
    lambda2_hat = max(0.0, min(1.0, lambda2_hat))
    alpha_hat = max(0.0, min(1.0, alpha_hat))
    v_hat = max(0.0, min(alpha_hat, v_hat))
    b_tilde_hat = float(np.clip(b_tilde_hat, -lambda2_hat, lambda2_hat))

    gamma_hat = float(v_hat / alpha_hat) if alpha_hat > eps else 0.0
    gamma_hat = max(0.0, min(1.0, gamma_hat))
    bias_hat = float(b_tilde_hat / lambda2_hat) if lambda2_hat > eps else 0.0
    bias_hat = max(-1.0, min(1.0, bias_hat))

    w_hat_blocks = [
        _row_normalize_matrix(gamma_hat * abar_blocks[i] + (1.0 - gamma_hat) * np.eye(n, dtype=float))
        for i in range(len(abar_blocks))
    ]
    fitted_blocks = [
        lambda1_hat * x0_blocks[i] + lambda2_hat * bias_hat * np.ones((1, n), dtype=float) + alpha_hat * (x_blocks[i] @ w_hat_blocks[i].T)
        for i in range(len(x_blocks))
    ]
    fitted_pool = np.vstack(fitted_blocks)
    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

    return {
        "lambda1": lambda1_hat,
        "lambda2": lambda2_hat,
        "alpha": alpha_hat,
        "gamma": gamma_hat,
        "bias": bias_hat,
        "b_tilde": b_tilde_hat,
        "Abar_blocks": {run_names[i]: abar_blocks[i] for i in range(len(run_names))},
        "W_blocks": {run_names[i]: w_hat_blocks[i] for i in range(len(run_names))},
        "X_pool": x_pool,
        "Y_pool": y_pool,
        "X0_pool": x0_pool,
        "mse_pool": mse_pool,
        "status": problem.status,
        "objective": float(problem.value) if problem.value is not None else np.nan,
    }


def friedkin_johnsen_adjacency_rollout(w, bias, x0, horizon, lambda1, lambda2):
    alpha = 1.0 - lambda1 - lambda2
    x0 = np.asarray(x0, dtype=float)
    current_x = x0.copy()
    predictions = [current_x.copy()]

    for _ in range(horizon):
        current_x = lambda1 * x0 + lambda2 * float(bias) + alpha * (w @ current_x)
        predictions.append(current_x.copy())

    return predictions


def select_friedkin_johnsen_adjacency_lambdas(run_traj_map, run_neighbors, lambda_grid=None):
    if lambda_grid is not None and len(lambda_grid) > 0:
        # Sweep over lambda1 and lambda2 grid
        best_result = None
        best_mse = float('inf')
        results_list = []
        
        for lambda1 in lambda_grid:
            for lambda2 in lambda_grid:
                if lambda1 + lambda2 <= 1.0:
                    result = fit_friedkin_johnsen_adjacency(run_traj_map, run_neighbors, lambda1, lambda2)
                    mse = result['mse_pool']
                    results_list.append({
                        "lambda1": float(lambda1),
                        "lambda2": float(lambda2),
                        "mse_pool": float(mse),
                        "gamma": float(result['gamma']),
                        "bias": float(result['bias']),
                    })
                    if mse < best_mse:
                        best_mse = mse
                        best_result = results_list[-1]
        
        if best_result is None:
            # Fallback to joint optimization if sweep yields no valid results
            adj_result = fit_friedkin_johnsen_adjacency_joint(run_traj_map, run_neighbors)
            best_result = {
                "lambda1": float(adj_result["lambda1"]),
                "lambda2": float(adj_result["lambda2"]),
                "mse_pool": float(adj_result["mse_pool"]),
                "gamma": float(adj_result["gamma"]),
                "bias": float(adj_result["bias"]),
            }
            results_list = [best_result]
        
        return best_result, results_list
    else:
        # Use joint optimization when no lambda_grid provided
        adj_result = fit_friedkin_johnsen_adjacency_joint(run_traj_map, run_neighbors)
        best_result = {
            "lambda1": float(adj_result["lambda1"]),
            "lambda2": float(adj_result["lambda2"]),
            "mse_pool": float(adj_result["mse_pool"]),
            "gamma": float(adj_result["gamma"]),
            "bias": float(adj_result["bias"]),
        }
        return best_result, [best_result]

def select_base_friedkin_johnsen_adjacency_lambda(run_traj_map, run_neighbors, lambda_grid=None):
    if lambda_grid is not None and len(lambda_grid) > 0:
        # Sweep over lambda1 grid
        best_result = None
        best_mse = float('inf')
        results_list = []
        
        for lambda1 in lambda_grid:
            if 0 <= lambda1 <= 1:
                result = fit_base_friedkin_johnson_adjency(run_traj_map, run_neighbors, lambda1)
                mse = result['mse_pool']
                results_list.append({
                    "lambda1": float(lambda1),
                    "mse_pool": float(mse),
                    "gamma": float(result['gamma']),
                })
                if mse < best_mse:
                    best_mse = mse
                    best_result = results_list[-1]
        
        if best_result is None:
            # Fallback to joint optimization if sweep yields no valid results
            adj_result = fit_base_friedkin_johnsen_adjacency_joint(run_traj_map, run_neighbors)
            best_result = {
                "lambda1": float(adj_result["lambda1"]),
                "mse_pool": float(adj_result["mse_pool"]),
                "gamma": float(adj_result["gamma"]),
            }
            results_list = [best_result]
        
        return best_result, results_list
    else:
        # Use joint optimization when no lambda_grid provided
        adj_result = fit_base_friedkin_johnsen_adjacency_joint(run_traj_map, run_neighbors)
        best_result = {
            "lambda1": float(adj_result["lambda1"]),
            "mse_pool": float(adj_result["mse_pool"]),
            "gamma": float(adj_result["gamma"]),
        }
        return best_result, [best_result]
