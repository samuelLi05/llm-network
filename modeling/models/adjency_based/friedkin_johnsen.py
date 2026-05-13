"""Adjacency-based Friedkin-Johnsen fitting and rollout."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from data_prep import build_dataset_from_run, build_expected_message_matrix


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
    constraints = [gamma >= 0, gamma <= 1, bias >= -1, bias <= 1]

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


def friedkin_johnsen_adjacency_rollout(w, bias, x0, horizon, lambda1, lambda2):
    alpha = 1.0 - lambda1 - lambda2
    x0 = np.asarray(x0, dtype=float)
    current_x = x0.copy()
    predictions = [current_x.copy()]

    for _ in range(horizon):
        current_x = lambda1 * x0 + lambda2 * float(bias) + alpha * (w @ current_x)
        predictions.append(current_x.copy())

    return predictions


def select_friedkin_johnsen_adjacency_lambdas(run_traj_map, run_neighbors, lambda_grid):
    best_result = None
    all_results = []

    for lambda1 in lambda_grid:
        for lambda2 in lambda_grid:
            if lambda1 + lambda2 > 1:
                continue

            adj_result = fit_friedkin_johnsen_adjacency(run_traj_map, run_neighbors, lambda1, lambda2)

            mse_pool = adj_result["mse_pool"]
            result = {
                "lambda1": float(lambda1),
                "lambda2": float(lambda2),
                "mse_pool": mse_pool,
                "gamma": adj_result["gamma"],
                "bias": adj_result["bias"],
            }
            all_results.append(result)

            if best_result is None or mse_pool < best_result["mse_pool"]:
                best_result = result

    return best_result, all_results
