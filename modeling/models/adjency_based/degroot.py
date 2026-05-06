"""Adjacency-based DeGroot fitting and rollout."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from data_prep import build_dataset_from_run, build_row_normalized_adjacency


def fit_degroot_adjacency_scalar(run_traj_map, run_neighbors):
    run_names = sorted(run_traj_map.keys())
    ref_neighbors = run_neighbors[run_names[0]]

    for rn in run_names[1:]:
        if run_neighbors[rn] != ref_neighbors:
            raise ValueError("RUN_NEIGHBORS must be identical across runs for pooled fitting.")

    x_blocks = []
    y_blocks = []

    for rn in run_names:
        traj = np.asarray(run_traj_map[rn], dtype=float)
        x, y = build_dataset_from_run(traj)
        x_blocks.append(x)
        y_blocks.append(y)

    x_pool = np.vstack(x_blocks)
    y_pool = np.vstack(y_blocks)

    n = x_pool.shape[1]
    abar = build_row_normalized_adjacency(ref_neighbors, n)

    gamma = cp.Variable()
    pred_pool = gamma * (x_pool @ abar.T) + (1.0 - gamma) * x_pool

    objective = cp.Minimize(cp.sum_squares(y_pool - pred_pool))
    constraints = [gamma >= 0, gamma <= 1]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)

    if gamma.value is None:
        raise RuntimeError("Adjacency-scalar DeGroot optimization failed to produce a solution.")

    gamma_hat = float(gamma.value)
    w_hat = gamma_hat * abar + (1.0 - gamma_hat) * np.eye(n, dtype=float)
    fitted_pool = x_pool @ w_hat.T
    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

    return {
        "gamma": gamma_hat,
        "Abar": abar,
        "W": w_hat,
        "X_pool": x_pool,
        "Y_pool": y_pool,
        "mse_pool": mse_pool,
        "status": problem.status,
        "objective": float(problem.value) if problem.value is not None else np.nan,
    }


def degroot_rollout_prediction(w, x0, horizon):
    predictions = [x0]
    current_x = np.asarray(x0, dtype=float).copy()
    for _ in range(horizon):
        current_x = w @ current_x
        predictions.append(current_x.copy())
    return predictions
