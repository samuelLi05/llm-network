"""Adjacency-based DeGroot fitting and rollout."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from data_prep import build_dataset_from_run, build_row_normalized_adjacency


def fit_degroot_adjacency_scalar(run_traj_map, run_neighbors):
    run_names = sorted(run_traj_map.keys())
    # Allow runs to have differing neighbor structures by aggregating
    # neighbor sets across runs into a single adjacency used for fitting.

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

    # Build a row-normalized adjacency matrix for each run and form
    # a block-wise prediction expression so runs can have different neighbor sets.
    abar_blocks = []
    for rn in run_names:
        neigh = run_neighbors.get(rn, {})
        abar_blocks.append(build_row_normalized_adjacency(neigh, n))

    gamma = cp.Variable()
    pred_blocks = [
        gamma * (x_blocks[i] @ abar_blocks[i].T) + (1.0 - gamma) * x_blocks[i]
        for i in range(len(x_blocks))
    ]
    pred_pool = cp.vstack(pred_blocks)

    objective = cp.Minimize(cp.sum_squares(y_pool - pred_pool))
    constraints = [gamma >= 0, gamma <= 1]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)

    if gamma.value is None:
        raise RuntimeError("Adjacency-scalar DeGroot optimization failed to produce a solution.")

    gamma_hat = float(gamma.value)
    # Build per-run W matrices and per-run fitted blocks
    w_blocks = [gamma_hat * A + (1.0 - gamma_hat) * np.eye(n, dtype=float) for A in abar_blocks]
    fitted_blocks = [x_blocks[i] @ w_blocks[i].T for i in range(len(x_blocks))]
    fitted_pool = np.vstack(fitted_blocks)
    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

    # Map run names to their Abar/W for caller inspection
    abar_map = {run_names[i]: abar_blocks[i] for i in range(len(run_names))}
    w_map = {run_names[i]: w_blocks[i] for i in range(len(run_names))}

    return {
        "name": "degroot_adjacency_scalar",
        "gamma": gamma_hat,
        "gamma_grid": np.asarray([gamma_hat], dtype=float),
        "Abar_blocks": abar_map,
        "W_blocks": w_map,
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
