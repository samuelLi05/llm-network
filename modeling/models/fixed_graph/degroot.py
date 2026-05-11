"""Explicit-W DeGroot fitting and rollout."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from data_prep import build_dataset_from_run


def fit_row_stochastic_W_from_pooled_runs(run_traj_map, run_neighbors):
    run_names = sorted(run_traj_map.keys())
    ref_neighbors = run_neighbors[run_names[0]]

    for run_name in run_names[1:]:
        if run_neighbors[run_name] != ref_neighbors:
            raise ValueError("RUN_NEIGHBORS must be identical across runs for pooled fitting.")

    x_blocks = []
    y_blocks = []
    for run_name in run_names:
        x, y = build_dataset_from_run(np.asarray(run_traj_map[run_name], dtype=float))
        x_blocks.append(x)
        y_blocks.append(y)

    x_pool = np.vstack(x_blocks)
    y_pool = np.vstack(y_blocks)

    n = x_pool.shape[1]
    w = np.zeros((n, n), dtype=float)

    for i in range(n):
        ns = ref_neighbors[i]
        x_ns = x_pool[:, ns]
        y = y_pool[:, i]

        w_ns = cp.Variable(len(ns))
        objective = cp.Minimize(cp.sum_squares(x_ns @ w_ns - y))
        constraints = [w_ns >= 0, cp.sum(w_ns) == 1]
        problem = cp.Problem(objective, constraints)
        problem.solve()

        row = np.zeros(n, dtype=float)
        row[ns] = np.asarray(w_ns.value, dtype=float).ravel()
        w[i] = row

    return w, x_pool, y_pool

def degroot_rollout_prediction(W, x0, horizon):
    predictions = [x0]
    current_x = x0.copy()
    for t in range(horizon):
        current_x = W @ current_x
        predictions.append(current_x.copy())
    return predictions
