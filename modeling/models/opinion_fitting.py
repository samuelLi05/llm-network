"""Homophily model fitting built on baseline adjacency utilities."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

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
