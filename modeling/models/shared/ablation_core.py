"""Shared ablation-core: unified opinion-dynamics fitting and rollout.

All model families (adjacency_scalar, explicit_row) combine from the same
ablation feature set: degroot / stubbornness / fj / homophily.
Each model file becomes a thin wrapper that calls into this core.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cvxpy as cp
import numpy as np

from data_prep import build_dataset_from_run, build_row_normalized_adjacency
from analysis_utils import build_gamma_line_search_grid
from analysis_utils import (
    compute_mean_per_timestep,
    compute_variance_per_timestep,
    compute_wasserstein_distance_per_timestep,
)

# ---------------------------------------------------------------------------
# Data prep helpers -- eliminate the ~6 identical pooling blocks
# ---------------------------------------------------------------------------

def validate_neighbors(run_traj_map, run_neighbors):
    """Ensure all runs share the same neighbor structure."""
    run_names = sorted(run_traj_map.keys())
    ref = run_neighbors[run_names[0]]
    for rn in run_names[1:]:
        if run_neighbors[rn] != ref:
            raise ValueError("RUN_NEIGHBORS must be identical across runs for pooled fitting.")
    return ref


def pool_data(traj_map, include_x0=False):
    """Pool x, y (and optionally x0) blocks across runs."""
    x_blocks, y_blocks, x0_blocks = [], [], []
    for rn in sorted(traj_map.keys()):
        traj = np.asarray(traj_map[rn], dtype=float)
        x, y = build_dataset_from_run(traj)
        x_blocks.append(x)
        y_blocks.append(y)
        if include_x0:
            x0_blocks.append(np.repeat(traj[0].reshape(1, -1), x.shape[0], axis=0))
    kwargs = {}
    if include_x0:
        kwargs["X0_pool"] = np.vstack(x0_blocks)
    return np.vstack(x_blocks), np.vstack(y_blocks), kwargs


def homophily_weight(x_t, gamma, abar):
    """Compute homophily-weighted neighbor average for a single timestep."""
    x_t = np.asarray(x_t, dtype=float).ravel()
    diff = np.abs(x_t[:, None] - x_t[None, :])
    raw = abar * np.exp(-gamma * diff)
    row_sums = raw.sum(axis=1, keepdims=True)
    w = np.zeros_like(raw, dtype=float)
    valid = row_sums[:, 0] > 0
    w[valid] = raw[valid] / row_sums[valid]
    return w @ x_t


# ---------------------------------------------------------------------------
# Internal: compute alpha from feature flags
# ---------------------------------------------------------------------------

def _alpha_for(features):
    if "stubbornness" in features:
        return 1.0
    if "fj" in features:
        return None  # computed from lambda1, lambda2
    return 1.0


# ---------------------------------------------------------------------------
# Unified fitting
# ---------------------------------------------------------------------------

FEATURE_DEFAULTS = ["degroot"]


def fit_opinion_dynamics(
    run_traj_map,
    run_neighbors,
    *,
    weight_type: str = "adjacency_scalar",
    features: list = None,
    gamma0: float = 1.0,
    lambda1: float = None,
    lambda2: float = None,
):
    """Unified opinion-dynamics fitting.

    weight_type: "adjacency_scalar" | "explicit_row"
    features: list of ablation feature flags ["degroot","fj","stubbornness","homophily"]
    gamma0: initial gamma for homophily line search
    lambda1, lambda2: fixed FJ parameters (used when weight_type == "adjacency_scalar")
    """
    features = features if features is not None else FEATURE_DEFAULTS
    features = list(features)
    gamma0 = float(gamma0)

    ref = validate_neighbors(run_traj_map, run_neighbors)
    need_x0 = "fj" in features or "stubbornness" in features
    x_pool, y_pool, pool_kw = pool_data(traj_map=run_traj_map, include_x0=need_x0)
    n = x_pool.shape[1]
    abar = build_row_normalized_adjacency(ref, n)

    # Build weight matrix W
    if weight_type == "adjacency_scalar":
        W, gamma_hat = _fit_adjacency_scalar(x_pool, y_pool, abar, features, lambda1, lambda2)
    else:
        W, gamma_hat = _fit_explicit_row(x_pool, y_pool, ref, features)

    # Homophily line search
    hres = {}
    if "homophily" in features:
        W, gamma_hat, hres = _grid_homophily(x_pool, y_pool, abar, features, lambda1, lambda2, gamma0)

    fitted_pool = _predict_fitted(W, x_pool, pool_kw.get("X0_pool"), features, lambda1, lambda2)
    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

    result = {
        "mse_pool": mse_pool,
        "status": "OPTIMAL",
        "objective": mse_pool,
        "gamma": gamma_hat,
        "Abar": abar,
        "W": W,
        "X_pool": x_pool,
        "Y_pool": y_pool,
        "success": True,
        "nit": 0,
    }
    if need_x0:
        result["X0_pool"] = pool_kw["X0_pool"]
    if hres:
        for k in ("bias", "lambda1", "lambda2", "b_tilde", "alpha", "lambda_self", "gamma_grid"):
            if k in hres:
                result[k] = hres[k]

    return result


def _fit_adjacency_scalar(x_pool, y_pool, abar, features, lambda1, lambda2):
    """Fit gamma for adjacency-scalar model."""
    n = x_pool.shape[1]
    gamma = cp.Variable()
    w_expr = gamma * abar + (1.0 - gamma) * np.eye(n)
    pred_pool = x_pool @ w_expr.T
    objective = cp.Minimize(cp.sum_squares(y_pool - pred_pool))
    constraints = [gamma >= 0, gamma <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)
    if gamma.value is None:
        raise RuntimeError("Adjacency-scalar optimization failed.")
    gamma_hat = float(gamma.value)
    W_hat = gamma_hat * abar + (1.0 - gamma_hat) * np.eye(n)
    return W_hat, gamma_hat


def _fit_explicit_row(x_pool, y_pool, ref, features):
    """Fit explicit row-stochastic W with optional FJ/stubbornness."""
    n = x_pool.shape[1]
    w_tilde = cp.Variable((n, n))
    objective_terms = []
    constraints = []
    alpha = _alpha_for(features)

    for i in range(n):
        ns = ref[i]
        if not ns:
            continue
        w_ns = cp.Variable(len(ns))
        pred = alpha * (x_pool[:, ns] @ w_ns)
        objective_terms.append(cp.sum_squares(y_pool[:, i] - pred))
        constraints += [w_ns >= 0, cp.sum(w_ns) == 1]

    objective = cp.Minimize(cp.sum(objective_terms))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)

    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        ns = ref[i]
        if not ns:
            continue
        W[i, ns] = np.asarray(w_ns.value).ravel()
    return W, 0.0


def _predict_fitted(W, x_pool, x0_pool, features, lambda1, lambda2):
    alpha = _alpha_for(features)
    if "fj" in features:
        pred = lambda1 * x0_pool + (1.0 - lambda1) * (x_pool @ W.T) if x0_pool is not None else None
        return pred
    return x_pool @ W.T


def _grid_homophily(x_pool, y_pool, abar, features, lambda1, lambda2, gamma0):
    """Run homophily gamma line search, return updated W and results dict."""
    grid = build_gamma_line_search_grid(gamma0)
    best = None
    best_mse = float("inf")

    for gc in grid:
        homo_pool = np.array([homophily_weight(x_pool[t], gc, abar) for t in range(x_pool.shape[0])], dtype=float)
        c = _solve_homophily_model(x_pool, y_pool, homo_pool, features, lambda1, lambda2)
        mse = c["mse_pool"]
        if mse < best_mse:
            best_mse = mse
            best = c

    result = {"gamma": best["gamma"], "W": best["W"], "mse_pool": best["mse_pool"], "gamma_grid": list(grid)}
    result.update({k: v for k, v in best.items() if k not in ("gamma", "W", "mse_pool")})
    return result["W"], result["gamma"], result


def _solve_homophily_model(x_pool, y_pool, homo_pool, features, lambda1, lambda2):
    if "stubbornness" in features:
        return _solve_stubborness(x_pool, y_pool, homo_pool)
    if "fj" in features:
        return _solve_fj(x_pool, y_pool, homo_pool)
    return _solve_plain(x_pool, y_pool, homo_pool)


def _solve_plain(x_pool, y_pool, homo_pool):
    n = x_pool.shape[1]
    lam = cp.Variable(nonneg=True)
    pred = lam * x_pool + (1.0 - lam) * homo_pool
    problem = cp.Problem(cp.Minimize(cp.sum_squares(y_pool - pred)), [lam <= 1.0])
    problem.solve(solver=cp.OSQP)
    lam_hat = float(np.clip(float(lam.value), 0.0, 1.0))
    alpha = 1.0 - lam_hat
    fitted = lam_hat * x_pool + alpha * homo_pool
    return {"gamma": 0, "W": np.eye(n), "mse_pool": float(np.mean((y_pool - fitted) ** 2)),
            "lambda_self": lam_hat, "alpha": alpha}


def _solve_stubborness(x_pool, y_pool, homo_pool):
    n = x_pool.shape[1]
    lam_s, l1, l2 = cp.Variable(nonneg=True), cp.Variable(nonneg=True), cp.Variable(nonneg=True)
    b = cp.Variable()
    alpha_expr = 1.0 - lam_s - l1 - l2
    pred = lam_s * x_pool + l1 * x_pool + b + alpha_expr * homo_pool
    constraints = [lam_s + l1 + l2 <= 1.0, lam_s <= 1.0, l1 <= 1.0, l2 <= 1.0, b <= l2, b >= -l2]
    problem = cp.Problem(cp.Minimize(cp.sum_squares(y_pool - pred)), constraints)
    problem.solve(solver=cp.OSQP)
    ls = float(np.clip(float(lam_s.value), 0, 1))
    l1h = float(np.clip(float(l1.value), 0, 1))
    l2h = float(np.clip(float(l2.value), 0, 1))
    bh = float(np.clip(float(b.value), -l2h, l2h))
    bias = float(bh / l2h) if l2h > 1e-8 else 0.0
    alpha = 1.0 - ls - l1h - l2h
    fitted = ls * x_pool + l1h * x_pool + bh + alpha * homo_pool
    return {"gamma": 0, "W": np.eye(n), "mse_pool": float(np.mean((y_pool - fitted) ** 2)),
            "lambda_self": ls, "alpha": alpha, "lambda1": l1h, "lambda2": l2h, "bias": bias, "b_tilde": bh}


def _solve_fj(x_pool, y_pool, homo_pool):
    n = x_pool.shape[1]
    lam_s, l1 = cp.Variable(nonneg=True), cp.Variable(nonneg=True)
    alpha_expr = 1.0 - lam_s - l1
    pred = lam_s * x_pool + l1 * x_pool + alpha_expr * homo_pool
    constraints = [lam_s + l1 <= 1.0, lam_s <= 1.0, l1 <= 1.0]
    problem = cp.Problem(cp.Minimize(cp.sum_squares(y_pool - pred)), constraints)
    problem.solve(solver=cp.OSQP)
    ls = float(np.clip(float(lam_s.value), 0, 1))
    l1h = float(np.clip(float(l1.value), 0, 1))
    alpha = 1.0 - ls - l1h
    fitted = ls * x_pool + l1h * x_pool + alpha * homo_pool
    return {"gamma": 0, "W": np.eye(n), "mse_pool": float(np.mean((y_pool - fitted) ** 2)),
            "lambda_self": ls, "alpha": alpha, "lambda1": l1h}


# ---------------------------------------------------------------------------
# Unified rollout
# ---------------------------------------------------------------------------

def rollout_opinion_dynamics(
    W_or_Abar,
    x0,
    horizon,
    *,
    bias=0.0,
    lambda_self=0.0,
    lambda1=0.0,
    lambda2=0.0,
    features: list = None,
    use_homophily=False,
    gamma=0.0,
):
    """Unified forward simulation.

    For adjacency_scalar models pass Abar.
    For explicit_row models pass W.
    use_homophily + gamma controls per-step homophily kernel.
    """
    features = features if features is not None else FEATURE_DEFAULTS
    x0 = np.asarray(x0, dtype=float).ravel()
    current = x0.copy()
    predictions = [current.copy()]

    alpha = _alpha_for(features)

    for _ in range(int(horizon)):
        if use_homophily:
            diff = np.abs(current[:, None] - current[None, :])
            raw = W_or_Abar * np.exp(-gamma * diff)
            row_sums = raw.sum(axis=1, keepdims=True)
            w = np.zeros_like(raw, dtype=float)
            valid = row_sums[:, 0] > 0
            w[valid] = raw[valid] / row_sums[valid]
            current = float(lambda_self) * current + float(lambda1) * x0 + float(lambda2) * float(bias) + float(alpha) * (w @ current)
        else:
            current = float(lambda_self) * current + float(lambda1) * x0 + float(lambda2) * float(bias) + float(alpha) * (current @ W_or_Abar.T)
        predictions.append(current.copy())
    return np.asarray(predictions, dtype=float)


# ---------------------------------------------------------------------------
# Evaluation helper -- replaces evaluate_rollout_model
# ---------------------------------------------------------------------------

def evaluate_rollout(run_traj_map, rollout_fn):
    """Evaluate rollout across runs (identical to evaluate_rollout_model)."""
    per_run, mean_true_curves, mean_pred_curves, var_true_curves, var_pred_curves, wass_curves = {}, [], [], [], [], []
    transition_mses = []
    for rn in sorted(run_traj_map.keys()):
        obs = np.asarray(run_traj_map[rn], dtype=float)
        pred = np.asarray(rollout_fn(obs), dtype=float)
        t = min(obs.shape[0], pred.shape[0])
        obs, pred = obs[:t], pred[:t]
        mt, mp = compute_mean_per_timestep(obs, pred)
        vt, vp = compute_variance_per_timestep(obs, pred)
        w = compute_wasserstein_distance_per_timestep(obs, pred)
        mse = float(np.mean((obs - pred) ** 2))
        per_run[rn] = dict(observed=obs, predicted=pred, mean_true=mt, mean_pred=mp,
                           var_true=vt, var_pred=vp, wasserstein=w, transition_mse=mse)
        mean_true_curves.append(mt)
        mean_pred_curves.append(mp)
        var_true_curves.append(vt)
        var_pred_curves.append(vp)
        wass_curves.append(w)
        transition_mses.append(mse)

    def _stack(curves):
        curves = [np.asarray(c, dtype=float).ravel() for c in curves if len(c) > 0]
        if not curves:
            return np.empty((0, 0), dtype=float)
        ct = min(c.shape[0] for c in curves)
        return np.stack([c[:ct] for c in curves], axis=0)

    return {
        "per_run": per_run,
        "mean_true_stack": _stack(mean_true_curves),
        "mean_pred_stack": _stack(mean_pred_curves),
        "var_true_stack": _stack(var_true_curves),
        "var_pred_stack": _stack(var_pred_curves),
        "wasserstein_stack": _stack(wass_curves),
        "transition_mse_mean": float(np.mean(transition_mses)),
    }


# ---------------------------------------------------------------------------
# Lambda grid search -- replaces select_friedkin_johnsen_lambdas
# ---------------------------------------------------------------------------

def select_lambda_grid(
    run_traj_map,
    run_neighbors,
    lambda_grid,
    *,
    weight_type="adjacency_scalar",
    features=None,
):
    """Grid search over (lambda1, lambda2), return (best, all_results)."""
    features = features or ["degroot", "fj"]
    best_result = None
    all_results = []
    for l1 in lambda_grid:
        for l2 in lambda_grid:
            if l1 + l2 > 1:
                continue
            result = fit_opinion_dynamics(
                run_traj_map, run_neighbors,
                weight_type=weight_type, features=features,
                lambda1=l1, lambda2=l2,
            )
            entry = {"lambda1": float(l1), "lambda2": float(l2), "mse_pool": result["mse_pool"]}
            if "gamma" in result:
                entry["gamma"] = result["gamma"]
            if "bias" in result:
                entry["bias"] = result["bias"]
            all_results.append(entry)
            if best_result is None or result["mse_pool"] < best_result["mse_pool"]:
                best_result = entry
    return best_result, all_results


__all__ = [
    "fit_opinion_dynamics",
    "rollout_opinion_dynamics",
    "evaluate_rollout",
    "select_lambda_grid",
    "validate_neighbors",
    "pool_data",
    "homophily_weight",
]
