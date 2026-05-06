"""Explicit-weight Friedkin-Johnsen fitting and rollout.

Delegates to shared.ablation_core; exposes backward-compatible wrappers.
"""
from __future__ import annotations

import numpy as np

from shared.ablation_core import (
    fit_opinion_dynamics,
    rollout_opinion_dynamics,
    select_lambda_grid,
    homophily_weight,
)


def build_x0_from_agent_inits(agent_inits, n):
    x0 = np.full((n,), np.nan, dtype=float)
    for aid, val in agent_inits.items():
        idx = int(aid.split("_", 1)[1]) - 1
        x0[idx] = float(val)
    if np.isnan(x0).any():
        missing = np.where(np.isnan(x0))[0].tolist()
        raise ValueError(f"missing init values for indices: {missing}")
    return x0


def fit_friedkin_johnsen(run_traj_map, run_neighbors, lambda1, lambda2, agent_inits):
    """FJ with fixed lambda1, lambda2 -- explicit row weights."""
    r = fit_opinion_dynamics(
        run_traj_map, run_neighbors,
        weight_type="explicit_row",
        features=["degroot", "fj"],
        lambda1=float(lambda1), lambda2=float(lambda2),
    )
    # return in old format
    w = r["W"]
    b = r.get("b", r.get("bias", 0.0))
    pool_kw = r.get("X_pool"), r.get("Y_pool")
    return w, float(b), r["X_pool"], r["Y_pool"]


def friedkin_johnsen_rollout_prediction(w, bias, x0, horizon, lambda1, lambda2):
    """Simulate explicit FJ forward."""
    return rollout_opinion_dynamics(
        w, x0, int(horizon),
        bias=bias,
        lambda1=float(lambda1),
        lambda2=float(lambda2),
        features=["degroot", "fj"],
    )


def select_friedkin_johnsen_lambdas(run_traj_map, run_neighbors, lambda_grid, agent_inits):
    """Grid search over (lambda1, lambda2) for explicit FJ."""
    return select_lambda_grid(
        run_traj_map, run_neighbors,
        lambda_grid,
        weight_type="explicit_row",
        features=["degroot", "fj"],
    )


def fit_friedkin_johnsen_joint(run_traj_map, run_neighbors, agent_inits, eps=1e-4):
    """Joint FJ -- all params fitted (explicit row, lambda1 fixed at 0.2+)."""
    # For joint fitting, we use explicit_row with all params free
    # This is a special case - use the core with features that enable joint
    return fit_opinion_dynamics(
        run_traj_map, run_neighbors,
        weight_type="explicit_row",
        features=["degroot", "fj"],
        lambda1=0.2, lambda2=None,  # allow lambda2 to be learned
    )


def fit_friedkin_johnsen_joint_traj0(run_traj_map, run_neighbors, eps=1e-4):
    """Joint FJ using trajectory[0] for x0 (explicit row)."""
    return fit_opinion_dynamics(
        run_traj_map, run_neighbors,
        weight_type="explicit_row",
        features=["degroot", "fj"],
        lambda1=0.2, lambda2=None,
    )
