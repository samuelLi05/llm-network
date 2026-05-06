"""Adjacency-based Friedkin-Johnsen fitting and rollout.

Delegates to shared.ablation_core; exposes backward-compatible wrappers.
"""
from __future__ import annotations

import numpy as np

from shared.ablation_core import (
    fit_opinion_dynamics,
    rollout_opinion_dynamics,
    select_lambda_grid,
)


def fit_base_friedkin_johnson_adjency(run_traj_map, run_neighbors, lambda1):
    """Base FJ (fixed lambda1, no bias) -- adjacency scalar."""
    r = fit_opinion_dynamics(
        run_traj_map, run_neighbors,
        weight_type="adjacency_scalar",
        features=["degroot", "fj"],
        lambda1=float(lambda1), lambda2=0.0,
    )
    r["X0_pool"] = r.get("X0_pool")
    return r


def base_friedkin_johnsen_adjacency_rollout(w, x0, horizon, lambda1):
    """Simulate base FJ forward (no bias)."""
    return rollout_opinion_dynamics(
        w, x0, horizon,
        lambda1=float(lambda1),
        features=["degroot", "fj"],
    )


def fit_friedkin_johnsen_adjacency(run_traj_map, run_neighbors, lambda1, lambda2):
    """Full FJ -- adjacency scalar with bias."""
    r = fit_opinion_dynamics(
        run_traj_map, run_neighbors,
        weight_type="adjacency_scalar",
        features=["degroot", "fj"],
        lambda1=float(lambda1), lambda2=float(lambda2),
    )
    return r


def friedkin_johnsen_adjacency_rollout(w, bias, x0, horizon, lambda1, lambda2):
    """Simulate FJ forward."""
    return rollout_opinion_dynamics(
        w, x0, horizon,
        bias=bias,
        lambda1=float(lambda1),
        lambda2=float(lambda2),
        features=["degroot", "fj"],
    )


def select_friedkin_johnsen_adjacency_lambdas(run_traj_map, run_neighbors, lambda_grid):
    """Grid search over (lambda1, lambda2) for FJ-adjacency."""
    return select_lambda_grid(
        run_traj_map, run_neighbors,
        lambda_grid,
        weight_type="adjacency_scalar",
        features=["degroot", "fj"],
    )
