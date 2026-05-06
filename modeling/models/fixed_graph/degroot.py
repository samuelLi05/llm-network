"""Explicit-weight DeGroot fitting and rollout.

Delegates to shared.ablation_core; exposes backward-compatible wrappers.
"""
from __future__ import annotations

from shared.ablation_core import fit_opinion_dynamics, rollout_opinion_dynamics


def fit_row_stochastic_W_from_pooled_runs(run_traj_map, run_neighbors):
    """Plain DeGroot -- explicit row weights (degenerate ablation)."""
    r = fit_opinion_dynamics(
        run_traj_map, run_neighbors,
        weight_type="explicit_row",
        features=["degroot"],
    )
    return r["W"], r["X_pool"], r["Y_pool"]
