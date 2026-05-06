"""Adjacency-based DeGroot fitting and rollout.

Delegates to shared.ablation_core; exposes backward-compatible wrappers.
"""
from __future__ import annotations

from shared.ablation_core import fit_opinion_dynamics, rollout_opinion_dynamics


def fit_degroot_adjacency_scalar(run_traj_map, run_neighbors):
    """Plain DeGroot -- adjacency scalar (degenerate ablation: only degroot)."""
    return fit_opinion_dynamics(
        run_traj_map, run_neighbors,
        weight_type="adjacency_scalar",
        features=["degroot"],
    )


def degroot_rollout_prediction(w, x0, horizon):
    """Simulate DeGroot forward."""
    return rollout_opinion_dynamics(
        w, x0, int(horizon),
        features=["degroot"],
    )
