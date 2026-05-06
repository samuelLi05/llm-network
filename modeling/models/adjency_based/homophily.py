"""Adjacency-based homophily model variants (plain, stubbornness, FJ).

All fitting/rollout logic is delegated to shared.ablation_core; this module
exposes backward-compatible function names for notebooks and tests.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from shared.ablation_core import (
    fit_opinion_dynamics,
    rollout_opinion_dynamics,
    evaluate_rollout,
    build_gamma_line_search_grid,
    homophily_weight,
)
from analysis_utils import (
    compute_mean_per_timestep,
    compute_variance_per_timestep,
    compute_wasserstein_distance_per_timestep,
)

Array = np.ndarray


def _pooled_blocks(run_traj_map: Dict[str, Array]) -> Tuple[Array, Array]:
    """Pool x, y blocks across runs (deprecated alias)."""
    from data_prep import build_dataset_from_run
    x_blocks, y_blocks = [], []
    for rn in sorted(run_traj_map.keys()):
        traj = np.asarray(run_traj_map[rn], dtype=float)
        x, y = build_dataset_from_run(traj)
        x_blocks.append(x)
        y_blocks.append(y)
    return np.vstack(x_blocks), np.vstack(y_blocks)


def fit_homophily(
    run_traj_map,
    run_neighbors,
    gamma0: float = 1.0,
) -> Dict[str, object]:
    """Plain homophily model: degroot + homophily."""
    r = fit_opinion_dynamics(
        run_traj_map, run_neighbors,
        weight_type="adjacency_scalar",
        features=["degroot", "homophily"],
        gamma0=gamma0,
    )
    r["name"] = "homophily"
    return r


def fit_homophily_stubborness(
    run_traj_map,
    run_neighbors,
    gamma0: float = 1.0,
) -> Dict[str, object]:
    """Homophily + stubbornness model."""
    r = fit_opinion_dynamics(
        run_traj_map, run_neighbors,
        weight_type="adjacency_scalar",
        features=["degroot", "stubbornness", "homophily"],
        gamma0=gamma0,
    )
    r["name"] = "homophily_stubborness"
    return r


def rollout_with_homophily_stubborness(
    Abar: Array,
    gamma: float,
    bias: float,
    lambda1: float,
    lambda2: float,
    x0: Array,
    horizon: int,
    *,
    lambda_self: float = 0.0,
) -> Array:
    """Simulate homophily + stubbornness forward."""
    return rollout_opinion_dynamics(
        Abar, x0, horizon,
        bias=bias,
        lambda_self=lambda_self,
        lambda1=lambda1,
        lambda2=lambda2,
        features=["degroot", "stubbornness", "homophily"],
        use_homophily=True,
        gamma=gamma,
    )


def rollout_with_homophily(
    Abar: Array,
    gamma: float,
    x0: Array,
    horizon: int,
    *,
    lambda_self: float = 0.0,
    lambda1: float = 0.0,
) -> Array:
    """Simulate plain homophily forward."""
    return rollout_opinion_dynamics(
        Abar, x0, horizon,
        lambda_self=lambda_self,
        lambda1=lambda1,
        features=["degroot", "homophily"],
        use_homophily=True,
        gamma=gamma,
    )


def evaluate_rollout_model(
    run_traj_map: Dict[str, Array],
    rollout_fn: Callable[[Array], Array],
) -> Dict[str, object]:
    """Evaluate rollout across runs (deprecated alias for compat)."""
    return evaluate_rollout(run_traj_map, rollout_fn)


def fit_homophily_friedkin_johnsen(
    run_traj_map,
    run_neighbors,
    gamma0: float = 1.0,
) -> Dict[str, object]:
    """Homophily + FJ model."""
    r = fit_opinion_dynamics(
        run_traj_map, run_neighbors,
        weight_type="adjacency_scalar",
        features=["degroot", "fj", "homophily"],
        gamma0=gamma0,
    )
    r["name"] = "homophily_friedkin_johnsen"
    return r


def rollout_with_homophily_friedkin_johnsen(
    Abar: Array,
    gamma: float,
    lambda1: float,
    x0: Array,
    horizon: int,
    *,
    lambda_self: float = 0.0,
) -> Array:
    """Simulate homophily + FJ forward."""
    return rollout_opinion_dynamics(
        Abar, x0, horizon,
        lambda_self=lambda_self,
        lambda1=lambda1,
        features=["degroot", "fj", "homophily"],
        use_homophily=True,
        gamma=gamma,
    )
