from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

from modeling.models.data_prep import(
    build_dataset_from_run, 
    sanitize_array,
    _make_homophily_step,
)

Array = np.ndarray
EPS = 0.0


def _normalize(values: Array) -> Array:
    values = np.asarray(values, dtype=float).ravel()
    values = np.maximum(values, 0.0)
    total = float(np.sum(values))
    if total <= 0.0:
        return np.full_like(values, 1.0 / max(len(values), 1), dtype=float)
    return values / total


def _build_row_stochastic_W(theta_rows: Array, supports: List[np.ndarray], n: int) -> Array:
    w = np.zeros((n, n), dtype=float)
    for i, support in enumerate(supports):
        row_weights = _normalize(theta_rows[i])
        w[i, support] = row_weights
    return w


def _kernel_homophily_step(x_t: Array, W: Array, gamma: float) -> Array:
    x_t = sanitize_array(np.asarray(x_t, dtype=float)).ravel()
    diff = np.abs(x_t[:, None] - x_t[None, :])
    raw = np.asarray(W, dtype=float) * np.exp(-float(gamma) * diff)
    row_sums = raw.sum(axis=1, keepdims=True)
    h_t = np.zeros_like(raw, dtype=float)
    valid = row_sums[:, 0] > 0
    h_t[valid] = raw[valid] / row_sums[valid]
    return h_t @ x_t


def _stack_run_datasets(run_traj_map: Dict[str, Array]) -> Tuple[Array, Array, Array]:
    run_names = sorted(run_traj_map.keys())
    x_blocks = []
    y_blocks = []
    x0_blocks = []

    for run_name in run_names:
        traj = np.asarray(run_traj_map[run_name], dtype=float)
        x, y = build_dataset_from_run(traj)
        x_blocks.append(x)
        y_blocks.append(y)
        x0_blocks.append(np.repeat(traj[0].reshape(1, -1), x.shape[0], axis=0))

    x_pool = np.vstack(x_blocks)
    y_pool = np.vstack(y_blocks)
    x0_pool = np.vstack(x0_blocks)
    return x_pool, y_pool, x0_pool


def _build_supports(ref_neighbors: Dict[int, List[int]], n: int) -> Tuple[List[np.ndarray], List[int]]:
    supports: List[np.ndarray] = []
    support_sizes: List[int] = []

    for i in range(n):
        ns = np.asarray(ref_neighbors[i], dtype=int)
        # make sure each node has at least self-loop support
        if ns.size == 0:
            ns = np.asarray([i], dtype=int)
        elif i not in ns:
            ns = np.append(ns, i)
            # sort to ensure consistent ordering of supports
            ns = np.sort(ns)
        supports.append(ns)
        support_sizes.append(int(ns.size))

    return supports, support_sizes


def _split_row_params(row_params: Array, support_sizes: List[int]) -> List[np.ndarray]:
    row_params = np.asarray(row_params, dtype=float).ravel()
    rows: List[np.ndarray] = []
    cursor = 0
    for size in support_sizes:
        rows.append(row_params[cursor:cursor + size])
        cursor += size
    return rows


def _evaluate_pool(x_pool: Array, W: Array, gamma: float) -> Array:
    return np.asarray([
        _kernel_homophily_step(x_pool[t], W, gamma)
        for t in range(x_pool.shape[0])
    ], dtype=float)


def _row_block_start(support_sizes: List[int], scale: float = 1.0) -> Array:
    blocks = [np.full(size, scale, dtype=float) for size in support_sizes]
    return np.concatenate(blocks) if blocks else np.asarray([], dtype=float)


def _row_block_start_random(support_sizes: List[int], rng: np.random.Generator) -> Array:
    blocks = [rng.exponential(scale=1.0, size=size) for size in support_sizes]
    return np.concatenate(blocks) if blocks else np.asarray([], dtype=float)


def _random_start_fg_fj_bias(support_sizes: List[int], rng: np.random.Generator, gamma_max: float = 2.0) -> Array:
    """Uniform random start over all parameters for fit_fg_fj_bias_homophily.

    - gamma        ~ Uniform(0, gamma_max)
    - (lambda_h, lambda_i, bias) sampled uniformly subject to
      lambda_h >= 0, lambda_i >= 0, lambda_h + lambda_i + |bias| <= 1
    """
    # Sample lambdas + bias uniformly in the feasible simplex via rejection
    while True:
        lh = rng.uniform(0.0, 1.0)
        li = rng.uniform(0.0, 1.0 - lh)
        b  = rng.uniform(-(1.0 - lh - li), (1.0 - lh - li))
        if lh + li + abs(b) <= 1.0:
            break
    gamma = rng.uniform(0.0, gamma_max)
    row_params = _row_block_start_random(support_sizes, rng)
    return np.concatenate(([gamma, lh, li, b], row_params))


def _run_multistart(
    objective: Callable[[Array], float],
    starts: List[Array],
    bounds: List[Tuple[float | None, float | None]],
    constraints: Tuple[dict, ...] = (),
    show_progress: bool = True,
) -> Tuple[object, float]:
    best_result = None
    best_value = float("inf")

    iterable = tqdm(starts, desc="multistart", leave=False) if show_progress else starts
    for start in iterable:
        result = minimize(
            objective,
            np.asarray(start, dtype=float),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10, "disp": False},
        )
        value = float(result.fun) if result.fun is not None else float("inf")
        if not np.isfinite(value):
            value = float(objective(np.asarray(result.x, dtype=float))) if result.x is not None else float("inf")
        if best_result is None or value < best_value:
            best_result = result
            best_value = value

    if best_result is None:
        raise RuntimeError("Nonlinear optimization did not produce any result")

    return best_result, best_value



def fit_fg_homophily(
    run_traj_map: Dict[str, Array],
    run_neighbors: Dict[str, Dict[int, List[int]]],
    gamma0: float = 1.0,
) -> Dict[str, object]:
    """Fit fixed-graph homophily with a direct nonlinear solve."""
    run_names = sorted(run_traj_map.keys())
    if not run_names:
        raise ValueError("run_traj_map is empty")

    ref_neighbors = run_neighbors[run_names[0]]
    for rn in run_names[1:]:
        if run_neighbors[rn] != ref_neighbors:
            raise ValueError("RUN_NEIGHBORS must be identical across runs for pooled fitting.")

    x_pool, y_pool, _ = _stack_run_datasets(run_traj_map)
    _, n = x_pool.shape
    supports, support_sizes = _build_supports(ref_neighbors, n)
    row_size = int(sum(support_sizes))

    def objective_fn(theta: Array) -> float:
        theta = np.asarray(theta, dtype=float).ravel()
        gamma_val = float(theta[0])
        row_params = theta[1:]
        if gamma_val <= EPS or row_params.size != row_size:
            return float("inf")

        w_hat = _build_row_stochastic_W(_split_row_params(row_params, support_sizes), supports, n)
        homo_pool = _evaluate_pool(x_pool, w_hat, gamma_val)
        residual = y_pool - homo_pool
        return float(np.sum(residual ** 2))

    gamma_base = max(float(gamma0), EPS)
    row_start = _row_block_start(support_sizes, scale=1.0)
    starts = [
        np.concatenate(([gamma_base * 0.25], row_start)),
        np.concatenate(([gamma_base * 0.40], row_start)),
        np.concatenate(([gamma_base * 0.55], row_start)),
        np.concatenate(([gamma_base * 0.70], row_start)),
        np.concatenate(([gamma_base * 0.85], row_start)),
        np.concatenate(([gamma_base * 1.00], row_start)),
        np.concatenate(([gamma_base * 1.25], row_start)),
        np.concatenate(([gamma_base * 1.50], row_start)),
        np.concatenate(([gamma_base * 2.00], row_start)),
        np.concatenate(([gamma_base * 3.00], row_start)),
    ]
    bounds = [(EPS, None)] + [(EPS, None)] * row_size

    result, objective_value = _run_multistart(objective_fn, starts, bounds)

    gamma_hat = float(result.x[0])
    row_params_hat = np.asarray(result.x[1:], dtype=float)
    w_hat = _build_row_stochastic_W(_split_row_params(row_params_hat, support_sizes), supports, n)
    fitted_pool = _evaluate_pool(x_pool, w_hat, gamma_hat)
    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))
    solver_iters = int(getattr(result, "nit", -1))

    return {
        "name": "fg_homophily",
        "gamma": gamma_hat,
        "W": w_hat,
        "X_pool": x_pool,
        "Y_pool": y_pool,
        "mse_pool": mse_pool,
        "gamma_grid": np.asarray([gamma_hat], dtype=float),
        "status": str(result.message),
        "success": bool(result.success),
        "nit": solver_iters,
        "objective": float(objective_value),
    }


def rollout_fg_homophily(
    W: Array,
    gamma: float,
    x0: Array,
    horizon: int,
) -> Array:
    """Rollout fixed-graph homophily predictions with the exponential kernel."""

    homophily_step = _make_homophily_step(np.asarray(W, dtype=float))
    x_init = sanitize_array(np.asarray(x0, dtype=float).ravel())
    current = x_init.copy()
    predictions = [current.copy()]

    for _ in range(int(horizon)):
        homo_influence = homophily_step(current, float(gamma))
        current = homo_influence
        predictions.append(current.copy())

    return np.asarray(predictions, dtype=float)


def fit_fg_fj_homophily(
    run_traj_map: Dict[str, Array],
    run_neighbors: Dict[str, Dict[int, List[int]]],
    gamma0: float = 1.0,
) -> Dict[str, object]:
    """Fit fixed-graph homophily with initial-opinion bias via nonlinear optimization."""
    run_names = sorted(run_traj_map.keys())
    if not run_names:
        raise ValueError("run_traj_map is empty")

    ref_neighbors = run_neighbors[run_names[0]]
    for rn in run_names[1:]:
        if run_neighbors[rn] != ref_neighbors:
            raise ValueError("RUN_NEIGHBORS must be identical across runs for pooled fitting.")

    x_pool, y_pool, x0_pool = _stack_run_datasets(run_traj_map)
    _, n = x_pool.shape
    supports, support_sizes = _build_supports(ref_neighbors, n)
    row_size = int(sum(support_sizes))

    def objective_fn(theta: Array) -> float:
        theta = np.asarray(theta, dtype=float).ravel()
        gamma_val = float(theta[0])
        lambda_homophily = float(theta[1])
        row_params = theta[2:]
        if gamma_val <= EPS or row_params.size != row_size:
            return float("inf")
        if lambda_homophily < 0.0 or lambda_homophily > 1.0:
            return float("inf")

        w_hat = _build_row_stochastic_W(_split_row_params(row_params, support_sizes), supports, n)
        homo_pool = _evaluate_pool(x_pool, w_hat, gamma_val)
        fitted_pool = lambda_homophily * homo_pool + (1.0 - lambda_homophily) * x0_pool
        residual = y_pool - fitted_pool
        return float(np.sum(residual ** 2))

    gamma_base = max(float(gamma0), EPS)
    row_start = _row_block_start(support_sizes, scale=1.0)
    starts = [
        np.concatenate(([gamma_base * 0.20], [0.10], row_start)),
        np.concatenate(([gamma_base * 0.30], [0.15], row_start)),
        np.concatenate(([gamma_base * 0.40], [0.20], row_start)),
        np.concatenate(([gamma_base * 0.50], [0.25], row_start)),
        np.concatenate(([gamma_base * 0.60], [0.30], row_start)),
        np.concatenate(([gamma_base * 0.75], [0.40], row_start)),
        np.concatenate(([gamma_base * 0.90], [0.50], row_start)),
        np.concatenate(([gamma_base * 1.00], [0.60], row_start)),
        np.concatenate(([gamma_base * 1.25], [0.75], row_start)),
        np.concatenate(([gamma_base * 1.50], [0.85], row_start)),
        np.concatenate(([gamma_base * 1.75], [0.90], row_start)),
        np.concatenate(([gamma_base * 2.00], [0.95], row_start)),
        np.concatenate(([gamma_base * 3.00], [0.80], row_start)),
        np.concatenate(([gamma_base * 5.00], [0.50], row_start)),
        np.concatenate(([gamma_base * 8.00], [0.25], row_start)),
    ]
    bounds = [(EPS, None), (0.0, 1.0)] + [(EPS, None)] * row_size

    result, objective_value = _run_multistart(objective_fn, starts, bounds)

    gamma_hat = float(result.x[0])
    lambda_hat = float(result.x[1])
    row_params_hat = np.asarray(result.x[2:], dtype=float)
    w_hat = _build_row_stochastic_W(_split_row_params(row_params_hat, support_sizes), supports, n)
    homo_pool = _evaluate_pool(x_pool, w_hat, gamma_hat)
    fitted_pool = lambda_hat * homo_pool + (1.0 - lambda_hat) * x0_pool
    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))
    solver_iters = int(getattr(result, "nit", -1))

    return {
        "name": "fg_homophily_friedkin_johnsen",
        "gamma": gamma_hat,
        "W": w_hat,
        "lambda": lambda_hat,
        "lambda_homophily": lambda_hat,
        "alpha": 1.0 - lambda_hat,
        "gamma_grid": np.asarray([gamma_hat], dtype=float),
        "X_pool": x_pool,
        "Y_pool": y_pool,
        "X0_pool": x0_pool,
        "mse_pool": mse_pool,
        "status": str(result.message),
        "success": bool(result.success),
        "nit": solver_iters,
        "objective": float(objective_value),
    }


def rollout_fg_fj_homophily(
    W: Array,
    gamma: float,
    x0: Array,
    horizon: int,
    *,
    lambda_homophily: float = 1.0,
) -> Array:
    """Rollout fixed-graph homophily with initial-opinion bias."""

    homophily_step = _make_homophily_step(np.asarray(W, dtype=float))
    x_init = sanitize_array(np.asarray(x0, dtype=float).ravel())
    x0_init = x_init.copy()
    current = x_init.copy()
    predictions = [current.copy()]

    for _ in range(int(horizon)):
        homo_influence = homophily_step(current, float(gamma))
        current = lambda_homophily * homo_influence + (1.0 - lambda_homophily) * x0_init
        predictions.append(current.copy())

    return np.asarray(predictions, dtype=float)


def fit_fg_fj_bias_homophily(
    run_traj_map: Dict[str, Array],
    run_neighbors: Dict[str, Dict[int, List[int]]],
    gamma0: float = 1.0,
    init_mode: str = "uniform",
    n_random_starts: int = 10,
    random_seed: int = 0,
    fixed_start: Array | None = None,
    extra_starts: List[Array] | None = None,
    show_progress: bool = True,
) -> Dict[str, object]:
    """init_mode: 'grid' (deterministic sweep), 'uniform' (all-ones W, uniform scalars),
    or 'random' (fully random starts via _random_start_fg_fj_bias)."""
    """Fit fixed-graph homophily with initial-opinion bias and a signed offset."""
    run_names = sorted(run_traj_map.keys())
    if not run_names:
        raise ValueError("run_traj_map is empty")

    ref_neighbors = run_neighbors[run_names[0]]
    for rn in run_names[1:]:
        if run_neighbors[rn] != ref_neighbors:
            raise ValueError("RUN_NEIGHBORS must be identical across runs for pooled fitting.")

    x_pool, y_pool, x0_pool = _stack_run_datasets(run_traj_map)
    _, n = x_pool.shape
    supports, support_sizes = _build_supports(ref_neighbors, n)
    row_size = int(sum(support_sizes))

    def objective_fn(theta: Array) -> float:
        theta = np.asarray(theta, dtype=float).ravel()
        gamma_val = float(theta[0])
        lambda_homophily = float(theta[1])
        lambda_init = float(theta[2])
        b_tilde = float(theta[3])  # b_tilde = lambda2 * b  (optimised directly)
        row_params = theta[4:]

        # raise exception if row_params size doesn't match row_size
        if row_params.size != row_size:
            raise ValueError(f"Invalid parameters: row_params size {row_params.size} does not match expected row_size {row_size}.")

        w_hat = _build_row_stochastic_W(_split_row_params(row_params, support_sizes), supports, n)
        homo_pool = _evaluate_pool(x_pool, w_hat, gamma_val)
        fitted_pool = lambda_homophily * homo_pool + lambda_init * x0_pool + b_tilde
        residual = y_pool - fitted_pool
        return float(np.sum(residual ** 2))

    gamma_base = max(float(gamma0), EPS)
    rng = np.random.default_rng(random_seed)
    uniform_row = _row_block_start(support_sizes, scale=1.0)
    # Scalar params (gamma, lambda_h, lambda_i, bias) sweep deterministically.
    # Row params vary: half starts use the uniform W, half use independent
    # Exponential draws so that each row of W is a random simplex point.
    scalar_grid = [
        [gamma_base * 0.20, 0.05, 0.90, -0.90],
        [gamma_base * 0.30, 0.10, 0.80, -0.70],
        [gamma_base * 0.40, 0.15, 0.70, -0.50],
        [gamma_base * 0.50, 0.20, 0.60, -0.30],
        [gamma_base * 0.60, 0.25, 0.50, -0.10],
        [gamma_base * 0.70, 0.30, 0.40,  0.10],
        [gamma_base * 0.85, 0.35, 0.30,  0.30],
        [gamma_base * 1.00, 0.40, 0.20,  0.50],
        [gamma_base * 1.20, 0.45, 0.10,  0.70],
        [gamma_base * 1.40, 0.50, 0.00,  0.90],
        [gamma_base * 1.70, 0.55, 0.05, -1.00],
        [gamma_base * 2.00, 0.60, 0.15, -0.75],
        [gamma_base * 2.50, 0.65, 0.35, -0.50],
        [gamma_base * 3.00, 0.70, 0.55, -0.25],
        [gamma_base * 4.00, 0.75, 0.75,  0.00],
        [gamma_base * 5.00, 0.80, 0.95,  0.25],
        [gamma_base * 6.50, 0.85, 0.80,  0.50],
        [gamma_base * 8.00, 0.90, 0.50,  0.75],
        [gamma_base * 10.0, 0.95, 0.20,  1.00],
        [gamma_base * 12.0, 1.00, 0.00,  0.00],
    ]
    if fixed_start is not None:
        starts = [np.asarray(fixed_start, dtype=float)]
    elif init_mode == "random":
        starts = [_random_start_fg_fj_bias(support_sizes, rng) for _ in range(n_random_starts)]
    elif init_mode == "uniform":
        starts = [np.concatenate((scalars, uniform_row)) for scalars in scalar_grid]
    else:  
        raise ValueError(f"Invalid init_mode: {init_mode}. Must be 'grid', 'uniform', or 'random'.")
    if extra_starts is not None:
        starts = list(starts) + [np.asarray(s, dtype=float) for s in extra_starts]
    bounds = [(EPS, None), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0)] + [(EPS, None)] * row_size

    constraints = (
        {"type": "ineq", "fun": lambda theta: 1.0 - float(theta[1]) - float(theta[2]) - float(theta[3])},
        {"type": "ineq", "fun": lambda theta: 1.0 - float(theta[1]) - float(theta[2]) + float(theta[3])},
    )

    result, objective_value = _run_multistart(objective_fn, starts, bounds, constraints, show_progress=show_progress)

    gamma_hat = float(result.x[0])
    lambda_homophily_hat = float(result.x[1])
    lambda_init_hat = float(result.x[2])
    b_tilde_hat = float(result.x[3])
    eps = 1e-9
    b_hat = b_tilde_hat / (1 - lambda_init_hat - lambda_homophily_hat) if (1 - lambda_init_hat - lambda_homophily_hat) > eps else 0.0
    row_params_hat = np.asarray(result.x[4:], dtype=float)
    w_hat = _build_row_stochastic_W(_split_row_params(row_params_hat, support_sizes), supports, n)
    homo_pool = _evaluate_pool(x_pool, w_hat, gamma_hat)
    fitted_pool = lambda_homophily_hat * homo_pool + lambda_init_hat * x0_pool + b_tilde_hat
    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))
    solver_iters = int(getattr(result, "nit", -1))

    return {
        "name": "fg_homophily_fj_bias",
        "gamma": gamma_hat,
        "lambda1": lambda_homophily_hat,
        "lambda2": lambda_init_hat,
        "bias_tilde": b_tilde_hat,
        "bias": b_hat,
        "W": w_hat,
        "X_pool": x_pool,
        "Y_pool": y_pool,
        "X0_pool": x0_pool,
        "mse_pool": mse_pool,
        "gamma_grid": np.asarray([gamma_hat], dtype=float),
        "status": str(result.message),
        "success": bool(result.success),
        "nit": solver_iters,
        "objective": float(objective_value),
        "theta_opt": np.asarray(result.x, dtype=float)
    }


def rollout_fg_fj_bias_homophily(
    W: Array,
    gamma: float,
    x0: Array,
    horizon: int,
    *,
    bias_tilde: float = 0.0,
    lambda1: float = 1.0,
    lambda2: float = 0.0,
) -> Array:
    """Rollout fixed-graph homophily predictions with initial-opinion bias and offset."""
    if lambda1 < 0 or lambda2 < 0 or (lambda1 + lambda2 + abs(float(bias_tilde))) > 1:
        raise ValueError("lambda1 and lambda2 must be nonnegative and satisfy lambda1 + lambda2 + abs(bias_tilde) <= 1")

    homophily_step = _make_homophily_step(np.asarray(W, dtype=float))
    x_init = sanitize_array(np.asarray(x0, dtype=float).ravel())
    x0_init = x_init.copy()
    current = x_init.copy()
    predictions = [current.copy()]
    lambda_homophily = float(lambda1)
    lambda_init = float(lambda2)
    bias_tilde_val = float(bias_tilde)

    for _ in range(int(horizon)):
        homo_influence = homophily_step(current, float(gamma))
        current = lambda_homophily * homo_influence + lambda_init * x0_init + bias_tilde_val
        predictions.append(current.copy())

    return np.asarray(predictions, dtype=float)