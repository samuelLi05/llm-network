"""Explicit-W Friedkin-Johnsen fitting and rollout."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from modeling.models.data_prep import build_dataset_from_run, build_x0_from_agent_inits

def fit_friedkin_johnsen_no_bias(run_traj_map, run_neighbors, lamda1, agent_inits):
    run_names = sorted(run_traj_map.keys())
    ref_neighbors = run_neighbors[run_names[0]]

    for rn in run_names[1:]:
        if run_neighbors[rn] != ref_neighbors:
            raise ValueError("RUN_NEIGHBORS must be identical across runs")
    
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
    alpha = 1.0 - lamda1

    w_vars = []
    objective_terms = []
    x0_init = build_x0_from_agent_inits(agent_inits, n)

    for i in range(n):
        ns = ref_neighbors[i]
        if len(ns) == 0:
            continue

        w_ns = cp.Variable(len(ns))
        w_vars.append((i, ns, w_ns))

        x_ns = x_pool[:, ns]
        y = y_pool[:, i]
        x0i = float(x0_init[i])

        pred = lamda1 * x0i + alpha * (x_ns @ w_ns)
        objective_terms.append(cp.sum_squares(y - pred))

        constraints = [w_ns >= 0, cp.sum(w_ns) == alpha]
        objective_terms.append(cp.sum_squares(y - pred))

    objective = cp.Minimize(cp.sum(objective_terms))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    if prob.status != cp.OPTIMAL:
        raise RuntimeError(f"Friedkin-Johnsen optimization failed: status={prob.status}")
    
    w = np.zeros((n, n), dtype=float)
    for (i, ns, w_ns) in w_vars:
        w[i, ns] = np.asarray(w_ns.value, dtype=float).ravel()

    return w, x_pool, y_pool
def select_lambda_friedkin_johnsen_no_bias(run_traj_map, run_neighbors, lambda_grid, agent_inits):
    best_result = None
    all_results = []

    for lambda1 in lambda_grid:
        w_hat, x_pool, y_pool = fit_friedkin_johnsen_no_bias(run_traj_map, run_neighbors, lambda1, agent_inits)

        n = x_pool.shape[1]
        x0 = build_x0_from_agent_inits(agent_inits, n)
        alpha = 1.0 - lambda1
        pred_pool = lambda1 * x0[None, :] + alpha * (x_pool @ w_hat.T)
        mse_pool = float(np.mean((y_pool - pred_pool) ** 2))

        result = {
            "lambda1": float(lambda1),
            "mse_pool": mse_pool,
        }
        all_results.append(result)

        if best_result is None or mse_pool < best_result["mse_pool"]:
            best_result = result

    return best_result, all_results


def fit_friedkin_johnsen(run_traj_map, run_neighbors, lambda1, lambda2, agent_inits):
    if lambda1 < 0 or lambda2 < 0 or lambda1 + lambda2 > 1:
        raise ValueError("lambda1 and lambda2 must be nonnegative and satisfy lambda1 + lambda2 <= 1")

    run_names = sorted(run_traj_map.keys())
    ref_neighbors = run_neighbors[run_names[0]]

    for rn in run_names[1:]:
        if run_neighbors[rn] != ref_neighbors:
            raise ValueError("RUN_NEIGHBORS must be identical across runs")

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
    alpha = 1.0 - lambda1 - lambda2

    b = cp.Variable()
    w_vars = []
    objective_terms = []
    constraints = []

    x0_init = build_x0_from_agent_inits(agent_inits, n)

    for i in range(n):
        ns = ref_neighbors[i]
        if len(ns) == 0:
            continue

        w_ns = cp.Variable(len(ns))
        w_vars.append((i, ns, w_ns))

        x_ns = x_pool[:, ns]
        y = y_pool[:, i]
        x0i = float(x0_init[i])

        pred = lambda1 * x0i + lambda2 * b + alpha * (x_ns @ w_ns)
        objective_terms.append(cp.sum_squares(y - pred))

        constraints += [w_ns >= 0, cp.sum(w_ns) == 1]

    constraints += [b >= -1, b <= 1]

    objective = cp.Minimize(cp.sum(objective_terms))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    if b.value is None:
        raise RuntimeError("Solver failed")

    w = np.zeros((n, n), dtype=float)
    for (i, ns, w_ns) in w_vars:
        w[i, ns] = np.asarray(w_ns.value).ravel()

    return w, float(b.value), x_pool, y_pool


def friedkin_johnsen_rollout_prediction(w, bias, x0, horizon, lambda1, lambda2):
    alpha = 1.0 - lambda1 - lambda2
    x0 = np.asarray(x0, dtype=float)
    current_x = x0.copy()
    predictions = [current_x.copy()]

    for _ in range(horizon):
        current_x = lambda1 * x0 + lambda2 * float(bias) + alpha * (w @ current_x)
        predictions.append(current_x.copy())

    return predictions


def select_friedkin_johnsen_lambdas(run_traj_map, run_neighbors, lambda_grid, agent_inits):
    best_result = None
    all_results = []

    for lambda1 in lambda_grid:
        for lambda2 in lambda_grid:
            if lambda1 + lambda2 > 1:
                continue

            w_hat, b_hat, x_pool, y_pool = fit_friedkin_johnsen(
                run_traj_map,
                run_neighbors,
                lambda1,
                lambda2,
                agent_inits,
            )

            n = x_pool.shape[1]
            x0 = build_x0_from_agent_inits(agent_inits, n)
            alpha = 1.0 - lambda1 - lambda2
            pred_pool = lambda1 * x0[None, :] + lambda2 * b_hat + alpha * (x_pool @ w_hat.T)
            mse_pool = float(np.mean((y_pool - pred_pool) ** 2))

            result = {
                "lambda1": float(lambda1),
                "lambda2": float(lambda2),
                "mse_pool": mse_pool,
            }
            all_results.append(result)

            if best_result is None or mse_pool < best_result["mse_pool"]:
                best_result = result

    return best_result, all_results


def fit_friedkin_johnsen_joint(run_traj_map, run_neighbors, agent_inits, eps=1e-4):
    run_names = sorted(run_traj_map.keys())
    ref_neighbors = run_neighbors[run_names[0]]

    for rn in run_names[1:]:
        if run_neighbors[rn] != ref_neighbors:
            raise ValueError("RUN_NEIGHBORS must be identical across runs for pooled fitting.")

    x_blocks = []
    y_blocks = []

    for rn in run_names:
        traj = np.asarray(run_traj_map[rn], dtype=float)
        x, y = build_dataset_from_run(traj)
        x_blocks.append(x)
        y_blocks.append(y)

    x_pool = np.vstack(x_blocks)
    y_pool = np.vstack(y_blocks)
    _, n = x_pool.shape
    x0_init = build_x0_from_agent_inits(agent_inits, n)
    x0_pool = np.repeat(x0_init.reshape(1, -1), x_pool.shape[0], axis=0)

    lambda1 = cp.Variable(nonneg=True)
    lambda2 = cp.Variable(nonneg=True)
    b_tilde = cp.Variable()
    alpha = 1.0 - lambda1 - lambda2
    w_tilde = cp.Variable((n, n))

    ones_n = np.ones((n,), dtype=float)
    residual = y_pool - (lambda1 * x0_pool + b_tilde * ones_n[None, :] + x_pool @ w_tilde.T)
    objective = cp.Minimize(cp.sum_squares(residual))
    constraints = [
        lambda2 >= eps,
        lambda1 >= 0.2,
        lambda1 + lambda2 <= 1.0 - eps,
        lambda1 <= 1.0,
    ]

    for i in range(n):
        ns = ref_neighbors[i]
        allowed = np.zeros((n,), dtype=float)
        allowed[np.asarray(ns, dtype=int)] = 1.0

        constraints.append(w_tilde[i, :] >= 0)
        constraints.append(cp.sum(w_tilde[i, :]) == alpha)
        constraints.append(cp.multiply(1.0 - allowed, w_tilde[i, :]) == 0)

    constraints += [b_tilde <= lambda2, b_tilde >= -lambda2]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    if lambda1.value is None or lambda2.value is None or w_tilde.value is None or b_tilde.value is None:
        raise RuntimeError("Joint FJ optimization failed to produce a solution.")

    lambda1_hat = float(lambda1.value)
    lambda2_hat = float(lambda2.value)
    alpha_hat = 1.0 - lambda1_hat - lambda2_hat
    b_tilde_hat = float(b_tilde.value)
    w_tilde_hat = np.asarray(w_tilde.value, dtype=float)

    if alpha_hat <= eps:
        raise RuntimeError(f"Estimated alpha too small for stable W recovery: alpha={alpha_hat}")

    w_hat = w_tilde_hat / alpha_hat
    b_hat = b_tilde_hat / lambda2_hat

    fitted_pool = lambda1_hat * x0_pool + b_tilde_hat * ones_n[None, :] + x_pool @ w_tilde_hat.T
    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

    return {
        "lambda1": lambda1_hat,
        "lambda2": lambda2_hat,
        "alpha": alpha_hat,
        "b_tilde": b_tilde_hat,
        "W_tilde": w_tilde_hat,
        "W": w_hat,
        "b": float(b_hat),
        "X_pool": x_pool,
        "Y_pool": y_pool,
        "X0_pool": x0_pool,
        "mse_pool": mse_pool,
        "status": prob.status,
        "objective": float(prob.value) if prob.value is not None else np.nan,
    }


def fit_friedkin_johnsen_joint_traj0(run_traj_map, run_neighbors, eps=1e-4):
    run_names = sorted(run_traj_map.keys())
    ref_neighbors = run_neighbors[run_names[0]]

    for rn in run_names[1:]:
        if run_neighbors[rn] != ref_neighbors:
            raise ValueError("RUN_NEIGHBORS must be identical across runs for pooled fitting.")

    x_blocks = []
    y_blocks = []
    x0_blocks = []
    for rn in run_names:
        traj = np.asarray(run_traj_map[rn], dtype=float)
        x, y = build_dataset_from_run(traj)
        x_blocks.append(x)
        y_blocks.append(y)
        x0_blocks.append(np.repeat(traj[0].reshape(1, -1), x.shape[0], axis=0))

    x_pool = np.vstack(x_blocks)
    y_pool = np.vstack(y_blocks)
    x0_pool = np.vstack(x0_blocks)
    _, n = x_pool.shape

    lambda1 = cp.Variable(nonneg=True)
    lambda2 = cp.Variable(nonneg=True)
    b_tilde = cp.Variable()
    alpha = 1.0 - lambda1 - lambda2
    w_tilde = cp.Variable((n, n))
    ones_n = np.ones((n,), dtype=float)
    residual = y_pool - (lambda1 * x0_pool + b_tilde * ones_n[None, :] + x_pool @ w_tilde.T)
    objective = cp.Minimize(cp.sum_squares(residual))
    constraints = [lambda2 >= eps, lambda1 + lambda2 <= 1.0 - eps, lambda1 <= 1.0]

    for i in range(n):
        ns = ref_neighbors[i]
        allowed = np.zeros((n,), dtype=float)
        allowed[np.asarray(ns, dtype=int)] = 1.0
        constraints.append(w_tilde[i, :] >= 0)
        constraints.append(cp.sum(w_tilde[i, :]) == alpha)
        constraints.append(cp.multiply(1.0 - allowed, w_tilde[i, :]) == 0)
    constraints += [b_tilde <= lambda2, b_tilde >= -lambda2]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    if lambda1.value is None or lambda2.value is None or w_tilde.value is None or b_tilde.value is None:
        raise RuntimeError("Joint FJ traj0 optimization failed to produce a solution.")
    lambda1_hat = float(lambda1.value)
    lambda2_hat = float(lambda2.value)
    alpha_hat = 1.0 - lambda1_hat - lambda2_hat
    b_tilde_hat = float(b_tilde.value)
    w_tilde_hat = np.asarray(w_tilde.value, dtype=float)
    if alpha_hat <= eps:
        raise RuntimeError(f"Estimated alpha too small for stable W recovery: alpha={alpha_hat}")

    w_hat = w_tilde_hat / alpha_hat
    b_hat = b_tilde_hat / lambda2_hat
    fitted_pool = lambda1_hat * x0_pool + b_tilde_hat * ones_n[None, :] + x_pool @ w_tilde_hat.T
    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))
    return {
        "lambda1": lambda1_hat,
        "lambda2": lambda2_hat,
        "alpha": alpha_hat,
        "b_tilde": b_tilde_hat,
        "W_tilde": w_tilde_hat,
        "W": w_hat,
        "b": float(b_hat),
        "X_pool": x_pool,
        "Y_pool": y_pool,
        "X0_pool": x0_pool,
        "mse_pool": mse_pool,
        "status": prob.status,
        "objective": float(prob.value) if prob.value is not None else np.nan,
    }
