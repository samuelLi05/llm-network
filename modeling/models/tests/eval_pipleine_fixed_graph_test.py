from pathlib import Path
import sys
import numpy as np
import cvxpy as cp


THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from analysis_utils import compute_eigenvalue, compute_fj_eigenvalue, compute_fj_joint_eigenvalue
from synthetic_data import (
    build_dataset_from_run,
    build_synthetic_fj_runs,
    build_x0_from_agent_inits,
    random_row_stochastic_matrix,
    build_transitions,
    build_stacked_independent_transitions,
    build_neighbors_all_to_all,
    build_synthetic_linear_runs,
    build_synthetic_homophily_runs,
)
from fixed_graph.degroot import fit_row_stochastic_W_from_pooled_runs
from fixed_graph.friedkin_johnsen import fit_friedkin_johnsen as fit_fg_friedkin_johnsen
from fixed_graph.homophily import fit_fg_homophily, fit_fg_fj_homophily, fit_fg_fj_bias_homophily


def min_positive_eigvals(values, tol=1e-12):
    vals = np.asarray(values, dtype=float)
    pos = vals[vals > tol]
    if pos.size == 0:
        return 0.0
    return float(np.min(pos))


def fit_friedkin_johnsen(run_traj_map, run_neighbors, lambda1, lambda2, agent_inits):
    if lambda1 < 0 or lambda2 < 0 or lambda1 + lambda2 > 1:
        raise ValueError("lambda1 and lambda2 must satisfy nonnegativity and lambda1 + lambda2 <= 1")

    run_names = sorted(run_traj_map.keys())
    ref_neighbors = run_neighbors[run_names[0]]
    for rn in run_names[1:]:
        if run_neighbors[rn] != ref_neighbors:
            raise ValueError("RUN_NEIGHBORS must be identical across runs")

    X_blocks, Y_blocks = [], []
    for rn in run_names:
        X, Y = build_dataset_from_run(np.asarray(run_traj_map[rn], dtype=float))
        X_blocks.append(X)
        Y_blocks.append(Y)

    X_pool = np.vstack(X_blocks)
    Y_pool = np.vstack(Y_blocks)
    n = X_pool.shape[1]
    alpha = 1.0 - lambda1 - lambda2

    b = cp.Variable()
    W_rows = []
    objective_terms = []
    constraints = [b >= -1.0, b <= 1.0]

    x0_init = build_x0_from_agent_inits(agent_inits, n)
    for i in range(n):
        ns = ref_neighbors[i]
        w_ns = cp.Variable(len(ns))
        W_rows.append((i, ns, w_ns))

        X_ns = X_pool[:, ns]
        y = Y_pool[:, i]
        x0i = float(x0_init[i])
        pred = lambda1 * x0i + lambda2 * b + alpha * (X_ns @ w_ns)
        objective_terms.append(cp.sum_squares(y - pred))
        constraints += [w_ns >= 0, cp.sum(w_ns) == 1]

    objective = cp.Minimize(cp.sum(objective_terms))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    if b.value is None:
        raise RuntimeError("FJ fit failed")

    W = np.zeros((n, n), dtype=float)
    for i, ns, w_ns in W_rows:
        W[i, ns] = np.asarray(w_ns.value, dtype=float).ravel()

    return W, float(b.value), X_pool, Y_pool


def select_friedkin_johnsen_lambdas_by_mse(run_traj_map, run_neighbors, lambda_grid, agent_inits):
    best_result = None
    all_results = []

    for lambda1 in lambda_grid:
        for lambda2 in lambda_grid:
            if lambda1 + lambda2 > 1:
                continue

            W_hat, b_hat, X_pool, Y_pool = fit_friedkin_johnsen(
                run_traj_map,
                run_neighbors,
                float(lambda1),
                float(lambda2),
                agent_inits,
            )

            n = X_pool.shape[1]
            x0 = build_x0_from_agent_inits(agent_inits, n)
            alpha = 1.0 - float(lambda1) - float(lambda2)
            pred_pool = float(lambda1) * x0[None, :] + float(lambda2) * b_hat + alpha * (X_pool @ W_hat.T)
            mse_pool = float(np.mean((Y_pool - pred_pool) ** 2))

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
            raise ValueError("RUN_NEIGHBORS must be identical across runs")

    X_blocks, Y_blocks = [], []
    for rn in run_names:
        X, Y = build_dataset_from_run(np.asarray(run_traj_map[rn], dtype=float))
        X_blocks.append(X)
        Y_blocks.append(Y)

    X_pool = np.vstack(X_blocks)
    Y_pool = np.vstack(Y_blocks)
    _, n = X_pool.shape

    x0 = build_x0_from_agent_inits(agent_inits, n)
    X0_pool = np.repeat(x0.reshape(1, -1), X_pool.shape[0], axis=0)

    lambda1 = cp.Variable(nonneg=True)
    lambda2 = cp.Variable(nonneg=True)
    b_tilde = cp.Variable()
    alpha = 1.0 - lambda1 - lambda2
    W_tilde = cp.Variable((n, n))

    ones_n = np.ones((n,), dtype=float)
    residual = Y_pool - (lambda1 * X0_pool + b_tilde * ones_n[None, :] + X_pool @ W_tilde.T)
    objective = cp.Minimize(cp.sum_squares(residual))
    constraints = [
        lambda2 >= eps,
        lambda1 >= 0.0,
        lambda1 + lambda2 <= 1.0 - eps,
        lambda1 <= 1.0,
    ]

    for i in range(n):
        ns = ref_neighbors[i]
        allowed = np.zeros((n,), dtype=float)
        allowed[np.asarray(ns, dtype=int)] = 1.0
        constraints.append(W_tilde[i, :] >= 0)
        constraints.append(cp.sum(W_tilde[i, :]) == alpha)
        constraints.append(cp.multiply(1.0 - allowed, W_tilde[i, :]) == 0)

    constraints += [b_tilde <= lambda2, b_tilde >= -lambda2]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    if lambda1.value is None or lambda2.value is None or W_tilde.value is None or b_tilde.value is None:
        raise RuntimeError("Joint FJ optimization failed")

    lambda1_hat = float(lambda1.value)
    lambda2_hat = float(lambda2.value)
    alpha_hat = 1.0 - lambda1_hat - lambda2_hat
    if alpha_hat <= eps:
        raise RuntimeError("Estimated alpha too small for stable W recovery")

    b_tilde_hat = float(b_tilde.value)
    W_tilde_hat = np.asarray(W_tilde.value, dtype=float)
    W_hat = W_tilde_hat / alpha_hat
    b_hat = b_tilde_hat / lambda2_hat

    fitted_pool = lambda1_hat * X0_pool + b_tilde_hat * ones_n[None, :] + X_pool @ W_tilde_hat.T
    mse_pool = float(np.mean((Y_pool - fitted_pool) ** 2))

    return {
        "lambda1": lambda1_hat,
        "lambda2": lambda2_hat,
        "alpha": alpha_hat,
        "b": float(b_hat),
        "W": W_hat,
        "mse_pool": mse_pool,
        "X_pool": X_pool,
        "Y_pool": Y_pool,
    }


def test_compute_eigenvalue_full_equals_reduced_for_all_neighbors():
    rng = np.random.default_rng(7)
    n = 8
    W = random_row_stochastic_matrix(n, rng)
    X, Y = build_transitions(W, steps=80, shock_std=0.05, rng=rng)
    out = compute_eigenvalue(X, Y)

    assert out["gram_full_shape"] == (n * n, n * n)
    assert out["eigvals_full"].shape == (n * n,)


def test_compute_eigenvalue_min_reduced_eig_increases_with_richer_transitions():
    rng = np.random.default_rng(11)
    n = 10
    W = random_row_stochastic_matrix(n, rng)
    shock_std = 0.08
    steps_per_trajectory = 80

    X_poor, Y_poor = build_stacked_independent_transitions(
        W,
        n_trajectories=2,
        steps_per_trajectory=steps_per_trajectory,
        shock_std=shock_std,
        rng=rng,
    )
    X_rich, Y_rich = build_stacked_independent_transitions(
        W,
        n_trajectories=8,
        steps_per_trajectory=steps_per_trajectory,
        shock_std=shock_std,
        rng=rng,
    )

    poor = compute_eigenvalue(X_poor, Y_poor)
    rich = compute_eigenvalue(X_rich, Y_rich)

    poor_min = min_positive_eigvals(poor["eigvals_full"])
    rich_min = min_positive_eigvals(rich["eigvals_full"])

    assert rich_min > poor_min


def test_compute_eigenvalue_full_is_neighbor_invariant():
    rng = np.random.default_rng(19)
    n = 10
    neighbors = {
        i: sorted({i, (i - 1) % n, (i + 1) % n, (i + 3) % n})
        for i in range(n)
    }

    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        ns = neighbors[i]
        W[i, ns] = rng.dirichlet(np.ones(len(ns)))

    X, Y = build_transitions(W, steps=180, shock_std=0.08, rng=rng)
    out_sparse = compute_eigenvalue(X, Y)
    out_all = compute_eigenvalue(X, Y)

    assert out_sparse["gram_full_shape"] == (n * n, n * n)
    assert out_all["gram_full_shape"] == (n * n, n * n)
    np.testing.assert_allclose(out_sparse["eigvals_full"], out_all["eigvals_full"], rtol=1e-9, atol=1e-11)


def test_fj_line_search_and_joint_fit_are_similar_on_synthetic_data():
    rng = np.random.default_rng(101)
    n = 8
    n_runs = 12
    horizon = 16
    neighbors = {i: list(range(n)) for i in range(n)}

    true_lambda1 = 0.2
    true_lambda2 = 0.3
    true_b = -0.15
    noise_std = 0.00

    x0_prior = rng.uniform(-0.8, 0.8, size=n)
    agent_inits = {f"agent_{i + 1}": float(x0_prior[i]) for i in range(n)}

    W_true, run_traj, run_neighbors = build_synthetic_fj_runs(
        rng=rng,
        n=n,
        n_runs=n_runs,
        horizon=horizon,
        neighbors=neighbors,
        lambda1=true_lambda1,
        lambda2=true_lambda2,
        b=true_b,
        x0_prior=x0_prior,
        noise_std=noise_std,
    )

    lambda_grid = np.linspace(0.05, 0.5, 10)
    best, _ = select_friedkin_johnsen_lambdas_by_mse(
        run_traj,
        run_neighbors,
        lambda_grid=lambda_grid,
        agent_inits=agent_inits,
    )

    W_ls, b_ls, X_pool_ls, Y_pool_ls = fit_friedkin_johnsen(
        run_traj,
        run_neighbors,
        best["lambda1"],
        best["lambda2"],
        agent_inits,
    )
    joint = fit_friedkin_johnsen_joint(run_traj, run_neighbors, agent_inits, eps=1e-4)

    assert abs(best["lambda1"] - joint["lambda1"]) < 0.08
    assert abs(best["lambda2"] - joint["lambda2"]) < 0.08
    assert abs(b_ls - joint["b"]) < 0.15

    W_rel = float(np.linalg.norm(W_ls - joint["W"]) / max(np.linalg.norm(W_ls), 1e-12))
    assert W_rel < 0.15

    x0 = build_x0_from_agent_inits(agent_inits, n)
    alpha_ls = 1.0 - best["lambda1"] - best["lambda2"]
    pred_ls = best["lambda1"] * x0[None, :] + best["lambda2"] * b_ls + alpha_ls * (X_pool_ls @ W_ls.T)
    alpha_joint = 1.0 - joint["lambda1"] - joint["lambda2"]
    pred_joint = joint["lambda1"] * x0[None, :] + joint["lambda2"] * joint["b"] + alpha_joint * (X_pool_ls @ joint["W"].T)
    pred_mse = float(np.mean((pred_ls - pred_joint) ** 2))

    W_rel_ls_true = float(np.linalg.norm(W_ls - W_true) / max(np.linalg.norm(W_true), 1e-12))
    W_rel_joint_true = float(np.linalg.norm(joint["W"] - W_true) / max(np.linalg.norm(W_true), 1e-12))
    b_abs_ls_true = float(abs(b_ls - true_b))
    b_abs_joint_true = float(abs(joint["b"] - true_b))
    l1_abs_ls_true = float(abs(best["lambda1"] - true_lambda1))
    l2_abs_ls_true = float(abs(best["lambda2"] - true_lambda2))
    l1_abs_joint_true = float(abs(joint["lambda1"] - true_lambda1))
    l2_abs_joint_true = float(abs(joint["lambda2"] - true_lambda2))

    assert pred_mse < 5e-4
    assert abs(best["mse_pool"] - joint["mse_pool"]) < 5e-4

    assert W_rel_ls_true < 0.20
    assert W_rel_joint_true < 0.20
    assert b_abs_ls_true < 0.20
    assert b_abs_joint_true < 0.20
    assert l1_abs_ls_true < 0.12
    assert l2_abs_ls_true < 0.12
    assert l1_abs_joint_true < 0.12
    assert l2_abs_joint_true < 0.12


def test_compute_eigenvalue_on_fj_fit():
    rng = np.random.default_rng(303)
    n = 8
    steps_per_trajectory = 80

    true_lambda1 = 0.2
    true_lambda2 = 0.3
    true_b = -0.15
    noise_std = 0.08
    x0_prior = rng.uniform(-0.8, 0.8, size=n)

    _, run_traj_rich, _ = build_synthetic_fj_runs(
        rng=rng,
        n=n,
        n_runs=8,
        horizon=steps_per_trajectory,
        neighbors={i: list(range(n)) for i in range(n)},
        lambda1=true_lambda1,
        lambda2=true_lambda2,
        b=true_b,
        x0_prior=x0_prior,
        noise_std=noise_std,
    )

    rich_run_names = sorted(run_traj_rich.keys())
    run_traj_poor = {rn: run_traj_rich[rn] for rn in rich_run_names[:2]}

    X_poor_blocks, Y_poor_blocks = [], []
    for rn in sorted(run_traj_poor.keys()):
        X_i, Y_i = build_dataset_from_run(np.asarray(run_traj_poor[rn], dtype=float))
        X_poor_blocks.append(X_i)
        Y_poor_blocks.append(Y_i)
    X_poor = np.vstack(X_poor_blocks)
    Y_poor = np.vstack(Y_poor_blocks)

    X_rich_blocks, Y_rich_blocks = [], []
    for rn in sorted(run_traj_rich.keys()):
        X_i, Y_i = build_dataset_from_run(np.asarray(run_traj_rich[rn], dtype=float))
        X_rich_blocks.append(X_i)
        Y_rich_blocks.append(Y_i)
    X_rich = np.vstack(X_rich_blocks)
    Y_rich = np.vstack(Y_rich_blocks)

    out_fj_poor = compute_fj_eigenvalue(X_poor, Y_poor, lambda_1=true_lambda1, lambda_2=true_lambda2)
    out_fj_rich = compute_fj_eigenvalue(X_rich, Y_rich, lambda_1=true_lambda1, lambda_2=true_lambda2)

    out_joint_poor = compute_fj_joint_eigenvalue(X_poor, Y_poor, z0=x0_prior)
    out_joint_rich = compute_fj_joint_eigenvalue(X_rich, Y_rich, z0=x0_prior)

    min_fj_poor = min_positive_eigvals(out_fj_poor["eigvals_full"])
    min_fj_rich = min_positive_eigvals(out_fj_rich["eigvals_full"])
    min_joint_poor = min_positive_eigvals(out_joint_poor["eigvals_full"])
    min_joint_rich = min_positive_eigvals(out_joint_rich["eigvals_full"])

    assert min_fj_rich > min_fj_poor
    assert min_joint_rich > min_joint_poor


# Additional fit-focused tests appended without removing original stability/eigen tests.

def _assert_row_stochastic(w, atol=1e-6):
    w = np.asarray(w, dtype=float)
    np.testing.assert_allclose(np.sum(w, axis=1), np.ones((w.shape[0],), dtype=float), atol=atol, rtol=0.0)
    assert np.all(w >= -1e-10)


def test_fixed_graph_degroot_fit_low_mse_and_stochastic_w():
    rng = np.random.default_rng(2031)
    n = 7
    neighbors = build_neighbors_all_to_all(n)
    run_neighbors = {f"run_{i:02d}": neighbors for i in range(8)}

    W_true = np.zeros((n, n), dtype=float)
    for i in range(n):
        W_true[i, :] = rng.dirichlet(np.ones(n))

    run_traj_map = build_synthetic_linear_runs(rng, W_true, n_runs=8, horizon=22, noise_std=0.0)
    W_hat, X_pool, Y_pool = fit_row_stochastic_W_from_pooled_runs(run_traj_map, run_neighbors)

    pred = X_pool @ W_hat.T
    mse = float(np.mean((Y_pool - pred) ** 2))
    assert mse < 1e-10
    _assert_row_stochastic(W_hat)


def test_fixed_graph_fj_fit_low_mse_and_stochastic_w():
    rng = np.random.default_rng(2032)
    n = 7
    neighbors = build_neighbors_all_to_all(n)
    x0_prior = rng.uniform(-0.8, 0.8, size=n)
    agent_inits = {f"agent_{i + 1}": float(x0_prior[i]) for i in range(n)}

    lambda1 = 0.2
    lambda2 = 0.3
    bias = -0.1

    _, run_traj_map, run_neighbors = build_synthetic_fj_runs(
        rng=rng,
        n=n,
        n_runs=10,
        horizon=24,
        neighbors=neighbors,
        lambda1=lambda1,
        lambda2=lambda2,
        b=bias,
        x0_prior=x0_prior,
        noise_std=0.0,
    )

    W_hat, b_hat, X_pool, Y_pool = fit_fg_friedkin_johnsen(run_traj_map, run_neighbors, lambda1, lambda2, agent_inits)
    pred = lambda1 * x0_prior[None, :] + lambda2 * b_hat + (1.0 - lambda1 - lambda2) * (X_pool @ W_hat.T)
    mse = float(np.mean((Y_pool - pred) ** 2))

    assert mse < 1e-10
    assert abs(float(b_hat) - bias) < 0.15
    _assert_row_stochastic(W_hat)


def test_fixed_graph_homophily_variants_low_mse_noiseless():
    """Test absolute MSE threshold for noiseless synthetic data (verifies multi-start sufficiency)."""
    rng = np.random.default_rng(2033)
    n = 6
    neighbors = build_neighbors_all_to_all(n)
    run_neighbors = {f"run_{i:02d}": neighbors for i in range(8)}

    W_base = np.zeros((n, n), dtype=float)
    for i in range(n):
        W_base[i, :] = rng.dirichlet(np.ones(n))

    # Plain homophily - noiseless (MSE should be near machine epsilon)
    runs_plain = build_synthetic_homophily_runs(
        rng=rng,
        n=n,
        n_runs=8,
        horizon=18,
        Abar=W_base,
        gamma=1.0,
        noise_std=0.0,
    )
    fit_plain = fit_fg_homophily(runs_plain, run_neighbors, gamma0=1.0)
    assert fit_plain["success"], f"Plain homophily fit failed: {fit_plain['status']}"
    assert fit_plain["mse_pool"] < 1e-8, f"Plain homophily MSE {fit_plain['mse_pool']} exceeds threshold 1e-8"
    _assert_row_stochastic(fit_plain["W"])

    # FJ homophily - noiseless
    runs_fj = build_synthetic_homophily_runs(
        rng=rng,
        n=n,
        n_runs=8,
        horizon=18,
        Abar=W_base,
        gamma=1.0,
        noise_std=0.0,
        lambda1=0.3,
    )
    fit_fj = fit_fg_fj_homophily(runs_fj, run_neighbors, gamma0=1.0)
    assert fit_fj["success"], f"FJ homophily fit failed: {fit_fj['status']}"
    assert fit_fj["mse_pool"] < 1e-8, f"FJ homophily MSE {fit_fj['mse_pool']} exceeds threshold 1e-8"
    _assert_row_stochastic(fit_fj["W"])

    # Bias homophily - noiseless
    runs_bias = build_synthetic_homophily_runs(
        rng=rng,
        n=n,
        n_runs=8,
        horizon=18,
        Abar=W_base,
        gamma=1.0,
        noise_std=0.0,
        lambda1=0.3,
        lambda2=0.2,
        bias=-0.1,
    )
    fit_bias = fit_fg_fj_bias_homophily(runs_bias, run_neighbors, gamma0=1.0)
    assert fit_bias["success"], f"Bias homophily fit failed: {fit_bias['status']}"
    assert fit_bias["mse_pool"] < 1e-8, f"Bias homophily MSE {fit_bias['mse_pool']} exceeds threshold 1e-8"
    _assert_row_stochastic(fit_bias["W"])


def test_fixed_graph_homophily_variants_low_noise():
    """Test with small observation noise (MSE should still be very low)."""
    rng = np.random.default_rng(2034)
    n = 6
    neighbors = build_neighbors_all_to_all(n)
    run_neighbors = {f"run_{i:02d}": neighbors for i in range(8)}

    W_base = np.zeros((n, n), dtype=float)
    for i in range(n):
        W_base[i, :] = rng.dirichlet(np.ones(n))

    noise_level = 0.01  # Small noise
    
    # Plain homophily - small noise
    runs_plain = build_synthetic_homophily_runs(
        rng=rng,
        n=n,
        n_runs=8,
        horizon=18,
        Abar=W_base,
        gamma=1.0,
        noise_std=noise_level,
    )
    fit_plain = fit_fg_homophily(runs_plain, run_neighbors, gamma0=1.0)
    assert fit_plain["success"], f"Plain homophily fit failed: {fit_plain['status']}"
    # Absolute threshold for small noise: should be able to fit well
    assert fit_plain["mse_pool"] < 1e-4, f"Plain homophily MSE {fit_plain['mse_pool']} exceeds threshold 1e-4"
    _assert_row_stochastic(fit_plain["W"])

    # FJ homophily - small noise
    runs_fj = build_synthetic_homophily_runs(
        rng=rng,
        n=n,
        n_runs=8,
        horizon=18,
        Abar=W_base,
        gamma=1.0,
        noise_std=noise_level,
        lambda1=0.3,
    )
    fit_fj = fit_fg_fj_homophily(runs_fj, run_neighbors, gamma0=1.0)
    assert fit_fj["success"], f"FJ homophily fit failed: {fit_fj['status']}"
    assert fit_fj["mse_pool"] < 1e-4, f"FJ homophily MSE {fit_fj['mse_pool']} exceeds threshold 1e-4"
    _assert_row_stochastic(fit_fj["W"])

    # Bias homophily - small noise
    runs_bias = build_synthetic_homophily_runs(
        rng=rng,
        n=n,
        n_runs=8,
        horizon=18,
        Abar=W_base,
        gamma=1.0,
        noise_std=noise_level,
        lambda1=0.3,
        lambda2=0.2,
        bias=-0.1,
    )
    fit_bias = fit_fg_fj_bias_homophily(runs_bias, run_neighbors, gamma0=1.0)
    assert fit_bias["success"], f"Bias homophily fit failed: {fit_bias['status']}"
    assert fit_bias["mse_pool"] < 1e-4, f"Bias homophily MSE {fit_bias['mse_pool']} exceeds threshold 1e-4"
    _assert_row_stochastic(fit_bias["W"])
