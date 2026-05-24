"""Unit-test templates for fixed-graph homophily models."""

import unittest
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.models.fixed_graph.homophily import (
    fit_fg_fj_bias_homophily,
    rollout_fg_fj_bias_homophily,
    _build_supports,
    _build_row_stochastic_W,
    _split_row_params,
    _evaluate_pool,
    _stack_run_datasets,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _random_sparse(n, in_degree, rng):
    in_degree = max(1, min(in_degree, n - 1))
    return {i: rng.choice([j for j in range(n) if j != i], size=in_degree, replace=False).tolist() for i in range(n)}

def _row_normalize(W):
    W = np.asarray(W, dtype=float)
    s = W.sum(axis=1, keepdims=True)
    # raise exception if any negative entries or rows that sum to zero
    if np.any(W < -1e-10):
        raise ValueError("W has negative entries")
    if np.any(s < 1e-10):
        raise ValueError("W has rows that sum to zero")
    return W / np.where(s > 0, s, 1.0)

def _random_W(neighbors, n, rng):
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        cols = list(set(neighbors[i]) | {i})
        w = rng.exponential(size=len(cols))
        w /= w.sum()
        for c, v in zip(cols, w):
            W[i, c] = v
    return W



def _sim_fg_fj_bias_homophily(rng, W_true, gamma, lambda_h, lambda_i, lambda_b, bias, n_runs, horizon, noise_std=0.0):
    """x_{t+1} = lambda_h * homophily_step(x_t) + lambda_i * x0 + lambda_b * bias."""
    import math
    n = W_true.shape[0]
    run_traj = {}
    for r in range(n_runs):
        x0 = rng.uniform(-1.0, 1.0, size=n)
        x = x0.copy()
        states = [x0.copy()]
        for _ in range(horizon):
            hom_mat = np.zeros((n, n), dtype=float)
            x_flat = x.ravel()
            for i in range(n):
                for j in range(n):
                    hom_mat[i, j] = W_true[i, j] * math.exp(-gamma * abs(x_flat[i] - x_flat[j]))
            hom_mat = _row_normalize(hom_mat)
            x = lambda_h * (hom_mat @ x_flat) + lambda_i * x0 + lambda_b * bias
            if noise_std > 0:
                x += noise_std * rng.normal(size=n)
            states.append(x.copy())
        run_traj[f"run_{r:02d}"] = np.asarray(states, dtype=float)
    return run_traj


def _make_objective_fn(run_traj_map, run_neighbors):
    """Minimal recreation of the objective_fn used inside fit_fg_fj_bias_homophily.

    Returns a callable  f(theta) -> float  where theta is:
        [gamma, lambda_h, lambda_i, b_tilde, *row_params]
    with row_params laid out as concatenated per-row weight blocks
    ordered by _build_supports.
    """
    run_names = sorted(run_traj_map.keys())
    ref_neighbors = run_neighbors[run_names[0]]
    x_pool, y_pool, x0_pool = _stack_run_datasets(run_traj_map)
    _, n = x_pool.shape
    supports, support_sizes = _build_supports(ref_neighbors, n)

    def objective_fn(theta):
        theta = np.asarray(theta, dtype=float).ravel()
        gamma_val     = float(theta[0])
        lambda_h      = float(theta[1])
        lambda_i      = float(theta[2])
        b_tilde       = float(theta[3])
        row_params    = theta[4:]
        w_hat = _build_row_stochastic_W(_split_row_params(row_params, support_sizes), supports, n)
        homo_pool = _evaluate_pool(x_pool, w_hat, gamma_val)
        fitted = lambda_h * homo_pool + lambda_i * x0_pool + b_tilde
        return float(np.sum((y_pool - fitted) ** 2))

    return objective_fn


def _W_to_support_params(W, neighbors, n):
    """Extract per-row weight blocks from a full n×n W in the support order used by _build_supports.

    Returns the concatenated row-param vector as expected by _make_objective_fn / _build_row_stochastic_W.
    """
    supports, _ = _build_supports(neighbors, n)
    return np.concatenate([W[i, support] for i, support in enumerate(supports)])


def _perturb_support_params(W, neighbors, n, noise_scale, rng):
    """Perturb the support-restricted row parameters of W, clip to non-negative, re-normalise each row.

    Returns the perturbed row-param vector (same layout as _W_to_support_params).
    """
    supports, support_sizes = _build_supports(neighbors, n)
    row_params = np.concatenate([W[i, support] for i, support in enumerate(supports)])
    row_params = row_params + rng.uniform(-noise_scale, noise_scale, size=row_params.shape)
    row_params = np.maximum(row_params, 0.0)
    # re-normalise each row block independently
    blocks = []
    cursor = 0
    for size in support_sizes:
        block = row_params[cursor: cursor + size]
        s = float(block.sum())
        blocks.append(block / s if s > 1e-12 else np.full(size, 1.0 / size))
        cursor += size
    return np.concatenate(blocks)


def _project_scalar_params(gamma, lambda1, lambda2, b_tilde):
    """Project (gamma, lambda1, lambda2, b_tilde) onto the feasible region:
        gamma > 0
        lambda1, lambda2 >= 0
        b_tilde in [-1, 1]
        lambda1 + lambda2 + |b_tilde| <= 1
    """
    gamma   = max(float(gamma), 0.0)
    lambda1 = max(float(lambda1), 0.0)
    lambda2 = max(float(lambda2), 0.0)
    b_tilde = float(np.clip(b_tilde, -1.0, 1.0))
    excess = lambda1 + lambda2 + abs(b_tilde) - 1.0
    if excess > 0:
        scale = 1.0 / (lambda1 + lambda2 + abs(b_tilde))
        lambda1 *= scale
        lambda2 *= scale
        b_tilde *= scale
    return gamma, lambda1, lambda2, b_tilde


# ===========================================================================
# 1. fit_fg_fj_bias_homophily
# ===========================================================================

class TestFgFjBiasHomophily(unittest.TestCase):

    TOL_RANDOM_START = 0.05
    MSE_TOL = 0.01

    def test_fit_low_dimension(self):
        """Test that fit_fg_fj_bias_homophily recovers optimal parmameters on a low-dimensional noiseless dataset."""
        n = 2
        neighbors = {
            0: [0,1],
            1: [0,1],
        }
        W_true = _random_W(neighbors, n, np.random.default_rng(42))
        gamma_true = 0.5
        lambda_h_true = 0.6
        lambda_i_true = 0.3
        lambda_b_true = 0.1
        bias_true = 0.2
        n_runs = 10
        horizon = 10
        rng = np.random.default_rng(42)
        run_traj = _sim_fg_fj_bias_homophily(rng, W_true, gamma_true, lambda_h_true, lambda_i_true, lambda_b_true, bias_true, n_runs, horizon)
        run_neighbors = {f"run_{r:02d}": neighbors for r in range(n_runs)}
        fit_result = fit_fg_fj_bias_homophily(run_traj, run_neighbors, init_mode = "random", n_random_starts=5)
        self.assertIn("gamma", fit_result)
        self.assertIn("lambda1", fit_result)
        self.assertIn("lambda2", fit_result)
        self.assertIn("bias_tilde", fit_result)
        self.assertIn("bias", fit_result)
        self.assertIn("W", fit_result)

        # check that MSE is small
        mse = fit_result["mse_pool"]
        self.assertLess(mse, self.MSE_TOL, f"MSE {mse} is greater than tolerance {self.MSE_TOL}")

        true_param_vec = np.concatenate(([gamma_true, lambda_h_true, lambda_i_true, lambda_b_true * bias_true], W_true.ravel()))
        # repeat with initialization at the optimal aprameters
        fit_result_init_opt = fit_fg_fj_bias_homophily(run_traj, run_neighbors, fixed_start = true_param_vec)

        print("Test finished with MSE = ", mse)

        self.assertAlmostEqual(fit_result_init_opt["gamma"], gamma_true, delta=self.TOL_RANDOM_START)
        self.assertAlmostEqual(fit_result_init_opt["lambda1"], lambda_h_true, delta=self.TOL_RANDOM_START)
        self.assertAlmostEqual(fit_result_init_opt["lambda2"], lambda_i_true, delta=self.TOL_RANDOM_START)
        self.assertAlmostEqual(fit_result_init_opt["bias_tilde"], lambda_b_true * bias_true, delta=self.TOL_RANDOM_START)
        self.assertAlmostEqual(fit_result_init_opt["bias"], bias_true, delta=self.TOL_RANDOM_START)
        np.testing.assert_allclose(fit_result["W"], W_true, atol=self.TOL_RANDOM_START)

    INIT_PERT = 1e-03
    TOL_INIT = 1e-03

    def test_fit_with_optimal_init(self):

        """For moderately sized graphs, check that initializating at the optimal parameters leads to low MSE."""

        # Generate 5 random sparse graphs with n=10 and in-degree=3, and simulate noiseless trajectories with known parameters.
        # Check that fit_fg_fj_bias_homophily recovers parameters close to the true values when initialized at the optimal parameters.

        n = 10
        in_degree = 3
        n_runs = 5
        horizon = 15
        gamma_true = 0.4
        lambda_h_true = 0.5
        lambda_i_true = 0.3
        lambda_b_true = 0.2
        bias_true = 0.1
        rng = np.random.default_rng(42)

        for r in range(n_runs):
            neighbors = _random_sparse(n, in_degree, rng)
            W_true = _random_W(neighbors, n, rng)
            run_traj = _sim_fg_fj_bias_homophily(rng, W_true, gamma_true, lambda_h_true, lambda_i_true, lambda_b_true, bias_true, n_runs=n_runs, horizon=horizon)
            run_neighbors = {f"run_{r:02d}": neighbors for r in range(n_runs)}

            # construct optimal W by masking to neighbors + identity
            W_as_param =[]
            for i in range(n):
                row = np.zeros(len(neighbors[i])+1, dtype=float)
                idx_map = {j: idx for idx, j in enumerate(neighbors[i] + [i])}
                for j in neighbors[i]:
                    row[idx_map[j]] = W_true[i, j]
                row[idx_map[i]] = W_true[i, i]
                W_as_param.append(row)
            W_as_param = np.concatenate(W_as_param)

            true_param_vec = np.concatenate(([gamma_true, lambda_h_true, lambda_i_true, lambda_b_true * bias_true], W_as_param))
            fit_result = fit_fg_fj_bias_homophily(run_traj, run_neighbors, fixed_start=true_param_vec)
            mse = fit_result["mse_pool"]
            self.assertLess(mse, self.MSE_TOL, f"Run {r}: MSE {mse} is greater than tolerance {self.MSE_TOL}")
            self.assertAlmostEqual(fit_result["gamma"], gamma_true, delta=self.TOL_INIT)
            self.assertAlmostEqual(fit_result["lambda1"], lambda_h_true, delta=self.TOL_INIT)
            self.assertAlmostEqual(fit_result["lambda2"], lambda_i_true, delta=self.TOL_INIT)
            self.assertAlmostEqual(fit_result["bias_tilde"], lambda_b_true * bias_true, delta=self.TOL_INIT)
            self.assertAlmostEqual(fit_result["bias"], bias_true, delta=self.TOL_INIT)
            np.testing.assert_allclose(fit_result["W"], W_true, atol=self.TOL_INIT)
            print(f"Run {r} finished with MSE = ", mse)

            # repeat with the initial parameters perturbed slightly
            gamma_p = gamma_true + rng.uniform(-self.INIT_PERT, self.INIT_PERT)
            lambda_h_p = lambda_h_true + rng.uniform(-self.INIT_PERT, self.INIT_PERT)
            lambda_i_p = lambda_i_true + rng.uniform(-self.INIT_PERT, self.INIT_PERT)
            b_tilde_p = lambda_b_true * bias_true + rng.uniform(-self.INIT_PERT, self.INIT_PERT)
            gamma_p, lambda_h_p, lambda_i_p, b_tilde_p = _project_scalar_params(gamma_p, lambda_h_p, lambda_i_p, b_tilde_p)
            row_params_p = _perturb_support_params(W_true, neighbors, n, self.INIT_PERT, rng)
            perturbed_param_vec = np.concatenate(([gamma_p, lambda_h_p, lambda_i_p, b_tilde_p], row_params_p))
            fit_result_perturbed = fit_fg_fj_bias_homophily(run_traj, run_neighbors, fixed_start=perturbed_param_vec)
            mse_perturbed = fit_result_perturbed["mse_pool"]
            self.assertLess(mse_perturbed, self.MSE_TOL, f"Run {r} with perturbed init: MSE {mse_perturbed} is greater than tolerance {self.MSE_TOL}")
            print(f"Run {r} with perturbed init finished with MSE = ", mse_perturbed)
            self.assertAlmostEqual(fit_result_perturbed["gamma"], gamma_true, delta=self.TOL_INIT)
            self.assertAlmostEqual(fit_result_perturbed["lambda1"], lambda_h_true, delta=self.TOL_INIT)
            self.assertAlmostEqual(fit_result_perturbed["lambda2"], lambda_i_true, delta=self.TOL_INIT)
            self.assertAlmostEqual(fit_result_perturbed["bias_tilde"], lambda_b_true * bias_true, delta=self.TOL_INIT)
            self.assertAlmostEqual(fit_result_perturbed["bias"], bias_true, delta=self.TOL_INIT)
            np.testing.assert_allclose(fit_result_perturbed["W"], W_true, atol=self.TOL_INIT)
        

        # repeat for one case of n=30, in-degree=5
        n = 30
        in_degree = 5
        neighbors = _random_sparse(n, in_degree, rng)
        W_true = _random_W(neighbors, n, rng)
        run_traj = _sim_fg_fj_bias_homophily(rng, W_true, gamma_true, lambda_h_true, lambda_i_true, lambda_b_true, bias_true, n_runs=n_runs, horizon=horizon)
        run_neighbors = {f"run_{r:02d}": neighbors for r in range(n_runs)}
        W_as_param =[]
        for i in range(n):
            row = np.zeros(len(neighbors[i])+1, dtype=float)
            idx_map = {j: idx for idx, j in enumerate(neighbors[i] + [i])}
            for j in neighbors[i]:
                row[idx_map[j]] = W_true[i, j]
            row[idx_map[i]] = W_true[i, i]
            W_as_param.append(row)
        W_as_param = np.concatenate(W_as_param)
        true_param_vec = np.concatenate(([gamma_true, lambda_h_true, lambda_i_true, lambda_b_true * bias_true], W_as_param))
        fit_result = fit_fg_fj_bias_homophily(run_traj, run_neighbors, fixed_start=true_param_vec)
        mse = fit_result["mse_pool"]
        self.assertLess(mse, self.MSE_TOL, f"MSE {mse} is greater than tolerance {self.MSE_TOL}")
        self.assertAlmostEqual(fit_result["gamma"], gamma_true, delta=self.TOL_RANDOM_START)
        self.assertAlmostEqual(fit_result["lambda1"], lambda_h_true, delta=self.TOL_RANDOM_START)
        self.assertAlmostEqual(fit_result["lambda2"], lambda_i_true, delta=self.TOL_RANDOM_START)
        self.assertAlmostEqual(fit_result["bias_tilde"], lambda_b_true * bias_true, delta=self.TOL_RANDOM_START)
        self.assertAlmostEqual(fit_result["bias"], bias_true, delta=self.TOL_RANDOM_START)
        np.testing.assert_allclose(fit_result["W"], W_true, atol=self.TOL_RANDOM_START)
        print(f"Large n run finished with MSE = ", mse)


    PERT_NOISE = 1e-3
    PERT_TOL = 1e-3
    NUM_PERTS = 100

    def test_local_optimality(self):

        # Generate 5 random sparse graphs with n=10 and in-degree=3, and simulate noiseless trajectories with known parameters.
        # Check that fit_fg_fj_bias_homophily recovers parameters that are locally optimal

        n = 10
        in_degree = 2
        n_runs = 3
        horizon = 10
        gamma_true = 0.4
        lambda_h_true = 0.5
        lambda_i_true = 0.3
        lambda_b_true = 0.2
        bias_true = 0.1
        rng = np.random.default_rng(42)

        for r in range(n_runs):
            neighbors = _random_sparse(n, in_degree, rng)
            W_true = _random_W(neighbors, n, rng)
            run_traj = _sim_fg_fj_bias_homophily(rng, W_true, gamma_true, lambda_h_true, lambda_i_true, lambda_b_true, bias_true, n_runs=n_runs, horizon=horizon)
            run_neighbors = {f"run_{r:02d}": neighbors for r in range(n_runs)}

            obj = _make_objective_fn(run_traj, run_neighbors)
            fit_result = fit_fg_fj_bias_homophily(run_traj, run_neighbors, init_mode="random", n_random_starts=1)
            # perturb the fitted parameters and check that MSE increases
            mse_fitted = fit_result["mse_pool"]
            for _ in range(self.NUM_PERTS):
                gamma_p     = fit_result["gamma"]      + rng.uniform(-self.PERT_NOISE, self.PERT_NOISE)
                lambda1_p   = fit_result["lambda1"]    + rng.uniform(-self.PERT_NOISE, self.PERT_NOISE)
                lambda2_p   = fit_result["lambda2"]    + rng.uniform(-self.PERT_NOISE, self.PERT_NOISE)
                b_tilde_p   = fit_result["bias_tilde"] + rng.uniform(-self.PERT_NOISE, self.PERT_NOISE)
                gamma_p, lambda1_p, lambda2_p, b_tilde_p = _project_scalar_params(gamma_p, lambda1_p, lambda2_p, b_tilde_p)
                row_params_p = _perturb_support_params(fit_result["W"], neighbors, n, self.PERT_NOISE, rng)
                perturbed_param_vec = np.concatenate(
                    ([gamma_p, lambda1_p, lambda2_p, b_tilde_p], row_params_p)
                )
                mse_perturbed = obj(perturbed_param_vec)
                self.assertGreater(mse_perturbed, mse_fitted - self.PERT_TOL, f"Perturbed MSE {mse_perturbed} is not greater than fitted MSE {mse_fitted} within tolerance {self.PERT_TOL}")


        # repeat for scalar parameter initialization
        neighbors = _random_sparse(n, in_degree, rng)
        W_true = _random_W(neighbors, n, rng)
        run_traj = _sim_fg_fj_bias_homophily(rng, W_true, gamma_true, lambda_h_true, lambda_i_true, lambda_b_true, bias_true, n_runs=n_runs, horizon=horizon)
        run_neighbors = {f"run_{r:02d}": neighbors for r in range(n_runs)}
        fit_result = fit_fg_fj_bias_homophily(run_traj, run_neighbors,  init_mode="uniform")
        mse_fitted = fit_result["mse_pool"]
        for _ in range(self.NUM_PERTS):
            gamma_p     = fit_result["gamma"]      + rng.uniform(-self.PERT_NOISE, self.PERT_NOISE)
            lambda1_p   = fit_result["lambda1"]    + rng.uniform(-self.PERT_NOISE, self.PERT_NOISE)
            lambda2_p   = fit_result["lambda2"]    + rng.uniform(-self.PERT_NOISE, self.PERT_NOISE)
            b_tilde_p   = fit_result["bias_tilde"] + rng.uniform(-self.PERT_NOISE, self.PERT_NOISE)
            gamma_p, lambda1_p, lambda2_p, b_tilde_p = _project_scalar_params(gamma_p, lambda1_p, lambda2_p, b_tilde_p)
            row_params_p = _perturb_support_params(fit_result["W"], neighbors, n, self.PERT_NOISE, rng)
            perturbed_param_vec = np.concatenate(
                ([gamma_p, lambda1_p, lambda2_p, b_tilde_p], row_params_p)
            )
            mse_perturbed = obj(perturbed_param_vec)
            self.assertGreater(mse_perturbed, mse_fitted - self.PERT_TOL, f"Perturbed MSE {mse_perturbed} is not greater than fitted MSE {mse_fitted} within tolerance {self.PERT_TOL}")

            print(f"Run {r} finished with MSE = ", mse_fitted)

# ===========================================================================
# Rollout consistency
# ===========================================================================

class TestRollouts(unittest.TestCase):

    def test_rollout_fg_fj_bias_homophily_single_step(self):
        # TODO

        # fix lambda_h, lambda_i, lambda_b, bias
        lambda_h = 0.5
        lambda_i = 0.3
        lambda_b = 0.2
        bias = 0.5
        gamma = 0.4

        # construct a random W
        n = 10
        in_degree = 3
        rng = np.random.default_rng(42)
        neighbors = _random_sparse(n, in_degree, rng)
        W = _random_W(neighbors, n, rng)

        x0  = rng.uniform(-1.0, 1.0, size=n)

        predictions = rollout_fg_fj_bias_homophily(W, gamma=gamma, x0=x0, horizon=1, lambda1=lambda_h, lambda2=lambda_i, bias_tilde=bias*lambda_b)

        # check shape
        self.assertEqual(predictions.shape, (2, n))

        # check that the first state is x0
        np.testing.assert_allclose(predictions[0], x0, atol=1e-10)

        # compute the expected next state manually
        hom_mat = np.zeros((n, n), dtype=float)
        x_flat = x0.ravel()
        for i in range(n):
            for j in range(n):
                hom_mat[i, j] = W[i, j] * np.exp(-gamma * abs(x_flat[i] - x_flat[j]))
        hom_mat = _row_normalize(hom_mat)
        expected_next = lambda_h * (hom_mat @ x_flat) + lambda_i * x0 + lambda_b * bias
        np.testing.assert_allclose(predictions[1], expected_next, atol=1e-10)        


    INIT_PERT = 1e-3

    def test_rollout_matches_simulation(self):
        """One-step rollout should match _sim_fg_fj_bias_homophily on noiseless data."""
        
        # simulate a trajectory with known parameters
        n = 10
        in_degree = 3
        n_runs = 5
        horizon = 10
        gamma_true = 0.4
        lambda_h_true = 0.5
        lambda_i_true = 0.3
        lambda_b_true = 0.2
        bias_true = 0.3
        rng = np.random.default_rng(42)
        neighbors = _random_sparse(n, in_degree, rng)
        W_true = _random_W(neighbors, n, rng)
        run_traj = _sim_fg_fj_bias_homophily(rng, W_true, gamma_true, lambda_h_true, lambda_i_true, lambda_b_true, bias_true, n_runs=n_runs, horizon=horizon)
        run_neighbors = {f"run_{r:02d}": neighbors for r in range(n_runs)}

        # fit from trajectory with intialization close to the true parameters
        gamma_p = gamma_true + rng.uniform(-self.INIT_PERT, self.INIT_PERT)
        lambda_h_p = lambda_h_true + rng.uniform(-self.INIT_PERT, self.INIT_PERT)
        lambda_i_p = lambda_i_true + rng.uniform(-self.INIT_PERT, self.INIT_PERT)
        b_tilde_p = lambda_b_true * bias_true + rng.uniform(-self.INIT_PERT, self.INIT_PERT)
        gamma_p, lambda_h_p, lambda_i_p, b_tilde_p = _project_scalar_params(gamma_p, lambda_h_p, lambda_i_p, b_tilde_p)
        row_params_p = _perturb_support_params(W_true, neighbors, n, self.INIT_PERT, rng)
        perturbed_param_vec = np.concatenate(([gamma_p, lambda_h_p, lambda_i_p, b_tilde_p], row_params_p))
        fit_result = fit_fg_fj_bias_homophily(run_traj, run_neighbors, fixed_start=perturbed_param_vec)

        # check that rollout from the fitted parameters matches the original trajectory closely
        # get initial state and run from first run
        x0 = run_traj["run_00"][0]
        rollout_pred = rollout_fg_fj_bias_homophily(fit_result["W"], gamma=fit_result["gamma"], x0=x0, horizon=horizon, 
                                                    lambda1=fit_result["lambda1"], lambda2=fit_result["lambda2"], bias_tilde=fit_result["bias_tilde"])
        np.testing.assert_allclose(rollout_pred, run_traj["run_00"], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
