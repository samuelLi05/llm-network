"""Unit-test templates for fixed-graph homophily models."""

import unittest
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.models.fixed_graph.homophily import (
    _kernel_homophily_step,
    fit_fg_fj_bias_homophily,
    rollout_fg_fj_bias_homophily,
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
            self.assertAlmostEqual(fit_result["gamma"], gamma_true, delta=self.TOL_RANDOM_START)
            self.assertAlmostEqual(fit_result["lambda1"], lambda_h_true, delta=self.TOL_RANDOM_START)
            self.assertAlmostEqual(fit_result["lambda2"], lambda_i_true, delta=self.TOL_RANDOM_START)
            self.assertAlmostEqual(fit_result["bias_tilde"], lambda_b_true * bias_true, delta=self.TOL_RANDOM_START)
            self.assertAlmostEqual(fit_result["bias"], bias_true, delta=self.TOL_RANDOM_START)
            np.testing.assert_allclose(fit_result["W"], W_true, atol=self.TOL_RANDOM_START)
            print(f"Run {r} finished with MSE = ", mse)

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

    def test_local_optimality(self):
        raise NotImplementedError("TODO: implement test that checks that the fitted parameters are a local optimum of the objective, e.g. by checking that the gradient is close to zero or that small perturbations do not decrease MSE.")


# ===========================================================================
# 2. Rollout consistency
# ===========================================================================

# class TestRollouts(unittest.TestCase):

#     def test_rollout_fg_homophily_shape(self):
#         """Rollout should return array of shape (horizon+1, n)."""
#         # TODO: call rollout_fg_homophily, assert shape
#         raise NotImplementedError

#     def test_rollout_fg_fj_bias_homophily_shape(self):
#         # TODO
#         raise NotImplementedError

#     def test_rollout_matches_simulation(self):
#         """One-step rollout should match _sim_fg_fj_bias_homophily on noiseless data."""
#         # TODO: compare rollout trajectory to manual simulation
#         raise NotImplementedError


if __name__ == "__main__":
    unittest.main()
