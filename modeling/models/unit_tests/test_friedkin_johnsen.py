"""Unit tests for Friedkin-Johnsen adjacency-based fitting and rollout.

Covers:
  - Output contract (keys, shapes, value ranges) for joint solvers
  - Parameter recovery on noiseless synthetic data across graph topologies
    (complete, ring, star, chain, random sparse) for both Base-FJ and full FJ
  - Multiple runs with shared and per-run-different neighbor structures
  - Grid-search selectors validate that joint solver MSE is within grid-size
    of the coarse grid result (grid search is an upper bound on joint quality)
  - Rollout shape and consistency checks for both Base-FJ and full FJ
"""

import unittest
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.models.adjacency_based.friedkin_johnsen import (
    fit_base_friedkin_johnsen_adjacency_joint,
    fit_base_friedkin_johnson_adjency,
    base_friedkin_johnsen_adjacency_rollout,
    fit_friedkin_johnsen_adjacency,
    fit_friedkin_johnsen_adjacency_joint,
    friedkin_johnsen_adjacency_rollout,
    select_friedkin_johnsen_adjacency_lambdas,
    select_base_friedkin_johnsen_adjacency_lambda,
)
from modeling.models.data_prep import build_expected_message_matrix


# ---------------------------------------------------------------------------
# Graph helpers (identical to those in test_degroot for consistency)
# ---------------------------------------------------------------------------

def _all_to_all_neighbors(n: int):
    return {i: list(range(n)) for i in range(n)}


def _ring_neighbors(n: int):
    return {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}


def _star_neighbors(n: int):
    nbrs = {0: list(range(1, n))}
    for i in range(1, n):
        nbrs[i] = [0]
    return nbrs


def _chain_neighbors(n: int):
    nbrs = {0: [1]}
    for i in range(1, n):
        nbrs[i] = [i - 1]
    return nbrs


def _random_sparse_neighbors(n: int, in_degree: int, rng: np.random.Generator) -> dict:
    if n < 2:
        raise ValueError("n must be >= 2")
    in_degree = max(1, min(in_degree, n - 1))
    nbrs = {}
    for i in range(n):
        pool = [j for j in range(n) if j != i]
        chosen = rng.choice(pool, size=in_degree, replace=False).tolist()
        nbrs[i] = chosen
    return nbrs


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _row_normalize(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    row_sums = w.sum(axis=1, keepdims=True)
    out = np.zeros_like(w)
    valid = row_sums[:, 0] > 0.0
    out[valid] = w[valid] / row_sums[valid]
    for i in np.where(~valid)[0]:
        out[i, i] = 1.0
    return out


def _make_W(Abar: np.ndarray, gamma: float) -> np.ndarray:
    n = Abar.shape[0]
    return _row_normalize(gamma * Abar + (1.0 - gamma) * np.eye(n))


def _make_base_fj_traj(rng, Abar, lambda1_true, gamma_true, n_runs, horizon, noise_std=0.0):
    """Simulate Base-FJ: x_{t+1} = lambda1*x0 + (1-lambda1)*W @ x_t."""
    n = Abar.shape[0]
    alpha = 1.0 - lambda1_true
    W = _make_W(Abar, gamma_true)
    run_traj = {}
    for r in range(n_runs):
        x0 = rng.uniform(-1.0, 1.0, size=n)
        x = x0.copy()
        states = [x0.copy()]
        for _ in range(horizon):
            x = lambda1_true * x0 + alpha * (W @ x)
            if noise_std > 0.0:
                x = x + noise_std * rng.normal(size=n)
            states.append(x.copy())
        run_traj[f'run_{r:02d}'] = np.asarray(states, dtype=float)
    return run_traj


def _make_fj_traj(rng, Abar, lambda1_true, lambda2_true, gamma_true, bias_true, n_runs, horizon, noise_std=0.0):
    """Simulate full FJ: x_{t+1} = lambda1*x0 + lambda2*bias + alpha*W @ x_t."""
    n = Abar.shape[0]
    alpha = 1.0 - lambda1_true - lambda2_true
    W = _make_W(Abar, gamma_true)
    run_traj = {}
    for r in range(n_runs):
        x0 = rng.uniform(-1.0, 1.0, size=n)
        x = x0.copy()
        states = [x0.copy()]
        for _ in range(horizon):
            x = lambda1_true * x0 + lambda2_true * bias_true + alpha * (W @ x)
            if noise_std > 0.0:
                x = x + noise_std * rng.normal(size=n)
            states.append(x.copy())
        run_traj[f'run_{r:02d}'] = np.asarray(states, dtype=float)
    return run_traj


LOW_NOISE = 1e-4


# ===========================================================================
# Output-contract tests
# ===========================================================================

class TestBaseFJJointOutputContract(unittest.TestCase):
    """fit_base_friedkin_johnsen_adjacency_joint returns well-formed output."""

    def setUp(self):
        rng = np.random.default_rng(0)
        n = 5
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_base_fj_traj(rng, Abar, 0.2, 0.6, n_runs=4, horizon=15)
        run_neighbors = {rn: nbrs for rn in run_traj}
        self.fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.n_runs = 4

    def test_required_keys(self):
        for key in ('lambda1', 'alpha', 'gamma', 'mse_pool', 'X_pool', 'Y_pool',
                    'X0_pool', 'W_blocks', 'Abar_blocks'):
            self.assertIn(key, self.fit)

    def test_lambda1_in_unit_interval(self):
        self.assertGreaterEqual(self.fit['lambda1'], 0.0)
        self.assertLessEqual(self.fit['lambda1'], 1.0)

    def test_alpha_in_unit_interval(self):
        self.assertGreaterEqual(self.fit['alpha'], 0.0)
        self.assertLessEqual(self.fit['alpha'], 1.0)

    def test_gamma_in_unit_interval(self):
        self.assertGreaterEqual(self.fit['gamma'], 0.0)
        self.assertLessEqual(self.fit['gamma'], 1.0)

    def test_lambda1_plus_alpha_leq_one(self):
        self.assertLessEqual(self.fit['lambda1'] + self.fit['alpha'], 1.0 + 1e-6)

    def test_mse_non_negative(self):
        self.assertGreaterEqual(self.fit['mse_pool'], 0.0)

    def test_w_blocks_row_stochastic(self):
        for w in self.fit['W_blocks'].values():
            w = np.asarray(w, dtype=float)
            np.testing.assert_allclose(w.sum(axis=1), np.ones(w.shape[0]), atol=1e-6)

    def test_x_y_pool_shapes_match(self):
        self.assertEqual(
            np.asarray(self.fit['X_pool']).shape,
            np.asarray(self.fit['Y_pool']).shape,
        )

    def test_w_blocks_count(self):
        self.assertEqual(len(self.fit['W_blocks']), self.n_runs)


class TestFJJointOutputContract(unittest.TestCase):
    """fit_friedkin_johnsen_adjacency_joint returns well-formed output."""

    def setUp(self):
        rng = np.random.default_rng(1)
        n = 5
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_fj_traj(rng, Abar, 0.2, 0.1, 0.6, 0.1, n_runs=4, horizon=15)
        run_neighbors = {rn: nbrs for rn in run_traj}
        self.fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.n_runs = 4

    def test_required_keys(self):
        for key in ('lambda1', 'lambda2', 'alpha', 'gamma', 'bias', 'b_tilde',
                    'mse_pool', 'X_pool', 'Y_pool', 'X0_pool', 'W_blocks', 'Abar_blocks'):
            self.assertIn(key, self.fit)

    def test_lambda1_in_unit_interval(self):
        self.assertGreaterEqual(self.fit['lambda1'], 0.0)
        self.assertLessEqual(self.fit['lambda1'], 1.0)

    def test_lambda2_in_unit_interval(self):
        self.assertGreaterEqual(self.fit['lambda2'], 0.0)
        self.assertLessEqual(self.fit['lambda2'], 1.0)

    def test_gamma_in_unit_interval(self):
        self.assertGreaterEqual(self.fit['gamma'], 0.0)
        self.assertLessEqual(self.fit['gamma'], 1.0)

    def test_bias_in_unit_interval(self):
        self.assertGreaterEqual(self.fit['bias'], -1.0)
        self.assertLessEqual(self.fit['bias'], 1.0)

    def test_lambda_sum_leq_one(self):
        total = self.fit['lambda1'] + self.fit['lambda2'] + self.fit['alpha']
        self.assertAlmostEqual(total, 1.0, delta=1e-4)

    def test_mse_non_negative(self):
        self.assertGreaterEqual(self.fit['mse_pool'], 0.0)

    def test_w_blocks_row_stochastic(self):
        for w in self.fit['W_blocks'].values():
            w = np.asarray(w, dtype=float)
            np.testing.assert_allclose(w.sum(axis=1), np.ones(w.shape[0]), atol=1e-6)

    def test_w_blocks_count(self):
        self.assertEqual(len(self.fit['W_blocks']), self.n_runs)


# ===========================================================================
# Base-FJ joint: parameter recovery across graph topologies
# ===========================================================================

class TestBaseFJJointRecoveryCompleteGraph(unittest.TestCase):
    """Base-FJ joint recovers (lambda1, gamma) on noiseless complete-graph data."""

    def _check(self, lambda1_true, gamma_true, n=6, n_runs=8, horizon=20, delta=1e-3):
        rng = np.random.default_rng(10)
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_base_fj_traj(rng, Abar, lambda1_true, gamma_true, n_runs, horizon)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['lambda1'], lambda1_true, delta=delta)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=delta)

    def test_lambda1_02_gamma_05(self):
        self._check(0.2, 0.5)

    def test_lambda1_03_gamma_08(self):
        self._check(0.3, 0.8)

    def test_lambda1_05_gamma_03(self):
        self._check(0.5, 0.3)

    def test_lambda1_01_gamma_07(self):
        self._check(0.1, 0.7)

    def test_mse_near_zero_noiseless(self):
        rng = np.random.default_rng(11)
        n = 6
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_base_fj_traj(rng, Abar, 0.2, 0.6, n_runs=8, horizon=20)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertLess(fit['mse_pool'], 1e-8)


class TestBaseFJJointRecoveryRingGraph(unittest.TestCase):
    """Base-FJ joint recovery on a sparse ring topology."""

    def _check(self, lambda1_true, gamma_true, n=8, n_runs=8, horizon=25, delta=1e-3):
        rng = np.random.default_rng(20)
        nbrs = _ring_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_base_fj_traj(rng, Abar, lambda1_true, gamma_true, n_runs, horizon)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['lambda1'], lambda1_true, delta=delta)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=delta)

    def test_lambda1_02_gamma_04_ring(self):
        self._check(0.2, 0.4)

    def test_lambda1_03_gamma_07_ring(self):
        self._check(0.3, 0.7)

    def test_mse_near_zero_ring(self):
        rng = np.random.default_rng(21)
        n = 8
        nbrs = _ring_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_base_fj_traj(rng, Abar, 0.25, 0.6, n_runs=10, horizon=25)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertLess(fit['mse_pool'], 1e-8)


class TestBaseFJJointRecoveryStarGraph(unittest.TestCase):
    """Base-FJ joint recovery on a star topology."""

    def test_star_recovery(self):
        rng = np.random.default_rng(30)
        n, lambda1_true, gamma_true = 7, 0.25, 0.6
        nbrs = _star_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_base_fj_traj(rng, Abar, lambda1_true, gamma_true, n_runs=8, horizon=25)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['lambda1'], lambda1_true, delta=1e-3)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=1e-3)

    def test_mse_near_zero_star(self):
        rng = np.random.default_rng(31)
        n = 7
        nbrs = _star_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_base_fj_traj(rng, Abar, 0.25, 0.6, n_runs=8, horizon=20)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertLess(fit['mse_pool'], 1e-8)


class TestBaseFJJointRecoveryChainGraph(unittest.TestCase):
    """Base-FJ joint recovery on a directed chain (sparse, asymmetric)."""

    def test_chain_recovery(self):
        rng = np.random.default_rng(40)
        n, lambda1_true, gamma_true = 8, 0.3, 0.55
        nbrs = _chain_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_base_fj_traj(rng, Abar, lambda1_true, gamma_true, n_runs=10, horizon=25)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['lambda1'], lambda1_true, delta=1e-3)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=1e-3)


class TestBaseFJJointRecoveryRandomGraphs(unittest.TestCase):
    """Base-FJ joint recovery on random sparse graphs."""

    def _check(self, seed, n, in_degree, lambda1_true, gamma_true, n_runs=8, horizon=25, delta=1e-3):
        rng = np.random.default_rng(seed)
        nbrs = _random_sparse_neighbors(n, in_degree, rng)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_base_fj_traj(rng, Abar, lambda1_true, gamma_true, n_runs, horizon)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['lambda1'], lambda1_true, delta=delta)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=delta)
        self.assertLess(fit['mse_pool'], 1e-8)

    def test_n8_indegree1_lambda02_gamma05(self):
        self._check(seed=50, n=8, in_degree=1, lambda1_true=0.2, gamma_true=0.5)

    def test_n8_indegree2_lambda03_gamma07(self):
        self._check(seed=51, n=8, in_degree=2, lambda1_true=0.3, gamma_true=0.7)

    def test_n12_indegree3_lambda015_gamma06(self):
        self._check(seed=52, n=12, in_degree=3, lambda1_true=0.15, gamma_true=0.6)

    def test_n10_halfdense_lambda025_gamma04(self):
        self._check(seed=53, n=10, in_degree=5, lambda1_true=0.25, gamma_true=0.4)

    def test_multi_seed_n8_indegree2(self):
        for seed in range(54, 58):
            with self.subTest(seed=seed):
                self._check(seed=seed, n=8, in_degree=2, lambda1_true=0.2, gamma_true=0.6)


class TestBaseFJJointMultipleRuns(unittest.TestCase):
    """Multiple runs shared vs per-run-different neighbor structures."""

    def test_many_runs_complete(self):
        rng = np.random.default_rng(60)
        n, lambda1_true, gamma_true = 5, 0.2, 0.5
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_base_fj_traj(rng, Abar, lambda1_true, gamma_true, n_runs=20, horizon=10)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['lambda1'], lambda1_true, delta=1e-3)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=1e-3)
        self.assertEqual(len(fit['W_blocks']), 20)

    def test_single_run_complete(self):
        rng = np.random.default_rng(61)
        n = 5
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_base_fj_traj(rng, Abar, 0.2, 0.5, n_runs=1, horizon=30)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertGreaterEqual(fit['lambda1'], 0.0)
        self.assertGreaterEqual(fit['gamma'], 0.0)

    def test_mixed_graph_per_run(self):
        """Half runs on ring, half on complete; gamma shared."""
        rng = np.random.default_rng(62)
        n, lambda1_true, gamma_true = 6, 0.2, 0.6
        nbrs_ring = _ring_neighbors(n)
        nbrs_complete = _all_to_all_neighbors(n)
        run_traj, run_neighbors = {}, {}
        for r in range(8):
            nbrs = nbrs_ring if r % 2 == 0 else nbrs_complete
            Abar = build_expected_message_matrix(nbrs, n)
            W = _make_W(Abar, gamma_true)
            x0 = rng.uniform(-1.0, 1.0, size=n)
            x = x0.copy()
            states = [x0.copy()]
            alpha = 1.0 - lambda1_true
            for _ in range(20):
                x = lambda1_true * x0 + alpha * (W @ x)
                states.append(x.copy())
            rn = f'run_{r:02d}'
            run_traj[rn] = np.asarray(states, dtype=float)
            run_neighbors[rn] = nbrs
        fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['lambda1'], lambda1_true, delta=1e-3)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=1e-3)
        self.assertEqual(set(fit['Abar_blocks'].keys()), set(run_traj.keys()))


class TestBaseFJJointWithNoise(unittest.TestCase):
    """Base-FJ joint is approximately correct under low noise."""

    def test_low_noise_complete(self):
        rng = np.random.default_rng(70)
        n, lambda1_true, gamma_true = 6, 0.2, 0.6
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_base_fj_traj(rng, Abar, lambda1_true, gamma_true, n_runs=12, horizon=20, noise_std=LOW_NOISE)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['lambda1'], lambda1_true, delta=1e-2)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=1e-2)

    def test_low_noise_ring(self):
        rng = np.random.default_rng(71)
        n, lambda1_true, gamma_true = 8, 0.25, 0.7
        nbrs = _ring_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_base_fj_traj(rng, Abar, lambda1_true, gamma_true, n_runs=12, horizon=20, noise_std=LOW_NOISE)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['lambda1'], lambda1_true, delta=1e-2)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=1e-2)


# ===========================================================================
# Full FJ joint: parameter recovery across graph topologies
# ===========================================================================

class TestFJJointRecoveryCompleteGraph(unittest.TestCase):
    """Full FJ joint recovers (lambda1, lambda2, gamma, bias) on noiseless complete-graph data."""

    def _check(self, lambda1_true, lambda2_true, gamma_true, bias_true,
               n=6, n_runs=8, horizon=20, delta=1e-2):
        rng = np.random.default_rng(100)
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_fj_traj(rng, Abar, lambda1_true, lambda2_true, gamma_true, bias_true, n_runs, horizon)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['lambda1'], lambda1_true, delta=delta)
        self.assertAlmostEqual(fit['lambda2'], lambda2_true, delta=delta)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=delta)
        self.assertAlmostEqual(fit['bias'], bias_true, delta=delta)

    def test_standard_params(self):
        self._check(0.2, 0.1, 0.6, 0.1)

    def test_zero_bias(self):
        self._check(0.2, 0.1, 0.5, 0.0)

    def test_negative_bias(self):
        self._check(0.15, 0.1, 0.7, -0.2)

    def test_large_lambda1(self):
        self._check(0.4, 0.1, 0.5, 0.1)

    def test_mse_near_zero_noiseless(self):
        rng = np.random.default_rng(101)
        n = 6
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_fj_traj(rng, Abar, 0.2, 0.1, 0.6, 0.1, n_runs=8, horizon=20)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertLess(fit['mse_pool'], 1e-7)


class TestFJJointRecoveryRingGraph(unittest.TestCase):
    """Full FJ joint recovery on a sparse ring topology."""

    def _check(self, lambda1_true, lambda2_true, gamma_true, bias_true,
               n=8, n_runs=10, horizon=25, delta=1e-2):
        rng = np.random.default_rng(110)
        nbrs = _ring_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_fj_traj(rng, Abar, lambda1_true, lambda2_true, gamma_true, bias_true, n_runs, horizon)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['lambda1'], lambda1_true, delta=delta)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=delta)

    def test_ring_lambda1_02_gamma_04(self):
        self._check(0.2, 0.1, 0.4, 0.0)

    def test_ring_lambda1_03_gamma_07(self):
        self._check(0.3, 0.1, 0.7, 0.1)

    def test_mse_near_zero_ring(self):
        rng = np.random.default_rng(111)
        n = 8
        nbrs = _ring_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_fj_traj(rng, Abar, 0.2, 0.1, 0.6, 0.0, n_runs=10, horizon=25)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertLess(fit['mse_pool'], 1e-7)


class TestFJJointRecoveryStarGraph(unittest.TestCase):
    """Full FJ joint recovery on a star topology."""

    def test_star_recovery(self):
        rng = np.random.default_rng(120)
        n, lambda1_true, lambda2_true, gamma_true, bias_true = 7, 0.2, 0.1, 0.6, 0.1
        nbrs = _star_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_fj_traj(rng, Abar, lambda1_true, lambda2_true, gamma_true, bias_true, n_runs=8, horizon=25)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['lambda1'], lambda1_true, delta=1e-2)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=1e-2)

    def test_mse_near_zero_star(self):
        rng = np.random.default_rng(121)
        n = 7
        nbrs = _star_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_fj_traj(rng, Abar, 0.2, 0.1, 0.6, 0.1, n_runs=8, horizon=20)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertLess(fit['mse_pool'], 1e-7)


class TestFJJointRecoveryChainGraph(unittest.TestCase):
    """Full FJ joint recovery on a directed chain."""

    def test_chain_recovery(self):
        rng = np.random.default_rng(130)
        n, lambda1_true, lambda2_true, gamma_true, bias_true = 8, 0.2, 0.1, 0.55, 0.0
        nbrs = _chain_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_fj_traj(rng, Abar, lambda1_true, lambda2_true, gamma_true, bias_true, n_runs=10, horizon=25)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['lambda1'], lambda1_true, delta=1e-2)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=1e-2)


class TestFJJointRecoveryRandomGraphs(unittest.TestCase):
    """Full FJ joint recovery on random sparse graphs."""

    def _check(self, seed, n, in_degree, lambda1_true, lambda2_true, gamma_true, bias_true,
               n_runs=8, horizon=25, delta=1e-2):
        rng = np.random.default_rng(seed)
        nbrs = _random_sparse_neighbors(n, in_degree, rng)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_fj_traj(rng, Abar, lambda1_true, lambda2_true, gamma_true, bias_true, n_runs, horizon)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['lambda1'], lambda1_true, delta=delta)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=delta)
        self.assertLess(fit['mse_pool'], 1e-7)

    def test_n8_indegree1(self):
        self._check(seed=140, n=8, in_degree=1, lambda1_true=0.2, lambda2_true=0.1, gamma_true=0.5, bias_true=0.0)

    def test_n8_indegree2(self):
        self._check(seed=141, n=8, in_degree=2, lambda1_true=0.2, lambda2_true=0.1, gamma_true=0.7, bias_true=0.1)

    def test_n12_indegree3(self):
        self._check(seed=142, n=12, in_degree=3, lambda1_true=0.15, lambda2_true=0.1, gamma_true=0.6, bias_true=0.0)

    def test_n10_halfdense(self):
        self._check(seed=143, n=10, in_degree=5, lambda1_true=0.2, lambda2_true=0.1, gamma_true=0.4, bias_true=0.0)

    def test_multi_seed_n8_indegree2(self):
        for seed in range(144, 148):
            with self.subTest(seed=seed):
                self._check(seed=seed, n=8, in_degree=2, lambda1_true=0.2, lambda2_true=0.1, gamma_true=0.6, bias_true=0.0)


class TestFJJointMultipleRuns(unittest.TestCase):
    """Multiple runs shared vs per-run-different neighbor structures for full FJ."""

    def test_many_runs_complete(self):
        rng = np.random.default_rng(150)
        n, lambda1_true, lambda2_true, gamma_true, bias_true = 5, 0.2, 0.1, 0.5, 0.0
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_fj_traj(rng, Abar, lambda1_true, lambda2_true, gamma_true, bias_true, n_runs=20, horizon=10)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['lambda1'], lambda1_true, delta=1e-2)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=1e-2)
        self.assertEqual(len(fit['W_blocks']), 20)

    def test_single_run(self):
        rng = np.random.default_rng(151)
        n = 5
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_fj_traj(rng, Abar, 0.2, 0.1, 0.5, 0.0, n_runs=1, horizon=30)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertGreaterEqual(fit['lambda1'], 0.0)
        self.assertGreaterEqual(fit['gamma'], 0.0)

    def test_mixed_graph_per_run(self):
        rng = np.random.default_rng(152)
        n, lambda1_true, lambda2_true, gamma_true, bias_true = 6, 0.2, 0.1, 0.6, 0.0
        nbrs_ring = _ring_neighbors(n)
        nbrs_complete = _all_to_all_neighbors(n)
        run_traj, run_neighbors = {}, {}
        for r in range(8):
            nbrs = nbrs_ring if r % 2 == 0 else nbrs_complete
            Abar = build_expected_message_matrix(nbrs, n)
            W = _make_W(Abar, gamma_true)
            x0 = rng.uniform(-1.0, 1.0, size=n)
            x = x0.copy()
            states = [x0.copy()]
            alpha = 1.0 - lambda1_true - lambda2_true
            for _ in range(20):
                x = lambda1_true * x0 + lambda2_true * bias_true + alpha * (W @ x)
                states.append(x.copy())
            rn = f'run_{r:02d}'
            run_traj[rn] = np.asarray(states, dtype=float)
            run_neighbors[rn] = nbrs
        fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['lambda1'], lambda1_true, delta=1e-2)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=1e-2)
        self.assertEqual(set(fit['Abar_blocks'].keys()), set(run_traj.keys()))


class TestFJJointWithNoise(unittest.TestCase):
    """Full FJ joint is approximately correct under low noise."""

    def test_low_noise_complete(self):
        rng = np.random.default_rng(160)
        n, lambda1_true, lambda2_true, gamma_true, bias_true = 6, 0.2, 0.1, 0.6, 0.1
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_fj_traj(rng, Abar, lambda1_true, lambda2_true, gamma_true, bias_true,
                                  n_runs=12, horizon=20, noise_std=LOW_NOISE)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['lambda1'], lambda1_true, delta=5e-2)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=5e-2)

    def test_low_noise_ring(self):
        rng = np.random.default_rng(161)
        n, lambda1_true, lambda2_true, gamma_true, bias_true = 8, 0.2, 0.1, 0.7, 0.0
        nbrs = _ring_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_fj_traj(rng, Abar, lambda1_true, lambda2_true, gamma_true, bias_true,
                                  n_runs=12, horizon=20, noise_std=LOW_NOISE)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['lambda1'], lambda1_true, delta=5e-2)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=5e-2)


# ===========================================================================
# Grid-search selectors validate against joint solvers
# (grid search MSE >= joint MSE; difference bounded by grid resolution)
# ===========================================================================

class TestSelectBaseFJLambdaGridVsJoint(unittest.TestCase):
    """Grid search and joint solver agree within the grid step size.

    The joint solver is a convex QP; grid search + inner QP is an upper
    bound on the globally optimal MSE.  Therefore:
        grid_mse >= joint_mse
    and both should be close when the grid is fine enough.
    """

    GRID = np.linspace(0.0, 1.0, 11)   # step = 0.1
    GRID_STEP = 0.1

    def _make_data(self, seed, n=6, lambda1_true=0.2, gamma_true=0.6):
        rng = np.random.default_rng(seed)
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_base_fj_traj(rng, Abar, lambda1_true, gamma_true, n_runs=8, horizon=20)
        run_neighbors = {rn: nbrs for rn in run_traj}
        return run_traj, run_neighbors

    def test_joint_mse_leq_grid_mse(self):
        """Joint solver MSE should not be higher than any grid point MSE."""
        run_traj, run_neighbors = self._make_data(seed=200)
        grid_best, _ = select_base_friedkin_johnsen_adjacency_lambda(run_traj, run_neighbors, lambda_grid=self.GRID)
        joint_fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertLessEqual(joint_fit['mse_pool'], grid_best['mse_pool'] + 1e-6)

    def test_grid_mse_within_grid_size_of_joint(self):
        """Grid result MSE should be close to joint MSE (within reasonable margin)."""
        run_traj, run_neighbors = self._make_data(seed=201)
        grid_best, _ = select_base_friedkin_johnsen_adjacency_lambda(run_traj, run_neighbors, lambda_grid=self.GRID)
        joint_fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        # Grid MSE can be higher but shouldn't be vastly worse
        self.assertLess(grid_best['mse_pool'] - joint_fit['mse_pool'], 0.05)

    def test_grid_lambda1_within_grid_step_of_joint(self):
        """Grid best lambda1 should be within one grid step of the joint solution."""
        run_traj, run_neighbors = self._make_data(seed=202)
        grid_best, _ = select_base_friedkin_johnsen_adjacency_lambda(run_traj, run_neighbors, lambda_grid=self.GRID)
        joint_fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(grid_best['lambda1'], joint_fit['lambda1'], delta=self.GRID_STEP + 1e-3)

    def test_no_grid_falls_back_to_joint(self):
        """Calling with lambda_grid=None should return the joint optimizer result."""
        run_traj, run_neighbors = self._make_data(seed=203)
        best, results = select_base_friedkin_johnsen_adjacency_lambda(run_traj, run_neighbors, lambda_grid=None)
        joint_fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(best['lambda1'], joint_fit['lambda1'], delta=1e-4)
        self.assertAlmostEqual(best['gamma'], joint_fit['gamma'], delta=1e-4)

    def test_returns_tuple_of_best_and_list(self):
        run_traj, run_neighbors = self._make_data(seed=204)
        result = select_base_friedkin_johnsen_adjacency_lambda(run_traj, run_neighbors, lambda_grid=self.GRID)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        best, results_list = result
        self.assertIsInstance(results_list, list)
        self.assertIn('lambda1', best)
        self.assertIn('gamma', best)
        self.assertIn('mse_pool', best)

    def test_grid_recovery_ring(self):
        rng = np.random.default_rng(205)
        n, lambda1_true, gamma_true = 8, 0.3, 0.7
        nbrs = _ring_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_base_fj_traj(rng, Abar, lambda1_true, gamma_true, n_runs=8, horizon=25)
        run_neighbors = {rn: nbrs for rn in run_traj}
        grid_best, _ = select_base_friedkin_johnsen_adjacency_lambda(run_traj, run_neighbors, lambda_grid=self.GRID)
        self.assertAlmostEqual(grid_best['lambda1'], lambda1_true, delta=self.GRID_STEP + 1e-3)


class TestSelectFJLambdasGridVsJoint(unittest.TestCase):
    """Grid search (lambda1, lambda2) and full FJ joint agree within grid resolution."""

    GRID = np.linspace(0.0, 0.5, 6)   # step = 0.1, only to 0.5 to avoid lambda1+lambda2>1
    GRID_STEP = 0.1

    def _make_data(self, seed, n=6, lambda1_true=0.2, lambda2_true=0.1, gamma_true=0.6, bias_true=0.0):
        rng = np.random.default_rng(seed)
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_fj_traj(rng, Abar, lambda1_true, lambda2_true, gamma_true, bias_true, n_runs=8, horizon=20)
        run_neighbors = {rn: nbrs for rn in run_traj}
        return run_traj, run_neighbors

    def test_joint_mse_leq_grid_mse(self):
        """Joint solver MSE should not exceed grid search MSE."""
        run_traj, run_neighbors = self._make_data(seed=210)
        grid_best, _ = select_friedkin_johnsen_adjacency_lambdas(run_traj, run_neighbors, lambda_grid=self.GRID)
        joint_fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertLessEqual(joint_fit['mse_pool'], grid_best['mse_pool'] + 1e-6)

    def test_grid_mse_within_margin_of_joint(self):
        run_traj, run_neighbors = self._make_data(seed=211)
        grid_best, _ = select_friedkin_johnsen_adjacency_lambdas(run_traj, run_neighbors, lambda_grid=self.GRID)
        joint_fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertLess(grid_best['mse_pool'] - joint_fit['mse_pool'], 0.05)

    def test_grid_lambda1_within_grid_step_of_joint(self):
        run_traj, run_neighbors = self._make_data(seed=212)
        grid_best, _ = select_friedkin_johnsen_adjacency_lambdas(run_traj, run_neighbors, lambda_grid=self.GRID)
        joint_fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(grid_best['lambda1'], joint_fit['lambda1'], delta=self.GRID_STEP + 1e-3)

    def test_no_grid_falls_back_to_joint(self):
        run_traj, run_neighbors = self._make_data(seed=213)
        best, results = select_friedkin_johnsen_adjacency_lambdas(run_traj, run_neighbors, lambda_grid=None)
        joint_fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(best['lambda1'], joint_fit['lambda1'], delta=1e-4)
        self.assertAlmostEqual(best['gamma'], joint_fit['gamma'], delta=1e-4)

    def test_returns_tuple_best_and_list(self):
        run_traj, run_neighbors = self._make_data(seed=214)
        result = select_friedkin_johnsen_adjacency_lambdas(run_traj, run_neighbors, lambda_grid=self.GRID)
        self.assertIsInstance(result, tuple)
        best, results_list = result
        self.assertIn('lambda1', best)
        self.assertIn('lambda2', best)
        self.assertIn('gamma', best)
        self.assertIn('mse_pool', best)
        self.assertIsInstance(results_list, list)
        self.assertGreater(len(results_list), 0)

    def test_grid_recovery_ring(self):
        rng = np.random.default_rng(215)
        n, lambda1_true, lambda2_true, gamma_true, bias_true = 8, 0.2, 0.1, 0.7, 0.0
        nbrs = _ring_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_fj_traj(rng, Abar, lambda1_true, lambda2_true, gamma_true, bias_true, n_runs=10, horizon=25)
        run_neighbors = {rn: nbrs for rn in run_traj}
        grid_best, _ = select_friedkin_johnsen_adjacency_lambdas(run_traj, run_neighbors, lambda_grid=self.GRID)
        self.assertAlmostEqual(grid_best['lambda1'], lambda1_true, delta=self.GRID_STEP + 1e-3)

    def test_grid_all_results_have_required_keys(self):
        run_traj, run_neighbors = self._make_data(seed=216)
        _, results_list = select_friedkin_johnsen_adjacency_lambdas(run_traj, run_neighbors, lambda_grid=self.GRID)
        for r in results_list:
            for key in ('lambda1', 'lambda2', 'gamma', 'bias', 'mse_pool'):
                self.assertIn(key, r)


# ===========================================================================
# Rollout: Base-FJ and full FJ
# ===========================================================================

class TestBaseFJRollout(unittest.TestCase):
    """base_friedkin_johnsen_adjacency_rollout shape and consistency."""

    def setUp(self):
        n = 5
        W = np.eye(n, dtype=float)
        self.W = W
        self.x0 = np.array([0.1, -0.5, 0.3, 0.8, -0.2])
        self.n = n

    def test_output_shape(self):
        pred = base_friedkin_johnsen_adjacency_rollout(self.W, self.x0, horizon=10, lambda1=0.2)
        self.assertEqual(np.asarray(pred).shape, (11, self.n))

    def test_horizon_one(self):
        pred = base_friedkin_johnsen_adjacency_rollout(self.W, self.x0, horizon=1, lambda1=0.2)
        self.assertEqual(np.asarray(pred).shape, (2, self.n))

    def test_identity_w_converges_to_x0(self):
        """With W=I, x_{t+1} = lambda1*x0 + alpha*x_t.
        This converges to x_inf = lambda1/(1-alpha) * x0 = x0."""
        lambda1 = 0.3
        pred = np.asarray(base_friedkin_johnsen_adjacency_rollout(self.W, self.x0, horizon=100, lambda1=lambda1))
        # Fixed point of x = lambda1*x0 + (1-lambda1)*x is x0
        np.testing.assert_allclose(pred[-1], self.x0, atol=1e-6)

    def test_first_step_correct(self):
        lambda1 = 0.3
        n = self.n
        W = _make_W(np.ones((n, n)) / n, 0.8)
        alpha = 1.0 - lambda1
        pred = np.asarray(base_friedkin_johnsen_adjacency_rollout(W, self.x0, horizon=1, lambda1=lambda1))
        expected_step1 = lambda1 * self.x0 + alpha * (W @ self.x0)
        np.testing.assert_allclose(pred[1], expected_step1, atol=1e-10)

    def test_noiseless_simulation_match(self):
        rng = np.random.default_rng(300)
        n, lambda1_true, gamma_true = 5, 0.25, 0.6
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        W = _make_W(Abar, gamma_true)
        x0 = rng.uniform(-1.0, 1.0, size=n)
        horizon = 8
        alpha = 1.0 - lambda1_true
        x = x0.copy()
        expected = []
        for _ in range(horizon):
            x = lambda1_true * x0 + alpha * (W @ x)
            expected.append(x.copy())
        pred = np.asarray(base_friedkin_johnsen_adjacency_rollout(W, x0, horizon, lambda1_true))
        np.testing.assert_allclose(pred[1:], np.asarray(expected), atol=1e-10)

    def test_lambda1_zero_reduces_to_degroot(self):
        """With lambda1=0, Base-FJ reduces to pure DeGroot: x_{t+1} = W @ x_t."""
        rng = np.random.default_rng(301)
        n = 5
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        W = _make_W(Abar, 0.7)
        x0 = rng.uniform(-1.0, 1.0, size=n)
        pred = np.asarray(base_friedkin_johnsen_adjacency_rollout(W, x0, horizon=5, lambda1=0.0))
        x = x0.copy()
        expected = [x0.copy()]
        for _ in range(5):
            x = W @ x
            expected.append(x.copy())
        np.testing.assert_allclose(pred, np.asarray(expected), atol=1e-10)


class TestFJRollout(unittest.TestCase):
    """friedkin_johnsen_adjacency_rollout shape and consistency."""

    def setUp(self):
        self.n = 5
        self.x0 = np.array([0.1, -0.5, 0.3, 0.8, -0.2])

    def test_output_shape(self):
        W = np.eye(self.n, dtype=float)
        pred = friedkin_johnsen_adjacency_rollout(W, bias=0.0, x0=self.x0, horizon=10, lambda1=0.2, lambda2=0.1)
        self.assertEqual(np.asarray(pred).shape, (11, self.n))

    def test_horizon_one(self):
        W = np.eye(self.n, dtype=float)
        pred = friedkin_johnsen_adjacency_rollout(W, bias=0.0, x0=self.x0, horizon=1, lambda1=0.2, lambda2=0.1)
        self.assertEqual(np.asarray(pred).shape, (2, self.n))

    def test_first_step_correct(self):
        rng = np.random.default_rng(310)
        n = self.n
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        W = _make_W(Abar, 0.6)
        lambda1, lambda2, bias = 0.2, 0.1, 0.15
        alpha = 1.0 - lambda1 - lambda2
        pred = np.asarray(friedkin_johnsen_adjacency_rollout(W, bias, self.x0, horizon=1, lambda1=lambda1, lambda2=lambda2))
        expected_step1 = lambda1 * self.x0 + lambda2 * bias + alpha * (W @ self.x0)
        np.testing.assert_allclose(pred[1], expected_step1, atol=1e-10)

    def test_zero_bias_matches_base_fj(self):
        """With lambda2=0 (and bias=0), full FJ should match Base-FJ rollout."""
        rng = np.random.default_rng(311)
        n = 5
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        W = _make_W(Abar, 0.5)
        lambda1 = 0.2
        x0 = rng.uniform(-1.0, 1.0, size=n)
        pred_fj = np.asarray(friedkin_johnsen_adjacency_rollout(W, 0.0, x0, horizon=8, lambda1=lambda1, lambda2=0.0))
        pred_base = np.asarray(base_friedkin_johnsen_adjacency_rollout(W, x0, horizon=8, lambda1=lambda1))
        np.testing.assert_allclose(pred_fj, pred_base, atol=1e-10)

    def test_noiseless_simulation_match(self):
        rng = np.random.default_rng(312)
        n = 5
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        W = _make_W(Abar, 0.6)
        lambda1, lambda2, bias = 0.2, 0.1, 0.15
        alpha = 1.0 - lambda1 - lambda2
        x0 = rng.uniform(-1.0, 1.0, size=n)
        horizon = 8
        x = x0.copy()
        expected = []
        for _ in range(horizon):
            x = lambda1 * x0 + lambda2 * bias + alpha * (W @ x)
            expected.append(x.copy())
        pred = np.asarray(friedkin_johnsen_adjacency_rollout(W, bias, x0, horizon, lambda1, lambda2))
        np.testing.assert_allclose(pred[1:], np.asarray(expected), atol=1e-10)


# ===========================================================================
# Fixed-parameter fitters (sanity tests via the grid-search code paths)
# ===========================================================================

class TestFitBaseFJFixed(unittest.TestCase):
    """fit_base_friedkin_johnson_adjency (fixed lambda1) recovers gamma."""

    def test_gamma_recovery_complete(self):
        rng = np.random.default_rng(400)
        n, lambda1_true, gamma_true = 6, 0.3, 0.6
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_base_fj_traj(rng, Abar, lambda1_true, gamma_true, n_runs=8, horizon=20)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_base_friedkin_johnson_adjency(run_traj, run_neighbors, lambda1=lambda1_true)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=1e-3)
        self.assertLess(fit['mse_pool'], 1e-8)

    def test_gamma_recovery_ring(self):
        rng = np.random.default_rng(401)
        n, lambda1_true, gamma_true = 8, 0.25, 0.7
        nbrs = _ring_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_base_fj_traj(rng, Abar, lambda1_true, gamma_true, n_runs=8, horizon=25)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_base_friedkin_johnson_adjency(run_traj, run_neighbors, lambda1=lambda1_true)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=1e-3)

    def test_invalid_lambda1_raises(self):
        with self.assertRaises(ValueError):
            fit_base_friedkin_johnson_adjency({}, {}, lambda1=-0.1)

    def test_output_keys(self):
        rng = np.random.default_rng(402)
        n = 5
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_base_fj_traj(rng, Abar, 0.2, 0.5, n_runs=4, horizon=15)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_base_friedkin_johnson_adjency(run_traj, run_neighbors, lambda1=0.2)
        for key in ('gamma', 'mse_pool', 'X_pool', 'Y_pool', 'W_blocks', 'Abar_blocks'):
            self.assertIn(key, fit)


class TestFitFJFixed(unittest.TestCase):
    """fit_friedkin_johnsen_adjacency (fixed lambda1, lambda2) recovers gamma and bias."""

    def test_gamma_bias_recovery_complete(self):
        rng = np.random.default_rng(410)
        n, lambda1_true, lambda2_true, gamma_true, bias_true = 6, 0.2, 0.1, 0.6, 0.1
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_fj_traj(rng, Abar, lambda1_true, lambda2_true, gamma_true, bias_true, n_runs=8, horizon=20)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_friedkin_johnsen_adjacency(run_traj, run_neighbors, lambda1=lambda1_true, lambda2=lambda2_true)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=1e-3)
        self.assertAlmostEqual(fit['bias'], bias_true, delta=1e-3)
        self.assertLess(fit['mse_pool'], 1e-7)

    def test_gamma_recovery_ring(self):
        rng = np.random.default_rng(411)
        n, lambda1_true, lambda2_true, gamma_true, bias_true = 8, 0.2, 0.1, 0.7, 0.0
        nbrs = _ring_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_fj_traj(rng, Abar, lambda1_true, lambda2_true, gamma_true, bias_true, n_runs=8, horizon=25)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_friedkin_johnsen_adjacency(run_traj, run_neighbors, lambda1=lambda1_true, lambda2=lambda2_true)
        self.assertAlmostEqual(fit['gamma'], gamma_true, delta=1e-3)

    def test_invalid_params_raise(self):
        with self.assertRaises(ValueError):
            fit_friedkin_johnsen_adjacency({}, {}, lambda1=-0.1, lambda2=0.1)
        with self.assertRaises(ValueError):
            fit_friedkin_johnsen_adjacency({}, {}, lambda1=0.6, lambda2=0.6)

    def test_output_keys(self):
        rng = np.random.default_rng(412)
        n = 5
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_fj_traj(rng, Abar, 0.2, 0.1, 0.5, 0.0, n_runs=4, horizon=15)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_friedkin_johnsen_adjacency(run_traj, run_neighbors, lambda1=0.2, lambda2=0.1)
        for key in ('gamma', 'bias', 'mse_pool', 'X_pool', 'Y_pool', 'W_blocks', 'Abar_blocks'):
            self.assertIn(key, fit)


if __name__ == "__main__":
    unittest.main()
