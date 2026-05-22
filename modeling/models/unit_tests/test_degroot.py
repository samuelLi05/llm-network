"""Unit tests for fit_degroot_adjacency_scalar and degroot_rollout_prediction.

Covers:
  - Parameter recovery on noiseless synthetic data (complete and sparse graphs)
  - Multiple runs with a shared graph
  - Multiple runs with per-run different neighbor structures
  - Output dict contract (keys, shapes, value ranges)
  - Rollout shape and initial-condition consistency
"""

import unittest
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.models.adjacency_based.degroot import (
    fit_degroot_adjacency_scalar,
    degroot_rollout_prediction,
)
from modeling.models.data_prep import build_expected_message_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_to_all_neighbors(n: int):
    """Every agent reads from every other agent (complete graph)."""
    return {i: list(range(n)) for i in range(n)}


def _ring_neighbors(n: int):
    """Each agent reads only from its left and right neighbours (ring graph)."""
    return {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}


def _star_neighbors(n: int):
    """Hub-and-spoke: all non-hub agents read only from agent 0 (hub).
    Agent 0 reads from all spokes."""
    nbrs = {0: list(range(1, n))}
    for i in range(1, n):
        nbrs[i] = [0]
    return nbrs


def _chain_neighbors(n: int):
    """Agent i reads from agent i-1 only (directed chain).
    Agent 0 wraps around and reads from agent 1 to avoid a self-loop."""
    nbrs = {}
    nbrs[0] = [1]
    for i in range(1, n):
        nbrs[i] = [i - 1]
    return nbrs


def _make_run_traj(rng, Abar, gamma_true, n_runs, horizon, noise_std=0.0):
    """Simulate DeGroot trajectories: x_{t+1} = (gamma*Abar + (1-gamma)*I) @ x_t + noise."""
    W = gamma_true * Abar + (1.0 - gamma_true) * np.eye(Abar.shape[0], dtype=float)
    run_traj = {}
    for r in range(n_runs):
        x = rng.uniform(-1.0, 1.0, size=Abar.shape[0])
        states = [x.copy()]
        for _ in range(horizon):
            x = W @ x + noise_std * rng.normal(size=Abar.shape[0])
            states.append(x.copy())
        run_traj[f'run_{r:02d}'] = np.asarray(states, dtype=float)
    return run_traj


NOISELESS = 0.0
LOW_NOISE = 1e-4


def _random_sparse_neighbors(n: int, in_degree: int, rng: np.random.Generator) -> dict:
    """Each agent reads from exactly `in_degree` distinct sources chosen uniformly at random
    (excluding itself). At least one non-self neighbor is always guaranteed."""
    if n < 2:
        raise ValueError("n must be >= 2 to guarantee a non-self neighbor")
    in_degree = max(1, min(in_degree, n - 1))  # clamp: at least 1, at most n-1
    nbrs = {}
    for i in range(n):
        pool = [j for j in range(n) if j != i]
        chosen = rng.choice(pool, size=in_degree, replace=False).tolist()
        nbrs[i] = chosen
    return nbrs


# ---------------------------------------------------------------------------
# TestFitDeGrootAdjacencyScalar
# ---------------------------------------------------------------------------

class TestOutputContract(unittest.TestCase):
    """fit_degroot_adjacency_scalar returns expected keys and well-formed values."""

    def setUp(self):
        rng = np.random.default_rng(0)
        n, gamma_true = 5, 0.6
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_run_traj(rng, Abar, gamma_true, n_runs=4, horizon=15)
        run_neighbors = {rn: nbrs for rn in run_traj}
        self.fit = fit_degroot_adjacency_scalar(run_traj, run_neighbors)

    def test_required_keys_present(self):
        for key in ('gamma', 'mse_pool', 'X_pool', 'Y_pool', 'W_blocks', 'Abar_blocks'):
            self.assertIn(key, self.fit)

    def test_gamma_in_unit_interval(self):
        self.assertGreaterEqual(float(self.fit['gamma']), 0.0)
        self.assertLessEqual(float(self.fit['gamma']), 1.0)

    def test_mse_pool_non_negative(self):
        self.assertGreaterEqual(float(self.fit['mse_pool']), 0.0)

    def test_w_blocks_row_stochastic(self):
        for w in self.fit['W_blocks'].values():
            w = np.asarray(w, dtype=float)
            row_sums = w.sum(axis=1)
            np.testing.assert_allclose(row_sums, np.ones(len(row_sums)), atol=1e-6)

    def test_x_pool_y_pool_shapes_match(self):
        X = np.asarray(self.fit['X_pool'])
        Y = np.asarray(self.fit['Y_pool'])
        self.assertEqual(X.shape, Y.shape)


# ---------------------------------------------------------------------------

class TestGammaRecoveryCompleteGraph(unittest.TestCase):
    """Gamma is recovered on noiseless data with a complete (all-to-all) graph."""

    def _check_recovery(self, gamma_true, n=6, n_runs=6, horizon=20, delta=1e-4):
        rng = np.random.default_rng(42)
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_run_traj(rng, Abar, gamma_true, n_runs, horizon)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_degroot_adjacency_scalar(run_traj, run_neighbors)
        self.assertAlmostEqual(float(fit['gamma']), gamma_true, delta=delta)

    def test_gamma_0_2(self):
        self._check_recovery(0.2)

    def test_gamma_0_5(self):
        self._check_recovery(0.5)

    def test_gamma_0_8(self):
        self._check_recovery(0.8)

    def test_gamma_boundary_zero(self):
        self._check_recovery(0.0, delta=1e-4)

    def test_gamma_boundary_one(self):
        self._check_recovery(1.0, delta=1e-4)

    def test_mse_near_zero_noiseless(self):
        rng = np.random.default_rng(7)
        n, gamma_true = 5, 0.65
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_run_traj(rng, Abar, gamma_true, n_runs=8, horizon=20)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_degroot_adjacency_scalar(run_traj, run_neighbors)
        self.assertLess(float(fit['mse_pool']), 1e-8)


# ---------------------------------------------------------------------------

class TestGammaRecoveryRingGraph(unittest.TestCase):
    """Gamma recovery on a sparse ring topology (non-complete graph)."""

    def _check_recovery(self, gamma_true, n=8, n_runs=8, horizon=25, delta=1e-4):
        rng = np.random.default_rng(123)
        nbrs = _ring_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_run_traj(rng, Abar, gamma_true, n_runs, horizon)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_degroot_adjacency_scalar(run_traj, run_neighbors)
        self.assertAlmostEqual(float(fit['gamma']), gamma_true, delta=delta)

    def test_gamma_0_3_ring(self):
        self._check_recovery(0.3)

    def test_gamma_0_7_ring(self):
        self._check_recovery(0.7)

    def test_mse_near_zero_ring_noiseless(self):
        rng = np.random.default_rng(55)
        n, gamma_true = 6, 0.5
        nbrs = _ring_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_run_traj(rng, Abar, gamma_true, n_runs=8, horizon=20)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_degroot_adjacency_scalar(run_traj, run_neighbors)
        self.assertLess(float(fit['mse_pool']), 1e-8)


# ---------------------------------------------------------------------------

class TestGammaRecoveryStarGraph(unittest.TestCase):
    """Gamma recovery on a star (hub-and-spoke) topology."""

    def test_gamma_star(self):
        rng = np.random.default_rng(99)
        n, gamma_true = 7, 0.6
        nbrs = _star_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_run_traj(rng, Abar, gamma_true, n_runs=8, horizon=25)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_degroot_adjacency_scalar(run_traj, run_neighbors)
        self.assertAlmostEqual(float(fit['gamma']), gamma_true, delta=1e-4)

    def test_mse_near_zero_star_noiseless(self):
        rng = np.random.default_rng(100)
        n, gamma_true = 7, 0.6
        nbrs = _star_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_run_traj(rng, Abar, gamma_true, n_runs=8, horizon=20)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_degroot_adjacency_scalar(run_traj, run_neighbors)
        self.assertLess(float(fit['mse_pool']), 1e-8)


# ---------------------------------------------------------------------------

class TestGammaRecoveryChainGraph(unittest.TestCase):
    """Gamma recovery on a directed chain (sparse, asymmetric)."""

    def test_gamma_chain(self):
        rng = np.random.default_rng(200)
        n, gamma_true = 8, 0.55
        nbrs = _chain_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_run_traj(rng, Abar, gamma_true, n_runs=10, horizon=25)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_degroot_adjacency_scalar(run_traj, run_neighbors)
        self.assertAlmostEqual(float(fit['gamma']), gamma_true, delta=1e-4)


# ---------------------------------------------------------------------------

class TestMultipleRunsSharedGraph(unittest.TestCase):
    """Multiple runs all generated under the same neighbor structure."""

    def test_many_runs_complete(self):
        rng = np.random.default_rng(300)
        n, gamma_true = 5, 0.45
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_run_traj(rng, Abar, gamma_true, n_runs=20, horizon=10)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_degroot_adjacency_scalar(run_traj, run_neighbors)
        self.assertAlmostEqual(float(fit['gamma']), gamma_true, delta=1e-4)
        self.assertEqual(len(fit['W_blocks']), 20)

    def test_single_run(self):
        """Fit should work with a single run (edge case)."""
        rng = np.random.default_rng(301)
        n, gamma_true = 5, 0.5
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_run_traj(rng, Abar, gamma_true, n_runs=1, horizon=30)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_degroot_adjacency_scalar(run_traj, run_neighbors)
        self.assertGreaterEqual(float(fit['gamma']), 0.0)
        self.assertLessEqual(float(fit['gamma']), 1.0)


# ---------------------------------------------------------------------------

class TestMultipleRunsDifferentGraphs(unittest.TestCase):
    """Multiple runs each generated under a *different* neighbor structure.

    The fitter should handle heterogeneous adjacency across runs by using
    per-run Abar matrices.
    """

    def _make_mixed_runs(self, rng, gamma_true, n=6, horizon=20):
        """Half the runs on a ring, half on a complete graph."""
        nbrs_ring = _ring_neighbors(n)
        nbrs_complete = _all_to_all_neighbors(n)
        Abar_ring = build_expected_message_matrix(nbrs_ring, n)
        Abar_complete = build_expected_message_matrix(nbrs_complete, n)

        run_traj = {}
        run_neighbors = {}
        for r in range(4):
            Abar = Abar_ring if r % 2 == 0 else Abar_complete
            nbrs = nbrs_ring if r % 2 == 0 else nbrs_complete
            W = gamma_true * Abar + (1.0 - gamma_true) * np.eye(n)
            x = rng.uniform(-1.0, 1.0, size=n)
            states = [x.copy()]
            for _ in range(horizon):
                x = W @ x
                states.append(x.copy())
            rn = f'run_{r:02d}'
            run_traj[rn] = np.asarray(states, dtype=float)
            run_neighbors[rn] = nbrs
        return run_traj, run_neighbors

    def test_mixed_graph_recovery(self):
        rng = np.random.default_rng(400)
        gamma_true = 0.6
        run_traj, run_neighbors = self._make_mixed_runs(rng, gamma_true)
        fit = fit_degroot_adjacency_scalar(run_traj, run_neighbors)
        self.assertAlmostEqual(float(fit['gamma']), gamma_true, delta=1e-4)

    def test_w_blocks_count_matches_runs(self):
        rng = np.random.default_rng(401)
        run_traj, run_neighbors = self._make_mixed_runs(rng, gamma_true=0.5)
        fit = fit_degroot_adjacency_scalar(run_traj, run_neighbors)
        self.assertEqual(len(fit['W_blocks']), len(run_traj))

    def test_per_run_abar_keys_match_run_names(self):
        rng = np.random.default_rng(402)
        run_traj, run_neighbors = self._make_mixed_runs(rng, gamma_true=0.5)
        fit = fit_degroot_adjacency_scalar(run_traj, run_neighbors)
        self.assertEqual(set(fit['Abar_blocks'].keys()), set(run_traj.keys()))


# ---------------------------------------------------------------------------

class TestGammaRecoveryWithNoise(unittest.TestCase):
    """With low noise the fit should still be close to ground truth."""

    def test_low_noise_complete(self):
        rng = np.random.default_rng(500)
        n, gamma_true = 6, 0.55
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_run_traj(rng, Abar, gamma_true, n_runs=12, horizon=20, noise_std=LOW_NOISE)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_degroot_adjacency_scalar(run_traj, run_neighbors)
        self.assertAlmostEqual(float(fit['gamma']), gamma_true, delta=1e-3)

    def test_low_noise_ring(self):
        rng = np.random.default_rng(501)
        n, gamma_true = 8, 0.7
        nbrs = _ring_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_run_traj(rng, Abar, gamma_true, n_runs=12, horizon=20, noise_std=LOW_NOISE)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_degroot_adjacency_scalar(run_traj, run_neighbors)
        self.assertAlmostEqual(float(fit['gamma']), gamma_true, delta=1e-3)


# ---------------------------------------------------------------------------

class TestGammaRecoveryRandomGraphs(unittest.TestCase):
    """Gamma recovery on random sparse graphs with varying in-degree.

    Uses multiple random seeds to guard against lucky coincidences.
    """

    def _check_random(self, seed, n, in_degree, gamma_true, n_runs=8, horizon=25):
        rng = np.random.default_rng(seed)
        nbrs = _random_sparse_neighbors(n, in_degree, rng)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _make_run_traj(rng, Abar, gamma_true, n_runs, horizon)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_degroot_adjacency_scalar(run_traj, run_neighbors)
        self.assertAlmostEqual(float(fit['gamma']), gamma_true, delta=1e-4)
        self.assertLess(float(fit['mse_pool']), 1e-8)

    # --- in_degree=1 (very sparse) ---

    def test_random_n8_indegree1_gamma04_seed0(self):
        self._check_random(seed=700, n=8, in_degree=1, gamma_true=0.4)

    def test_random_n8_indegree1_gamma07_seed1(self):
        self._check_random(seed=701, n=8, in_degree=1, gamma_true=0.7)

    # --- in_degree=2 ---

    def test_random_n8_indegree2_gamma03_seed0(self):
        self._check_random(seed=710, n=8, in_degree=2, gamma_true=0.3)

    def test_random_n8_indegree2_gamma06_seed1(self):
        self._check_random(seed=711, n=8, in_degree=2, gamma_true=0.6)

    def test_random_n8_indegree2_gamma09_seed2(self):
        self._check_random(seed=712, n=8, in_degree=2, gamma_true=0.9)

    # --- in_degree=3, larger graph ---

    def test_random_n12_indegree3_gamma05_seed0(self):
        self._check_random(seed=720, n=12, in_degree=3, gamma_true=0.5)

    def test_random_n12_indegree3_gamma08_seed1(self):
        self._check_random(seed=721, n=12, in_degree=3, gamma_true=0.8)

    # --- half-dense (in_degree = n//2) ---

    def test_random_n10_halfdense_gamma05(self):
        self._check_random(seed=730, n=10, in_degree=5, gamma_true=0.5)

    def test_random_n10_halfdense_gamma025(self):
        self._check_random(seed=731, n=10, in_degree=5, gamma_true=0.25)

    # --- multiple random seeds for the same config ---

    def test_random_n8_indegree2_multipleseeds(self):
        for seed in range(740, 745):
            with self.subTest(seed=seed):
                self._check_random(seed=seed, n=8, in_degree=2, gamma_true=0.55)

    # --- per-run different random graphs ---

    def test_random_per_run_different_graphs(self):
        """Each run has an independently sampled random neighbor structure."""
        n, gamma_true = 8, 0.6
        n_runs = 6
        rng = np.random.default_rng(750)
        run_traj = {}
        run_neighbors = {}
        for r in range(n_runs):
            nbrs = _random_sparse_neighbors(n, in_degree=3, rng=rng)
            Abar = build_expected_message_matrix(nbrs, n)
            W = gamma_true * Abar + (1.0 - gamma_true) * np.eye(n)
            x = rng.uniform(-1.0, 1.0, size=n)
            states = [x.copy()]
            for _ in range(25):
                x = W @ x
                states.append(x.copy())
            rn = f'run_{r:02d}'
            run_traj[rn] = np.asarray(states, dtype=float)
            run_neighbors[rn] = nbrs
        fit = fit_degroot_adjacency_scalar(run_traj, run_neighbors)
        self.assertAlmostEqual(float(fit['gamma']), gamma_true, delta=1e-4)
        self.assertLess(float(fit['mse_pool']), 1e-8)


# ---------------------------------------------------------------------------
# TestDeGrootRolloutPrediction
# ---------------------------------------------------------------------------

class TestRolloutShape(unittest.TestCase):
    """degroot_rollout_prediction returns correctly shaped arrays."""

    def setUp(self):
        n = 5
        W = np.eye(n) * 0.5 + 0.1 * np.ones((n, n)) / n
        self.W = W / W.sum(axis=1, keepdims=True)
        self.x0 = np.random.default_rng(0).uniform(-1, 1, size=n)
        self.n = n

    def test_output_shape(self):
        pred = degroot_rollout_prediction(self.W, self.x0, 10)
        # returns x0 + 10 steps = 11 rows
        self.assertEqual(np.asarray(pred).shape, (11, self.n))

    def test_horizon_one(self):
        pred = degroot_rollout_prediction(self.W, self.x0, 1)
        # returns x0 + 1 step = 2 rows
        self.assertEqual(np.asarray(pred).shape, (2, self.n))

    def test_first_step_equals_W_times_x0(self):
        pred = degroot_rollout_prediction(self.W, self.x0, 1)
        # pred[0] is x0, pred[1] is W @ x0
        np.testing.assert_allclose(np.asarray(pred)[1], self.W @ self.x0, atol=1e-10)


class TestRolloutRecovery(unittest.TestCase):
    """Rollout with identity W should return x0 repeated."""

    def test_identity_w_no_change(self):
        n = 4
        W = np.eye(n, dtype=float)
        x0 = np.array([0.1, -0.5, 0.3, 0.8])
        pred = np.asarray(degroot_rollout_prediction(W, x0, horizon=5))
        # pred has 6 rows: x0 plus 5 steps, all equal to x0
        self.assertEqual(pred.shape, (6, n))
        for t in range(6):
            np.testing.assert_allclose(pred[t], x0, atol=1e-10)

    def test_rollout_noiseless_matches_simulation(self):
        rng = np.random.default_rng(600)
        n, gamma_true = 5, 0.6
        nbrs = _all_to_all_neighbors(n)
        Abar = build_expected_message_matrix(nbrs, n)
        W = gamma_true * Abar + (1.0 - gamma_true) * np.eye(n)
        x0 = rng.uniform(-1, 1, size=n)
        horizon = 8

        x = x0.copy()
        expected = []
        for _ in range(horizon):
            x = W @ x
            expected.append(x.copy())
        expected = np.asarray(expected)

        pred = np.asarray(degroot_rollout_prediction(W, x0, horizon))
        # pred[0] is x0; pred[1:] are the horizon steps
        np.testing.assert_allclose(pred[1:], expected, atol=1e-10)


if __name__ == "__main__":
    unittest.main()

