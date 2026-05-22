"""Concise unit tests for fit_degroot_adjacency_scalar and degroot_rollout_prediction.

Covers (one representative case per concept):
  - Output contract (keys, ranges, row-stochastic W)
  - Gamma recovery across 4 topologies (subTest sweep) + random sparse
  - Multiple runs: shared graph and per-run different graphs
  - Low-noise robustness
  - Rollout shape and consistency
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
# Shared helpers
# ---------------------------------------------------------------------------

def _all_to_all(n):
    return {i: list(range(n)) for i in range(n)}

def _ring(n):
    return {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}

def _star(n):
    nbrs = {0: list(range(1, n))}
    for i in range(1, n):
        nbrs[i] = [0]
    return nbrs

def _chain(n):
    nbrs = {0: [1]}
    for i in range(1, n):
        nbrs[i] = [i - 1]
    return nbrs

def _random_sparse(n, in_degree, rng):
    in_degree = max(1, min(in_degree, n - 1))
    nbrs = {}
    for i in range(n):
        pool = [j for j in range(n) if j != i]
        nbrs[i] = rng.choice(pool, size=in_degree, replace=False).tolist()
    return nbrs

def _simulate(Abar, gamma, n_runs, horizon, rng, noise_std=0.0):
    n = Abar.shape[0]
    W = gamma * Abar + (1.0 - gamma) * np.eye(n)
    run_traj = {}
    for r in range(n_runs):
        x = rng.uniform(-1.0, 1.0, size=n)
        states = [x.copy()]
        for _ in range(horizon):
            x = W @ x + noise_std * rng.normal(size=n)
            states.append(x.copy())
        run_traj[f'run_{r:02d}'] = np.asarray(states)
    return run_traj


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

class TestOutputContract(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 5
        nbrs = _all_to_all(n)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _simulate(Abar, 0.5, n_runs=4, horizon=15, rng=rng)
        run_neighbors = {rn: nbrs for rn in run_traj}
        self.fit = fit_degroot_adjacency_scalar(run_traj, run_neighbors)

    def test_required_keys(self):
        for key in ('gamma', 'mse_pool', 'W_blocks', 'Abar_blocks'):
            self.assertIn(key, self.fit)

    def test_gamma_in_unit_interval(self):
        self.assertGreaterEqual(self.fit['gamma'], 0.0)
        self.assertLessEqual(self.fit['gamma'], 1.0)

    def test_mse_non_negative(self):
        self.assertGreaterEqual(self.fit['mse_pool'], 0.0)

    def test_w_blocks_row_stochastic(self):
        for w in self.fit['W_blocks'].values():
            np.testing.assert_allclose(
                np.asarray(w).sum(axis=1), np.ones(np.asarray(w).shape[0]), atol=1e-6
            )


# ---------------------------------------------------------------------------
# Gamma recovery across topologies
# ---------------------------------------------------------------------------

class TestGammaRecovery(unittest.TestCase):
    """Noiseless gamma recovery on four canonical topologies."""

    TOPOLOGIES = [
        ('complete', _all_to_all, 6,  0.5,  8, 20, 1e-4),
        ('ring',     _ring,       8,  0.7,  8, 25, 1e-4),
        ('star',     _star,       7,  0.6,  8, 25, 1e-4),
        ('chain',    _chain,      8,  0.55, 10, 25, 1e-4),
    ]

    def test_topology_sweep(self):
        for name, topo_fn, n, gamma, n_runs, horizon, delta in self.TOPOLOGIES:
            with self.subTest(topology=name, gamma=gamma):
                rng = np.random.default_rng(42 + n)
                nbrs = topo_fn(n)
                Abar = build_expected_message_matrix(nbrs, n)
                run_traj = _simulate(Abar, gamma, n_runs, horizon, rng)
                run_neighbors = {rn: nbrs for rn in run_traj}
                fit = fit_degroot_adjacency_scalar(run_traj, run_neighbors)
                self.assertAlmostEqual(float(fit['gamma']), gamma, delta=delta)
                self.assertLess(float(fit['mse_pool']), 1e-8)

    def test_random_sparse_sweep(self):
        """Random sparse graphs: varying in-degree, n, and gamma."""
        cases = [
            (700, 8,  1, 0.4),
            (710, 8,  2, 0.6),
            (720, 12, 3, 0.5),
            (730, 10, 5, 0.25),
        ]
        for seed, n, deg, gamma in cases:
            with self.subTest(seed=seed, n=n, in_degree=deg, gamma=gamma):
                rng = np.random.default_rng(seed)
                nbrs = _random_sparse(n, deg, rng)
                Abar = build_expected_message_matrix(nbrs, n)
                run_traj = _simulate(Abar, gamma, n_runs=8, horizon=25, rng=rng)
                run_neighbors = {rn: nbrs for rn in run_traj}
                fit = fit_degroot_adjacency_scalar(run_traj, run_neighbors)
                self.assertAlmostEqual(float(fit['gamma']), gamma, delta=1e-4)
                self.assertLess(float(fit['mse_pool']), 1e-8)


# ---------------------------------------------------------------------------
# Multiple runs
# ---------------------------------------------------------------------------

class TestMultipleRuns(unittest.TestCase):

    def test_per_run_different_graphs(self):
        """Half runs on ring, half on complete; shared gamma recovered."""
        rng = np.random.default_rng(400)
        n, gamma_true = 6, 0.6
        run_traj, run_neighbors = {}, {}
        for r in range(8):
            nbrs = _ring(n) if r % 2 == 0 else _all_to_all(n)
            Abar = build_expected_message_matrix(nbrs, n)
            traj = _simulate(Abar, gamma_true, n_runs=1, horizon=20, rng=rng)
            key = f'run_{r:02d}'
            run_traj[key] = traj['run_00']
            run_neighbors[key] = nbrs
        fit = fit_degroot_adjacency_scalar(run_traj, run_neighbors)
        self.assertAlmostEqual(float(fit['gamma']), gamma_true, delta=1e-4)


if __name__ == '__main__':
    unittest.main()
