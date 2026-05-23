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

def _row_normalize(w):
    w = np.asarray(w, dtype=float)
    row_sums = w.sum(axis=1, keepdims=True)
    out = np.zeros_like(w)
    valid = row_sums[:, 0] > 0.0
    out[valid] = w[valid] / row_sums[valid]
    # raise error if any row sums are zero, since that would indicate an invalid W matrix
    if not np.all(valid):
        raise ValueError("Row-normalization failed: some rows sum to zero, indicating invalid W matrix.")
    return out

def _make_W(Abar, gamma):
    return _row_normalize(gamma * Abar + (1.0 - gamma) * np.eye(Abar.shape[0]))

def _sim_base_fj(rng, Abar, lambda1, gamma, n_runs, horizon, noise_std=0.0):
    """x_{t+1} = lambda1*x0 + (1-lambda1)*W @ x_t"""
    n = Abar.shape[0]
    alpha = 1.0 - lambda1
    W = _make_W(Abar, gamma)
    run_traj = {}
    for r in range(n_runs):
        x0 = rng.uniform(-1.0, 1.0, size=n)
        x = x0.copy()
        states = [x0.copy()]
        for _ in range(horizon):
            x = lambda1 * x0 + alpha * (W @ x)
            if noise_std > 0:
                x += noise_std * rng.normal(size=n)
            states.append(x.copy())
        run_traj[f'run_{r:02d}'] = np.asarray(states)
    return run_traj

def _sim_fj(rng, Abar, lambda1, lambda2, gamma, bias, n_runs, horizon, noise_std=0.0):
    """x_{t+1} = lambda1*x0 + lambda2*bias + (1-lambda1-lambda2)*W @ x_t"""
    n = Abar.shape[0]
    alpha = 1.0 - lambda1 - lambda2
    W = _make_W(Abar, gamma)
    run_traj = {}
    for r in range(n_runs):
        x0 = rng.uniform(-1.0, 1.0, size=n)
        x = x0.copy()
        states = [x0.copy()]
        for _ in range(horizon):
            x = lambda1 * x0 + lambda2 * bias + alpha * (W @ x)
            if noise_std > 0:
                x += noise_std * rng.normal(size=n)
            states.append(x.copy())
        run_traj[f'run_{r:02d}'] = np.asarray(states)
    return run_traj


# ===========================================================================
# Output contract
# ===========================================================================

class TestOutputContract(unittest.TestCase):
    """Both joint solvers return well-formed dicts."""

    def test_base_fj_contract(self):
        rng = np.random.default_rng(0)
        n, nbrs = 5, _all_to_all(5)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _sim_base_fj(rng, Abar, 0.2, 0.6, n_runs=4, horizon=15)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        for key in ('lambda1', 'alpha', 'gamma', 'mse_pool', 'W_blocks', 'Abar_blocks'):
            self.assertIn(key, fit)
        self.assertGreaterEqual(fit['lambda1'], 0.0)
        self.assertLessEqual(fit['lambda1'] + fit['alpha'], 1.0 + 1e-6)
        self.assertGreaterEqual(fit['mse_pool'], 0.0)
        for w in fit['W_blocks'].values():
            np.testing.assert_allclose(
                np.asarray(w).sum(axis=1), np.ones(np.asarray(w).shape[0]), atol=1e-6
            )

    def test_full_fj_contract(self):
        rng = np.random.default_rng(1)
        n, nbrs = 5, _all_to_all(5)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _sim_fj(rng, Abar, 0.2, 0.1, 0.6, 0.1, n_runs=4, horizon=15)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        for key in ('lambda1', 'lambda2', 'alpha', 'gamma', 'bias', 'mse_pool',
                    'W_blocks', 'Abar_blocks'):
            self.assertIn(key, fit)
        total = fit['lambda1'] + fit['lambda2'] + fit['alpha']
        self.assertAlmostEqual(total, 1.0, delta=1e-4)
        self.assertGreaterEqual(fit['mse_pool'], 0.0)
        self.assertEqual(len(fit['W_blocks']), 4)


# ===========================================================================
# Parameter recovery: Base-FJ
# ===========================================================================

class TestBaseFJRecovery(unittest.TestCase):
    """Noiseless (lambda1, gamma) recovery across topologies via subTest."""

    CASES = [
        # (name, topo_fn, n, lambda1, gamma, n_runs, horizon)
        ('complete', _all_to_all, 6, 0.2, 0.5, 8,  20),
        ('ring',     _ring,       8, 0.3, 0.7, 8,  25),
        ('star',     _star,       7, 0.25,0.6, 8,  25),
        ('chain',    _chain,      8, 0.3, 0.55,10, 25),
    ]

    def test_topology_sweep(self):
        for name, topo_fn, n, l1, gam, n_runs, horizon in self.CASES:
            with self.subTest(topology=name):
                rng = np.random.default_rng(hash(name) % (2**31))
                nbrs = topo_fn(n)
                Abar = build_expected_message_matrix(nbrs, n)
                run_traj = _sim_base_fj(rng, Abar, l1, gam, n_runs, horizon)
                run_neighbors = {rn: nbrs for rn in run_traj}
                fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
                self.assertAlmostEqual(fit['lambda1'], l1,  delta=1e-4)
                self.assertAlmostEqual(fit['gamma'],   gam, delta=1e-4)
                self.assertLess(fit['mse_pool'], 1e-8)


    def test_random_sparse_sweep(self):
        cases = [
            (50,  8,  1, 0.2, 0.5),
            (51,  8,  2, 0.3, 0.7),
            (52,  12, 3, 0.15,0.6),
            (53,  10, 5, 0.25,0.4),
        ]
        for seed, n, deg, l1, gam in cases:
            with self.subTest(seed=seed, n=n, in_degree=deg):
                rng = np.random.default_rng(seed)
                nbrs = _random_sparse(n, deg, rng)
                Abar = build_expected_message_matrix(nbrs, n)
                run_traj = _sim_base_fj(rng, Abar, l1, gam, n_runs=8, horizon=25)
                run_neighbors = {rn: nbrs for rn in run_traj}
                fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
                self.assertAlmostEqual(fit['lambda1'], l1,  delta=1e-4)
                self.assertAlmostEqual(fit['gamma'],   gam, delta=1e-4)
                self.assertLess(fit['mse_pool'], 1e-8)

    def test_per_run_different_graphs(self):
        """Half runs on ring, half on complete; shared (lambda1, gamma) recovered."""
        rng = np.random.default_rng(62)
        n, l1, gam = 6, 0.2, 0.6
        run_traj, run_neighbors = {}, {}
        for r in range(8):
            nbrs = _ring(n) if r % 2 == 0 else _all_to_all(n)
            Abar = build_expected_message_matrix(nbrs, n)
            W = _make_W(Abar, gam)
            x0 = rng.uniform(-1.0, 1.0, size=n)
            x, states = x0.copy(), [x0.copy()]
            for _ in range(20):
                x = l1 * x0 + (1 - l1) * (W @ x)
                states.append(x.copy())
            key = f'run_{r:02d}'
            run_traj[key] = np.asarray(states)
            run_neighbors[key] = nbrs
        fit = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['lambda1'], l1,  delta=1e-4)
        self.assertAlmostEqual(fit['gamma'],   gam, delta=1e-4)

# READ UP TO HERE BEFORE SUGGESTING CHANGES. The rest of the file is new and should be reviewed in its entirety, not line-by-line.

# ===========================================================================
# Parameter recovery: Full FJ
# ===========================================================================

class TestFullFJRecovery(unittest.TestCase):
    """Noiseless (lambda1, lambda2, gamma, bias) recovery across topologies."""

    CASES = [
        # (name, topo_fn, n, l1, l2, gam, bias, n_runs, horizon)
        ('complete', _all_to_all, 6, 0.2, 0.1, 0.6, 0.1,  8,  20),
        ('ring',     _ring,       8, 0.2, 0.1, 0.4, 0.0,  10, 25),
        ('star',     _star,       7, 0.2, 0.1, 0.6, 0.1,  8,  25),
        ('chain',    _chain,      8, 0.2, 0.1, 0.55,0.0,  10, 25),
    ]

    def test_topology_sweep(self):
        for name, topo_fn, n, l1, l2, gam, bias, n_runs, horizon in self.CASES:
            with self.subTest(topology=name):
                rng = np.random.default_rng(hash(name + 'fj') % (2**31))
                nbrs = topo_fn(n)
                Abar = build_expected_message_matrix(nbrs, n)
                run_traj = _sim_fj(rng, Abar, l1, l2, gam, bias, n_runs, horizon)
                run_neighbors = {rn: nbrs for rn in run_traj}
                fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
                self.assertAlmostEqual(fit['lambda1'], l1,   delta=1e-4)
                self.assertAlmostEqual(fit['lambda2'], l2,   delta=1e-4)
                self.assertAlmostEqual(fit['gamma'],   gam,  delta=1e-4)
                self.assertAlmostEqual(fit['bias'],    bias, delta=1e-4)
                self.assertLess(fit['mse_pool'], 1e-7)

    def test_random_sparse_sweep(self):
        cases = [
            (140, 8,  1, 0.2, 0.1, 0.5, 0.0),
            (141, 8,  2, 0.2, 0.1, 0.7, 0.1),
            (142, 12, 3, 0.15,0.1, 0.6, 0.0),
            (143, 10, 5, 0.2, 0.1, 0.4, 0.0),
        ]
        for seed, n, deg, l1, l2, gam, bias in cases:
            with self.subTest(seed=seed, n=n, in_degree=deg):
                rng = np.random.default_rng(seed)
                nbrs = _random_sparse(n, deg, rng)
                Abar = build_expected_message_matrix(nbrs, n)
                run_traj = _sim_fj(rng, Abar, l1, l2, gam, bias, n_runs=8, horizon=25)
                run_neighbors = {rn: nbrs for rn in run_traj}
                fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
                self.assertAlmostEqual(fit['lambda1'], l1,  delta=1e-4)
                self.assertAlmostEqual(fit['lambda2'], l2,   delta=1e-4)
                self.assertAlmostEqual(fit['gamma'],   gam, delta=1e-4)
                self.assertAlmostEqual(fit['bias'],    bias, delta=1e-4)
                self.assertLess(fit['mse_pool'], 1e-7)

    def test_negative_bias(self):
        rng = np.random.default_rng(100)
        n, nbrs = 6, _all_to_all(6)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _sim_fj(rng, Abar, 0.15, 0.1, 0.7, -0.2, n_runs=8, horizon=20)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['bias'], -0.2, delta=1e-4)

    def test_per_run_different_graphs(self):
        """Half runs on ring, half on complete; shared (lambda1, lambda2, gamma, bias) recovered."""
        rng = np.random.default_rng(162)
        n, l1, l2, gam, bias = 16, 0.2, 0.1, 0.6, 0.1
        run_traj, run_neighbors = {}, {}
        for r in range(8):
            nbrs = _ring(n) if r % 2 == 0 else _all_to_all(n)
            Abar = build_expected_message_matrix(nbrs, n)
            traj = _sim_fj(rng, Abar, l1, l2, gam, bias, n_runs=1, horizon=20)
            key = f'run_{r:02d}'
            run_traj[key] = traj['run_00']
            run_neighbors[key] = nbrs
        fit = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['lambda1'], l1,  delta=1e-4)
        self.assertAlmostEqual(fit['lambda2'], l2,  delta=1e-4)
        self.assertAlmostEqual(fit['gamma'],   gam, delta=1e-4)
        self.assertAlmostEqual(fit['bias'],    bias, delta=1e-4)
        self.assertLess(fit['mse_pool'], 1e-7)

# ===========================================================================
# Grid-search selectors vs joint solvers
# ===========================================================================

class TestGridSearchSelectors(unittest.TestCase):
    """Check that grid search methods match up to the grid space tolerance"""

    GRID = np.linspace(0.0, 1.0, 26)   # step = 0.01
    GRID_STEP = 1.0 / (len(GRID) - 1)

    # test on random graphs
    def _base_fj_data(self, seed=200):
        rng = np.random.default_rng(seed)
        n, nbrs = 15, _random_sparse(15, 3, rng)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _sim_base_fj(rng, Abar, 0.2, 0.6, n_runs=8, horizon=20)
        return run_traj, {rn: nbrs for rn in run_traj}

    def _fj_data(self, seed=210):
        rng = np.random.default_rng(seed)
        n, nbrs = 15, _random_sparse(15, 3, rng)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _sim_fj(rng, Abar, 0.2, 0.1, 0.6, 0.0, n_runs=8, horizon=20)
        return run_traj, {rn: nbrs for rn in run_traj}

    def test_base_fj_grid_lambda_within_step_of_joint(self):
        run_traj, run_neighbors = self._base_fj_data(seed=202)
        grid_best, _ = select_base_friedkin_johnsen_adjacency_lambda(
            run_traj, run_neighbors, lambda_grid=self.GRID)
        joint = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(grid_best['lambda1'], joint['lambda1'],
                               delta=self.GRID_STEP + 1e-6)

    def test_base_fj_no_grid_falls_back_to_joint(self):
        run_traj, run_neighbors = self._base_fj_data(seed=203)
        best, _ = select_base_friedkin_johnsen_adjacency_lambda(
            run_traj, run_neighbors, lambda_grid=None)
        joint = fit_base_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(best['lambda1'], joint['lambda1'], delta=1e-4)
        self.assertAlmostEqual(best['gamma'],   joint['gamma'],   delta=1e-4)

    def test_fj_grid_lambdas_within_step_of_joint(self):
        run_traj, run_neighbors = self._fj_data(seed=212)
        grid_best, _ = select_friedkin_johnsen_adjacency_lambdas(
            run_traj, run_neighbors, lambda_grid=self.GRID)
        joint = fit_friedkin_johnsen_adjacency_joint(run_traj, run_neighbors)
        self.assertAlmostEqual(grid_best['lambda1'], joint['lambda1'],
                               delta=self.GRID_STEP + 1e-6)
        self.assertAlmostEqual(grid_best['lambda2'], joint['lambda2'],
                               delta=self.GRID_STEP + 1e-6)


if __name__ == '__main__':
    unittest.main()
