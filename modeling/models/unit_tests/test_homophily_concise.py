import unittest
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.models.adjacency_based.homophily import (
    fit_homophily,
    fit_homophily_friedkin_johnsen,
    fit_homophily_stubborness,
    rollout_with_homophily,
    rollout_with_homophily_friedkin_johnsen,
    rollout_with_homophily_stubborness,
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

def _W_homophily(Abar, x, gamma):
    diff = np.abs(x[:, None] - x[None, :])
    raw = Abar * np.exp(-gamma * diff)
    row_sums = raw.sum(axis=1, keepdims=True)
    out = np.zeros_like(raw)
    valid = row_sums[:, 0] > 0
    out[valid] = raw[valid] / row_sums[valid]
    return out

def _sim_hom(rng, Abar, gamma, lambda_self, n_runs, horizon):
    """x_{t+1} = lambda_self*x_t + (1-lambda_self)*W_t @ x_t"""
    n = Abar.shape[0]
    alpha = 1.0 - lambda_self
    run_traj = {}
    for r in range(n_runs):
        x = rng.uniform(-1.0, 1.0, size=n)
        states = [x.copy()]
        for _ in range(horizon):
            x = lambda_self * x + alpha * (_W_homophily(Abar, x, gamma) @ x)
            states.append(x.copy())
        run_traj[f'run_{r:02d}'] = np.asarray(states)
    return run_traj

def _sim_hom_fj(rng, Abar, gamma, lambda_self, lambda1, n_runs, horizon):
    """x_{t+1} = lambda_self*x_t + lambda1*x0 + alpha*W_t @ x_t"""
    n = Abar.shape[0]
    alpha = 1.0 - lambda_self - lambda1
    run_traj = {}
    for r in range(n_runs):
        x0 = rng.uniform(-1.0, 1.0, size=n)
        x = x0.copy()
        states = [x0.copy()]
        for _ in range(horizon):
            x = lambda_self * x + lambda1 * x0 + alpha * (_W_homophily(Abar, x, gamma) @ x)
            states.append(x.copy())
        run_traj[f'run_{r:02d}'] = np.asarray(states)
    return run_traj

def _sim_hom_stub(rng, Abar, gamma, lambda_self, lambda1, lambda2, bias, n_runs, horizon):
    """x_{t+1} = lambda_self*x_t + lambda1*x0 + lambda2*bias + alpha*W_t @ x_t"""
    n = Abar.shape[0]
    alpha = 1.0 - lambda_self - lambda1 - lambda2
    run_traj = {}
    for r in range(n_runs):
        x0 = rng.uniform(-1.0, 1.0, size=n)
        x = x0.copy()
        states = [x0.copy()]
        for _ in range(horizon):
            x = lambda_self * x + lambda1 * x0 + lambda2 * bias + alpha * (_W_homophily(Abar, x, gamma) @ x)
            states.append(x.copy())
        run_traj[f'run_{r:02d}'] = np.asarray(states)
    return run_traj


# ===========================================================================
# Output contract
# ===========================================================================

class TestOutputContract(unittest.TestCase):
    """All three fitters return well-formed dicts."""

    def _base_data(self, seed=0, gamma=1.0, lambda_self=0.3):
        rng = np.random.default_rng(seed)
        n, nbrs = 5, _all_to_all(5)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _sim_hom(rng, Abar, gamma, lambda_self, n_runs=4, horizon=8)
        return run_traj, {rn: nbrs for rn in run_traj}

    def test_plain_homophily_contract(self):
        run_traj, run_neighbors = self._base_data()
        fit = fit_homophily(run_traj, run_neighbors)
        for key in ('gamma', 'lambda_self', 'alpha', 'mse_pool',
                    'gamma_grid', 'gamma_objective_map', 'success'):
            self.assertIn(key, fit)
        self.assertGreater(fit['gamma'], 0.0)
        self.assertGreaterEqual(fit['lambda_self'], 0.0)
        self.assertLessEqual(fit['lambda_self'], 1.0)
        self.assertAlmostEqual(fit['lambda_self'] + fit['alpha'], 1.0, delta=1e-5)
        self.assertGreaterEqual(fit['mse_pool'], 0.0)

    def test_fj_homophily_contract(self):
        rng = np.random.default_rng(1)
        n, nbrs = 5, _all_to_all(5)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _sim_hom_fj(rng, Abar, 1.0, 0.3, 0.2, n_runs=4, horizon=8)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_homophily_friedkin_johnsen(run_traj, run_neighbors)
        for key in ('gamma', 'lambda_self', 'lambda1', 'alpha', 'mse_pool', 'success'):
            self.assertIn(key, fit)
        total = fit['lambda_self'] + fit['lambda1'] + fit['alpha']
        self.assertAlmostEqual(total, 1.0, delta=1e-4)

    def test_stubborness_homophily_contract(self):
        rng = np.random.default_rng(2)
        n, nbrs = 5, _all_to_all(5)
        Abar = build_expected_message_matrix(nbrs, n)
        run_traj = _sim_hom_stub(rng, Abar, 1.0, 0.2, 0.15, 0.1, 0.0, n_runs=4, horizon=8)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_homophily_stubborness(run_traj, run_neighbors)
        for key in ('gamma', 'lambda_self', 'lambda1', 'lambda2', 'bias', 'alpha', 'mse_pool'):
            self.assertIn(key, fit)
        total = fit['lambda_self'] + fit['lambda1'] + fit['lambda2'] + fit['alpha']
        self.assertAlmostEqual(total, 1.0, delta=1e-4)
        bias = fit['bias']
        # check bias is in [-1,1]
        self.assertGreaterEqual(bias, -1.0)
        self.assertLessEqual(bias, 1.0)


# ===========================================================================
# Plain Homophily: gamma recovery
# ===========================================================================

class TestPlainHomophilyGammaRecovery(unittest.TestCase):
    """Noiseless gamma recovery across topologies and gamma magnitudes."""

    TOPOLOGIES = [
        # (name, topo_fn, n, gamma, lambda_self, n_runs, horizon, abs_delta, rel_tol)
        ('complete', _all_to_all, 6, 1.0, 0.3, 8,  10, 1e-04, 0.01),
        ('ring',     _ring,       8, 1.5, 0.3, 8,  12, 1e-04, 0.01),
        ('star',     _star,       7, 1.0, 0.3, 8,  12, 1e-04, 0.01),
        ('chain',   _chain,        8, 1.0, 0.3, 8,  12, 1e-04, 0.01),
    ]

    def test_topology_sweep(self):
        for name, topo_fn, n, gamma, ls, n_runs, horizon, delta, rel_tol in self.TOPOLOGIES:
            with self.subTest(topology=name):
                rng = np.random.default_rng(hash(name) % (2**31))
                nbrs = topo_fn(n)
                Abar = build_expected_message_matrix(nbrs, n)
                run_traj = _sim_hom(rng, Abar, gamma, ls, n_runs, horizon)
                run_neighbors = {rn: nbrs for rn in run_traj}
                fit = fit_homophily(run_traj, run_neighbors)
                if not name == 'chain':     # in chain, every neighbor has a single neighor, so gamma is not identifiable 
                    self.assertAlmostEqual(fit['gamma'], gamma, delta=delta)
                    self.assertLess(abs(fit['gamma'] - gamma) / gamma, rel_tol)
                self.assertAlmostEqual(fit['lambda_self'], ls, delta=delta)
                self.assertLess(fit['mse_pool'], 1e-7)

    def test_random_sparse(self):
        rng = np.random.default_rng(123)
        n, in_degree = 16, 3
        nbrs = _random_sparse(n, in_degree, rng)
        Abar = build_expected_message_matrix(nbrs, n)
        gamma, lambda_self = 1.0, 0.3
        run_traj = _sim_hom(rng, Abar, gamma, lambda_self, n_runs=8, horizon=12)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_homophily(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['gamma'], gamma, delta=1e-3)
        self.assertLess(abs(fit['gamma'] - gamma) / gamma, 0.01)
        self.assertAlmostEqual(fit['lambda_self'], lambda_self, delta=1e-4)
        self.assertLess(fit['mse_pool'], 1e-7)

    def test_per_run_different_graphs(self):
        """Half runs on ring, half on complete; """
        rng = np.random.default_rng(400)
        n, gamma, lambda_self = 16, 1.0, 0.3
        run_traj, run_neighbors = {}, {}
        for r in range(8):
            if r % 2 == 0:
                nbrs = _ring(n)
            else:
                nbrs = _all_to_all(n)
            Abar = build_expected_message_matrix(nbrs, n)
            run_traj[f'run_{r:02d}'] = _sim_hom(rng, Abar, gamma, lambda_self, n_runs=1, horizon=12)[f'run_00']
            run_neighbors[f'run_{r:02d}'] = nbrs
        fit = fit_homophily(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['gamma'], gamma, delta=1e-4)
        self.assertLess(abs(fit['gamma'] - gamma) / gamma, 0.01)
        self.assertAlmostEqual(fit['lambda_self'], lambda_self, delta=1e-4)
        self.assertLess(fit['mse_pool'], 1e-7)

    def test_gamma_range_sweep(self):
        """Gamma recovery across several decades: [0.1, 1.0, 10.0]."""
        gammas = [0.1, 1.0, 10.0]
        lambda_self = 0.3
        rng = np.random.default_rng(99)
        n, nbrs = 6, _all_to_all(6)
        Abar = build_expected_message_matrix(nbrs, n)
        for gamma in gammas:
            with self.subTest(gamma=gamma):
                run_traj = _sim_hom(rng, Abar, gamma, lambda_self, n_runs=8, horizon=10)
                run_neighbors = {rn: nbrs for rn in run_traj}
                fit = fit_homophily(run_traj, run_neighbors)
                rel_err = abs(fit['gamma'] - gamma) / gamma
                self.assertLess(rel_err, 0.01,
                    msg=f"gamma={gamma}: relative error {rel_err:.3f}")
                self.assertAlmostEqual(fit['lambda_self'], lambda_self, delta=1e-4)


# ===========================================================================
# FJ Homophily: parameter recovery
# ===========================================================================

class TestFJHomophilyRecovery(unittest.TestCase):
    """Noiseless recovery for Homophily-FJ across topologies."""

    TOPOLOGIES = [
        # (name, topo_fn, n, gamma, lambda_self, lambda1, n_runs, horizon, abs_delta, rel_tol)
        ('complete', _all_to_all, 6, 1.0, 0.3, 0.2, 8,  20, 1e-04, 0.01),
        ('ring',     _ring,       8, 1.5, 0.3, 0.2, 8,  25, 1e-04, 0.01),
        ('star',     _star,       7, 1.0, 0.3, 0.2, 8,  25, 1e-04, 0.01),
        ('chain',   _chain,        8, 1.0, 0.3, 0.2, 8,  25, 1e-04, 0.01),
    ]

    def test_topology_sweep(self):
        for name, topo_fn, n, gamma, ls, l1, n_runs, horizon, delta, rel_tol in self.TOPOLOGIES:
            with self.subTest(topology=name):
                rng = np.random.default_rng(hash(name + 'fj') % (2**31))
                nbrs = topo_fn(n)
                Abar = build_expected_message_matrix(nbrs, n)
                run_traj = _sim_hom_fj(rng, Abar, gamma, ls, l1, n_runs, horizon)
                run_neighbors = {rn: nbrs for rn in run_traj}
                fit = fit_homophily_friedkin_johnsen(run_traj, run_neighbors)
                if not name == 'chain':     # in chain, every neighbor has a single neighor, so gamma is not identifiable 
                    self.assertAlmostEqual(fit['gamma'], gamma, delta=delta)
                    self.assertLess(abs(fit['gamma'] - gamma) / gamma, rel_tol)
                self.assertAlmostEqual(fit['lambda_self'], ls, delta=delta)
                self.assertAlmostEqual(fit['lambda1'], l1, delta=delta)
                self.assertLess(fit['mse_pool'], 1e-7)

    def test_random_sparse(self):
        rng = np.random.default_rng(1234)
        n, in_degree = 16, 3
        nbrs = _random_sparse(n, in_degree, rng)
        Abar = build_expected_message_matrix(nbrs, n)
        gamma, lambda_self, lambda1 = 1.0, 0.3, 0.2
        run_traj = _sim_hom_fj(rng, Abar, gamma, lambda_self, lambda1, n_runs=8, horizon=12)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_homophily_friedkin_johnsen(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['gamma'], gamma, delta=1e-3)
        self.assertLess(abs(fit['gamma'] - gamma) / gamma, 0.01)
        self.assertAlmostEqual(fit['lambda_self'], lambda_self, delta=1e-4)
        self.assertAlmostEqual(fit['lambda1'], lambda1, delta=1e-4)
        self.assertLess(fit['mse_pool'], 1e-7)

    def test_per_run_different_graphs(self):
        """Half runs on ring, half on complete; shared gamma and lambdas recovered."""
        rng = np.random.default_rng(400)
        n, gamma, lambda_self, lambda1 = 16, 1.0, 0.3, 0.2
        run_traj, run_neighbors = {}, {}
        for r in range(8):
            if r % 2 == 0:
                nbrs = _ring(n)
            else:
                nbrs = _all_to_all(n)
            Abar = build_expected_message_matrix(nbrs, n)
            run_traj[f'run_{r:02d}'] = _sim_hom_fj(rng, Abar, gamma, lambda_self, lambda1, n_runs=1, horizon=25)[f'run_00']
            run_neighbors[f'run_{r:02d}'] = nbrs
        fit = fit_homophily_friedkin_johnsen(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['gamma'], gamma, delta=1e-4)
        self.assertLess(abs(fit['gamma'] - gamma) / gamma, 0.01)
        self.assertAlmostEqual(fit['lambda_self'], lambda_self, delta=1e-4)
        self.assertAlmostEqual(fit['lambda1'], lambda1, delta=1e-4)
        self.assertLess(fit['mse_pool'], 1e-7)

    def test_gamma_range_sweep(self):
        """Gamma recovery across several decades: [0.1, 1.0, 10.0]."""
        gammas = [0.1, 1.0, 10.0]
        lambda_self, lambda1 = 0.3, 0.2
        rng = np.random.default_rng(99)
        n, nbrs = 6, _all_to_all(6)
        Abar = build_expected_message_matrix(nbrs, n)
        lambda_self, lambda1 = 0.3, 0.2
        for gamma in gammas:
            with self.subTest(gamma=gamma):
                run_traj = _sim_hom_fj(rng, Abar, gamma, lambda_self, lambda1, n_runs=8, horizon=20)
                run_neighbors = {rn: nbrs for rn in run_traj}
                fit = fit_homophily_friedkin_johnsen(run_traj, run_neighbors)
                rel_err = abs(fit['gamma'] - gamma) / gamma
                self.assertLess(rel_err, 0.01,
                    msg=f"gamma={gamma}: relative error {rel_err:.3f}")
                self.assertAlmostEqual(fit['lambda_self'], lambda_self, delta=1e-4)
                self.assertAlmostEqual(fit['lambda1'], lambda1, delta=1e-4)

# ===========================================================================
# Stubbornness Homophily: parameter recovery
# ===========================================================================

class TestStubbornHomophilyRecovery(unittest.TestCase):
    """Noiseless (gamma, lambda_stub) recovery for Homophily-Stubbornness."""

    TOPOLOGIES = [
        # (name, topo_fn, n, gamma, lambda_self, lambda1, lambda2, bias, n_runs, horizon, abs_delta, rel_tol)
        ('complete', _all_to_all, 6, 1.0, 0.3, 0.2, 0.1, 0.3, 8,  10, 1e-04, 0.01),
        ('ring',     _ring,       8, 1.5, 0.3, 0.2, 0.1, -0.3, 8,  12, 1e-04, 0.01),
        ('star',     _star,       7, 1.0, 0.3, 0.2, 0.1, 0.2, 8,  12, 1e-04, 0.01),
        ('chain',   _chain,        8, 1.0, 0.3, 0.2, 0.1, -0.2, 8,  12, 1e-04, 0.01),
    ]

    def test_topology_sweep(self):
        for row in self.TOPOLOGIES:
            name, topo_fn, n, gamma, ls, l1, l2, bias, n_runs, horizon, delta, rel_tol = row
            with self.subTest(topology=name):
                rng = np.random.default_rng(hash(name + 'stub') % (2**31))
                nbrs = topo_fn(n)
                Abar = build_expected_message_matrix(nbrs, n)
                run_traj = _sim_hom_stub(rng, Abar, gamma, ls, l1, l2, bias, n_runs, horizon)
                run_neighbors = {rn: nbrs for rn in run_traj}
                fit = fit_homophily_stubborness(run_traj, run_neighbors)
                if not name == 'chain':     # in chain, every neighbor has a single neighor, so gamma is not identifiable 
                    self.assertAlmostEqual(fit['gamma'], gamma, delta=delta)
                    self.assertLess(abs(fit['gamma'] - gamma) / gamma, rel_tol)
                self.assertAlmostEqual(fit['lambda_self'], ls, delta=delta)
                self.assertAlmostEqual(fit['lambda1'], l1, delta=delta)
                self.assertAlmostEqual(fit['lambda2'], l2, delta=delta)
                self.assertAlmostEqual(fit['bias'], bias, delta=delta)
                self.assertLess(fit['mse_pool'], 1e-7)

    def test_random_sparse(self):
        rng = np.random.default_rng(12345)
        n, in_degree = 16, 3
        nbrs = _random_sparse(n, in_degree, rng)
        Abar = build_expected_message_matrix(nbrs, n)
        gamma, lambda_self, lambda1, lambda2, bias = 1.0, 0.3, 0.2, 0.1, 0.25
        run_traj = _sim_hom_stub(rng, Abar, gamma, lambda_self, lambda1, lambda2, bias, n_runs=8, horizon=12)
        run_neighbors = {rn: nbrs for rn in run_traj}
        fit = fit_homophily_stubborness(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['gamma'], gamma, delta=1e-3)
        self.assertLess(abs(fit['gamma'] - gamma) / gamma, 0.01)
        self.assertAlmostEqual(fit['lambda_self'], lambda_self, delta=1e-4)
        self.assertAlmostEqual(fit['lambda1'], lambda1, delta=1e-4)
        self.assertAlmostEqual(fit['lambda2'], lambda2, delta=1e-4)
        self.assertAlmostEqual(fit['bias'], bias, delta=1e-4)
        self.assertLess(fit['mse_pool'], 1e-7)

    def test_per_run_different_graphs(self):
        """Half runs on ring, half on complete; shared gamma and lambdas recovered."""
        rng = np.random.default_rng(400)
        n, gamma, lambda_self, lambda1, lambda2, bias = 16, 1.0, 0.3, 0.2, 0.1, 0.25
        run_traj, run_neighbors = {}, {}
        for r in range(8):
            if r % 2 == 0:
                nbrs = _ring(n)
            else:
                nbrs = _all_to_all(n)
            Abar = build_expected_message_matrix(nbrs, n)
            run_traj[f'run_{r:02d}'] = _sim_hom_stub(rng, Abar, gamma, lambda_self, lambda1, lambda2, bias, n_runs=1, horizon=12)[f'run_00']
            run_neighbors[f'run_{r:02d}'] = nbrs
        fit = fit_homophily_stubborness(run_traj, run_neighbors)
        self.assertAlmostEqual(fit['gamma'], gamma, delta=1e-3)
        self.assertLess(abs(fit['gamma'] - gamma) / gamma, 0.01)
        self.assertAlmostEqual(fit['lambda_self'], lambda_self, delta=1e-4)
        self.assertAlmostEqual(fit['lambda1'], lambda1, delta=1e-4)
        self.assertAlmostEqual(fit['lambda2'], lambda2, delta=1e-4)
        self.assertAlmostEqual(fit['bias'], bias, delta=1e-4)
        self.assertLess(fit['mse_pool'], 1e-7)

    def test_gamma_range_sweep(self):
        """Gamma recovery across several decades: [0.1, 1.0, 10.0]."""
        gammas = [0.1, 1.0, 10.0]
        lambda_self, lambda1, lambda2, bias = 0.3, 0.2, 0.1, -0.25
        rng = np.random.default_rng(99)
        n, nbrs = 6, _all_to_all(6)
        Abar = build_expected_message_matrix(nbrs, n)
        for gamma in gammas:
            with self.subTest(gamma=gamma):
                run_traj = _sim_hom_stub(rng, Abar, gamma, lambda_self, lambda1, lambda2, bias, n_runs=8, horizon=10)
                run_neighbors = {rn: nbrs for rn in run_traj}
                fit = fit_homophily_stubborness(run_traj, run_neighbors)
                rel_err = abs(fit['gamma'] - gamma) / gamma
                self.assertLess(rel_err, 0.01)
                self.assertAlmostEqual(fit['lambda_self'], lambda_self, delta=1e-4)
                self.assertAlmostEqual(fit['lambda1'], lambda1, delta=1e-4)
                self.assertAlmostEqual(fit['lambda2'], lambda2, delta=1e-4)
                self.assertAlmostEqual(fit['bias'], bias, delta=1e-4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
