import unittest
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.models.fixed_graph.friedkin_johnsen import fit_friedkin_johnsen_joint_traj0
from modeling.models.adjacency_based.friedkin_johnsen import friedkin_johnsen_adjacency_rollout


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_friedkin_johnsen_concise.py)
# ---------------------------------------------------------------------------

def _all_to_all(n):
    return {i: [j for j in range(n) if j != i] for i in range(n)}

def _ring(n):
    return {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}

def _star(n):
    nbrs = {0: list(range(1, n))}
    for i in range(1, n):
        nbrs[i] = [0]
    return nbrs

def _random_sparse(n, in_degree, rng):
    """Each agent follows exactly `in_degree` distinct others (no self-loops in neighbor list)."""
    in_degree = max(1, min(in_degree, n - 1))
    nbrs = {}
    for i in range(n):
        pool = [j for j in range(n) if j != i]
        nbrs[i] = rng.choice(pool, size=in_degree, replace=False).tolist()
    return nbrs


def _random_W_from_neighbors(neighbors, n, rng, include_self=True):
    """Build a random row-stochastic W consistent with `neighbors` (+ optional self-loop)."""
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        cols = list(neighbors[i])
        if include_self:
            cols = list(set(cols) | {i})
        weights = rng.exponential(scale=1.0, size=len(cols))
        weights /= weights.sum()
        for c, w in zip(cols, weights):
            W[i, c] = w
    return W

def _sim_fj_traj0(rng, W_true, lambda1, lambda2, bias, n_runs, horizon, noise_std=0.0):
    """Simulate  x_{t+1} = lambda1*x0 + lambda2*bias + alpha*(W_true @ x_t).

    Uses first row as traj[0] = x0, consistent with what fit_friedkin_johnsen_joint_traj0
    expects (it slices traj[0] for x0_pool).
    """
    n = W_true.shape[0]
    alpha = 1.0 - lambda1 - lambda2
    run_traj = {}
    for r in range(n_runs):
        x0 = rng.uniform(-1.0, 1.0, size=n)
        x = x0.copy()
        states = [x0.copy()]
        for _ in range(horizon):
            x = lambda1 * x0 + lambda2 * bias + alpha * (W_true @ x)
            if noise_std > 0.0:
                x = x + noise_std * rng.normal(size=n)
            states.append(x.copy())
        run_traj[f"run_{r:02d}"] = np.asarray(states, dtype=float)
    return run_traj


# ===========================================================================
# Recovery on noiseless synthetic data
# ===========================================================================

class TestRecoveryNoiseless(unittest.TestCase):
    """Fitter should recover planted parameters exactly when there is no noise."""

    TOL = 1e-3
    MSE_TOL = 1e-6

    def _make_run_traj_and_neighbors(self, n, neighbors, lambda1, lambda2, bias, W_true,
                                     n_runs=5, horizon=15, seed=42):
        rng = np.random.default_rng(seed)
        run_traj = _sim_fj_traj0(rng, W_true, lambda1, lambda2, bias, n_runs, horizon)
        run_neighbors = {rn: neighbors for rn in run_traj}
        return run_traj, run_neighbors

    def test_recovers_params_noiseless(self):
        """For each graph topology, should recover lambda1, lambda2, bias, and W to within
        tolerance and have near-zero training MSE on noiseless data."""

        lambda1_true = 0.3
        lambda2_true = 0.2
        bias_true    = 0.5   # scalar bias in [-1, 1]
        n            = 30
        n_runs       = 15
        horizon      = 20

        graph_cases = [
            ("all_to_all",    _all_to_all(n), True),
            ("ring",          _ring(n), True),
            ("star",          _star(n), True),
            ("random_sparse_deg2", _random_sparse(n, in_degree=2, rng=np.random.default_rng(10)), True),
            ("random_sparse_deg4", _random_sparse(n, in_degree=4, rng=np.random.default_rng(11)), True),
            ("random_sparse_deg6", _random_sparse(n, in_degree=6, rng=np.random.default_rng(12)), True),
            ("random_sparse_deg6_no_self", _random_sparse(n, in_degree=6, rng=np.random.default_rng(13)), False),
        ]

        for graph_name, neighbors, include_self in graph_cases:
            with self.subTest(graph=graph_name):
                W_true = _random_W_from_neighbors(
                    neighbors, n,
                    rng=np.random.default_rng(99),
                    include_self=include_self,
                )
                run_traj, run_neighbors = self._make_run_traj_and_neighbors(
                    n, neighbors, lambda1_true, lambda2_true, bias_true, W_true,
                    n_runs=n_runs, horizon=horizon, seed=42,
                )

                result = fit_friedkin_johnsen_joint_traj0(
                    run_traj, run_neighbors, eps=1e-6, turn_off_graph_constraints=False
                )

                # check correct ranges for lambdas and bias
                self.assertGreaterEqual(result['lambda1'], 0.0)
                self.assertGreaterEqual(result['lambda2'], 0.0)
                self.assertLessEqual(result['lambda1'] + result['lambda2'], 1.0 + self.TOL)
                self.assertLessEqual(result['b'], 1.0 + self.TOL)
                self.assertGreaterEqual(result['b'], -1.0 - self.TOL)
                # check row stochasticity and non-negativity of W
                W_est = result['W']
                self.assertTrue(np.all(W_est >= -self.TOL), "W has negative entries")
                np.testing.assert_allclose(W_est.sum(axis=1), 1.0, atol=self.TOL, err_msg="Rows of W do not sum to 1")

                self.assertAlmostEqual(result['lambda1'], lambda1_true, delta=self.TOL)
                self.assertAlmostEqual(result['lambda2'], lambda2_true, delta=self.TOL)
                self.assertAlmostEqual(result['b'], bias_true, delta=self.TOL)
                np.testing.assert_allclose(result['W'], W_true, atol=self.TOL)
                self.assertLessEqual(result['mse_pool'], self.MSE_TOL)


                # check sparsity structure of W: W[i, j] should be 0 for every j that is not in neighbors[i] and j != i
                for i in range(n):
                    allowed_js = set(neighbors[i]) | {i}
                    for j in range(n):
                        if j not in allowed_js:
                            self.assertAlmostEqual(W_est[i, j], 0.0, delta=self.TOL, msg=f"W[{i}, {j}] should be 0 but is {W_est[i, j]}")

                # repeat with graph constraints turned off; should still recover parameters well, but W may differ
                result_no_graph_constraints = fit_friedkin_johnsen_joint_traj0(
                    run_traj, run_neighbors, eps=1e-6, turn_off_graph_constraints=True
                )

                # check correct ranges for lambdas and bias
                self.assertGreaterEqual(result_no_graph_constraints['lambda1'], 0.0)
                self.assertGreaterEqual(result_no_graph_constraints['lambda2'], 0.0)
                self.assertLessEqual(result_no_graph_constraints['lambda1'] + result_no_graph_constraints['lambda2'], 1.0 + self.TOL)
                self.assertLessEqual(result_no_graph_constraints['b'], 1.0 + self.TOL)
                self.assertGreaterEqual(result_no_graph_constraints['b'], -1.0 - self.TOL)
                # check row stochasticity and non-negativity of W
                W_est_no_constraints = result_no_graph_constraints['W']
                self.assertTrue(np.all(W_est_no_constraints >= -self.TOL), "W has negative entries")
                np.testing.assert_allclose(W_est_no_constraints.sum(axis=1), 1.0, atol=self.TOL, err_msg="Rows of W do not sum to 1")

                self.assertAlmostEqual(result_no_graph_constraints['lambda1'], lambda1_true, delta=self.TOL)
                self.assertAlmostEqual(result_no_graph_constraints['lambda2'], lambda2_true, delta=self.TOL)
                self.assertAlmostEqual(result_no_graph_constraints['b'], bias_true, delta=self.TOL)
                self.assertLessEqual(result_no_graph_constraints['mse_pool'], self.MSE_TOL)
                np.testing.assert_allclose(result_no_graph_constraints['W'], W_true, atol=self.TOL)

                # validate that unconstrained mSE is lower
                self.assertLessEqual(result_no_graph_constraints['mse_pool'], result['mse_pool'] + self.MSE_TOL)



# ===========================================================================
# Self-loop expressiveness
# ===========================================================================

class TestSelfLoop(unittest.TestCase):
    """FJ traj0 allows self-loops (allowed[i] = 1); verify they are usable."""

    def test_self_loop_weight_nonzero_when_beneficial(self):
        """When true W has self-weight, the fitter should recover a nonzero W[i,i]."""

        # generate W with identity
        n = 10
        W_true = np.eye(n)
        lambda1 = 0.2
        lambda2 = 0.3
        bias = 0.4
        run_traj = _sim_fj_traj0(
            rng=np.random.default_rng(123),
            W_true=W_true,
            lambda1=lambda1,
            lambda2=lambda2,
            bias=bias,
            n_runs=5,
            horizon=15,
        )
        neighbors = {i: [] for i in range(n)}  # no neighbors, only self-loop allowed
        run_neighbors = {f"run_{r:02d}": neighbors for r in range(5)}

        result = fit_friedkin_johnsen_joint_traj0(
            run_traj, run_neighbors, eps=1e-6, turn_off_graph_constraints=False
        )

        # assert correct parameter recovery
        self.assertAlmostEqual(result['lambda1'], lambda1, delta=1e-4)
        self.assertAlmostEqual(result['lambda2'], lambda2, delta=1e-4)
        self.assertAlmostEqual(result['b'], bias, delta=1e-4)
        np.testing.assert_allclose(result['W'], W_true, atol=1e-4)
        

# ===========================================================================
# Validate correctness of rollout function from adj code
# ===========================================================================
class TestFJRolloutFixed(unittest.TestCase):
    """Validate that the rollout function from the adjacency-based code still works
        when given a W that is consistent with the fixed graph constraints."""
    
    def test_rollout_matches_traj0_simulation(self):
        """Validate that the rollout function from the adjacency-based code matches the traj0 simulation when given the same parameters and a W that is consistent with the fixed graph constraints."""

        n = 8
        neighbors = _ring(n)
        W_true = _random_W_from_neighbors(neighbors, n, rng=np.random.default_rng(2024), include_self=True)
        lambda1 = 0.25
        lambda2 = 0.15
        bias = 0.3
        run_traj = _sim_fj_traj0(
            rng=np.random.default_rng(1234),
            W_true=W_true,
            lambda1=lambda1,
            lambda2=lambda2,
            bias=bias,
            n_runs=3,
            horizon=10,
        )
        run_neighbors = {f"run_{r:02d}": neighbors for r in range(3)}
        result = fit_friedkin_johnsen_joint_traj0(
            run_traj, run_neighbors, eps=1e-6, turn_off_graph_constraints=False
        )

        W_est = result['W']
        b_est = result['b']
        lambda1_est = result['lambda1']
        lambda2_est = result['lambda2']

        for r in range(3):
            traj_rollout = friedkin_johnsen_adjacency_rollout(
                x0=run_traj[f"run_{r:02d}"][0],
                w=W_est,
                lambda1=    lambda1_est,
                lambda2=lambda2_est,
                bias=b_est,
                horizon=10,
            )
            np.testing.assert_allclose(traj_rollout, run_traj[f"run_{r:02d}"], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
