import unittest
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.models.adjacency_based.repulsion import (
    fit_repulsion_joint_add,
    fit_friedkin_johnsen_bias_tanh_repulsion_joint_add
)
from modeling.models.data_prep import (
    build_expected_message_matrix,
)

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

def _sim_tanh_repulsion_no_homophily(Abar,
                                     lambda_self = None,
                                     lambda_social = None,
                                     lambda_init = 0.0,
                                     lambda_bias = 0.0,
                                     bias = 0.0,
                                     lambda_repulsion = None,
                                     theta_repulsion = None,
                                     n_runs = None, 
                                     horizon = None,
                                     rng = None, 
                                     x0 = None,
                                     noise_std=0.0,):
    
    # check lambdas form a convex combination
    if not abs(lambda_self + lambda_social + lambda_init + lambda_bias + lambda_repulsion - 1.0) < 1e-8:
        raise ValueError("Sum of lambda_self, lambda_social, lambda_init, lambda_bias, and lambda_repulsion must equal 1.0")
    if not (lambda_self >= 0 and lambda_social >= 0 and lambda_init >= 0 and lambda_bias >= 0 and lambda_repulsion >= 0):
        raise ValueError("All lambda values must be non-negative")
    n = Abar.shape[0]
    run_traj = {}

    for r in range(n_runs):
        if x0 is None:
            x = rng.uniform(-1.0, 1.0, size=n)
        else:
            x = np.asarray(x0, dtype=float)
            if x.shape[0] != n:
                raise ValueError(f"x0 must have length {n}, but has length {x.shape[0]}")
        states = [x.copy()]
        for _ in range(horizon):
            # compute the repulsion term
            repulsion_term = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i != j and Abar[i, j] > 0:
                        diff = x[j] - x[i]
                        if abs(diff) > theta_repulsion:
                            repulsion_term[i] += -1*diff
            repulsion_term = np.tanh(repulsion_term)

            x = (lambda_self * x + lambda_social * Abar @ x \
                    + lambda_init * states[0] + \
                        lambda_bias * bias + \
                            lambda_repulsion * repulsion_term) 
            if noise_std > 0.0:
                noise =  noise_std * rng.normal(size=n)
                x += noise
            # clip to [-1, 1]
            x = np.clip(x, -1.0, 1.0)
            states.append(x.copy())

        run_traj[f'run_{r:02d}'] = np.asarray(states)
    return run_traj


class TestBaseRepulsionRecovery(unittest.TestCase):
    CASES = [
        # (name, topo_fn, n, lambda_soc, lambda_rep, theta_repulsion, n_runs, horizon)
        ('complete', _all_to_all, 6, 0.2, 0.5, 0.2, 8,  20),
        ('ring',     _ring,       8, 0.3, 0.7, 0.3, 8,  25),
        ('star',     _star,       7, 0.25,0.6, 1.0, 8,  25),
        ('chain',    _chain,      8, 0.3, 0.55,0.3, 10, 25),
    ]

    def test_topology_sweep(self):
        for name, topo_fn, n, l_soc, l_rep, theta_rep, n_runs, horizon in self.CASES:
            with self.subTest(topology=name):
                rng = np.random.default_rng(hash(name) % (2**31))
                nbrs = topo_fn(n)
                Abar = build_expected_message_matrix(nbrs, n)
                run_traj = _sim_tanh_repulsion_no_homophily(
                    Abar,
                    lambda_self = 1.0 - l_soc - l_rep,
                    lambda_social=l_soc,
                    lambda_repulsion=l_rep,
                    theta_repulsion=theta_rep,
                    n_runs=n_runs,
                    horizon=horizon,
                    rng=rng,
                )
                run_neighbors = {rn:nbrs for rn in run_traj.keys()}
                fit = fit_repulsion_joint_add(run_traj, run_neighbors)
                self.assertAlmostEqual(fit['lambda_social'], l_soc, delta=1e-04)
                self.assertAlmostEqual(fit['lambda_repulsion'], l_rep, delta=1e-04)
                self.assertAlmostEqual(fit['lambda_self'], 1.0 - l_soc - l_rep, delta=1e-04)

                self.assertLess(fit['mse_pool'], 1e-04)

    def test_random_sparse(self):
        rng = np.random.default_rng(42)
        n, in_degree = 16, 5

        # test on 5 different random graph
        for _ in range(5):
            nbrs = _random_sparse(n, in_degree, rng)
            Abar = build_expected_message_matrix(nbrs, n)
            l_soc, l_rep, theta_rep = 0.3, 0.6, 0.5

            run_traj = _sim_tanh_repulsion_no_homophily(
                Abar = Abar,
                lambda_self = 1.0 - l_soc - l_rep,
                lambda_social = l_soc,
                lambda_repulsion= l_rep,
                theta_repulsion=theta_rep,
                n_runs = 8,
                horizon = 10,
                rng=rng
            )

            run_neighbors = {rn:nbrs for rn in run_traj.keys()}
            fit = fit_repulsion_joint_add(run_traj, run_neighbors)

            # check that l_soc and l_rep are recovered accurately
            self.assertAlmostEqual(fit['lambda_social'], l_soc, delta=1e-04)
            self.assertAlmostEqual(fit['lambda_repulsion'], l_rep, delta=1e-04)
            self.assertAlmostEqual(fit['lambda_self'], 1.0 - l_soc - l_rep, delta=1e-04)

            self.assertLess(fit['mse_pool'], 1e-04)

    def test_per_run_different_graph(self):
        """Hald runs on ring, half on complete; """
        rng = np.random.default_rng(123)
        n, theta_rep, l_soc, l_rep = 15, 0.4, 0.2, 0.5

        run_traj ,run_neighbors = {},{}

        for r in range(16):
            if r % 2 == 0:
                nbrs = _ring(n)
            else:
                nbrs = _all_to_all(n)
            Abar = build_expected_message_matrix(nbrs, n)
            run_traj[f'run_{r:02d}'] = _sim_tanh_repulsion_no_homophily(
                Abar = Abar,
                lambda_self = 1.0 - l_soc - l_rep,
                lambda_social = l_soc,
                lambda_repulsion= l_rep,
                theta_repulsion=theta_rep,
                n_runs = 1,
                horizon = 10,
                rng=rng
            )[f'run_00']
            run_neighbors[f'run_{r:02d}'] = nbrs
        fit = fit_repulsion_joint_add(run_traj, run_neighbors)

        if not (fit['lambda_social'] - l_soc < 1e-04 and fit['lambda_repulsion'] - l_rep < 1e-04 and fit['lambda_self'] - (1.0 - l_soc - l_rep) < 1e-04):
            # if there's some mismatch, check that we at least get good values when we try and re-fit with
            #   a known theta_repulsion value, and check that when we rollout the problem, we get a trajectory that is close to the original trajectory
            fit_oracle = fit_repulsion_joint_add(run_traj, run_neighbors, custom_search_vals=[theta_rep])

            for r in range(16):
                nbrs = run_neighbors[f'run_{r:02d}']
                Abar = build_expected_message_matrix(nbrs, n)
                recovered_traj = _sim_tanh_repulsion_no_homophily(
                    Abar = Abar,
                    lambda_self = fit_oracle['lambda_self'],
                    lambda_social = fit_oracle['lambda_social'],
                    lambda_repulsion= fit_oracle['lambda_repulsion'],
                    theta_repulsion=fit_oracle['theta_rep'],
                    n_runs = 1,
                    horizon = 10,
                    x0 = run_traj[f'run_{r:02d}'][0],
                )[f'run_00']
                
                self.assertTrue(np.allclose(recovered_traj, run_traj[f'run_{r:02d}'], atol=1e-04), 
                                msg=f"Recovered trajectory does not match original trajectory for run {r}. Max abs diff: {np.max(np.abs(recovered_traj - run_traj[f'run_{r:02d}']))}")
                
        else:
            self.assertAlmostEqual(fit['lambda_social'], l_soc, delta=1e-04)
            self.assertAlmostEqual(fit['lambda_repulsion'], l_rep, delta=1e-04)
            self.assertAlmostEqual(fit['lambda_self'], 1.0 - l_soc - l_rep, delta=1e-04)

        self.assertLessEqual(fit['mse_pool'], 1e-04)

    def test_varied_graph_random_params(self):

        """Generate 10 sets of runs with
                - randomly sampled generation parmaeters
                - 10 runs with different random graphs
        """

        rng = np.random.default_rng(456)
        n, in_degree = 12, 4

        # generate a random list of 10 sets of parameters

        param_set = []
        for _ in range(10):
            l_soc = rng.uniform(0.0, 1.0)
            l_rep = rng.uniform(0.0, 1.0 - l_soc)
            theta_rep = rng.uniform(0.0, 2.0)

            param_set.append((l_soc, l_rep, theta_rep))

        # also push some edge cases
        param_set.append((0.0, 0.5, 0.5))
        param_set.append((0.5, 0.0, 0.5))
        param_set.append((0.5, 0.5, 0.5))
        param_set.append((0.3, 0.2, 0.0))
        param_set.append((0.2, 0.3, 2.0))

        for p in param_set:

            l_soc, l_rep, theta_rep = p

            run_traj, run_neighbors = {}, {}
            for r in range(10):
                nbrs = _random_sparse(n, in_degree, rng)
                Abar = build_expected_message_matrix(nbrs, n)
                run_traj[f'run_{r:02d}'] = _sim_tanh_repulsion_no_homophily(
                    Abar = Abar,
                    lambda_self = 1.0 - l_soc - l_rep,
                    lambda_social = l_soc,
                    lambda_repulsion= l_rep,
                    theta_repulsion=theta_rep,
                    n_runs = 1,
                    horizon = 10,
                    rng=rng
                )[f'run_00']
                run_neighbors[f'run_{r:02d}'] = nbrs
            fit = fit_repulsion_joint_add(run_traj, run_neighbors)

            DELTA = 1e-04

            if not (abs(fit['lambda_social'] - l_soc) < DELTA):

                # if there's some mismatch, check that we at least get good values when we try and re-fit with
                #   a known theta_repulsion value, and check that when we rollout the problem, we get a trajectory that is close to the original trajectory
                fit_oracle = fit_repulsion_joint_add(run_traj, run_neighbors, custom_search_vals=[theta_rep])

                for r in range(10):
                    nbrs = run_neighbors[f'run_{r:02d}']
                    Abar = build_expected_message_matrix(nbrs, n)
                    recovered_traj = _sim_tanh_repulsion_no_homophily(
                        Abar = Abar,
                        lambda_self = fit_oracle['lambda_self'],
                        lambda_social = fit_oracle['lambda_social'],
                        lambda_repulsion= fit_oracle['lambda_repulsion'],
                        theta_repulsion=fit_oracle['theta_rep'],
                        n_runs = 1,
                        horizon = 10,
                        x0 = run_traj[f'run_{r:02d}'][0],
                    )[f'run_00']
                    
                    self.assertTrue(np.allclose(recovered_traj, run_traj[f'run_{r:02d}'], atol=DELTA), 
                                    msg=f"Recovered trajectory does not match original trajectory for run {r} with parameters {p}. Max abs diff: {np.max(np.abs(recovered_traj - run_traj[f'run_{r:02d}']))}")
                    

                self.assertLessEqual(fit_oracle['mse_pool'], 1e-04)
            else:
                self.assertAlmostEqual(fit['lambda_social'], l_soc, delta=DELTA)
                self.assertAlmostEqual(fit['lambda_repulsion'], l_rep, delta=DELTA)
                self.assertAlmostEqual(fit['lambda_self'], 1.0 - l_soc - l_rep, delta=DELTA)

                self.assertLessEqual(fit['mse_pool'], 1e-04)

             
class TestBaseRepulsionFJBiasRecovery(unittest.TestCase):
    CASES = [
        # (name, topo_fn, n, lambda_soc, lambda_rep, lambda_bias, lambda_init, theta_repulsion, bias, n_runs, horizon)
        ('complete', _all_to_all, 6, 0.1, 0.2, 0.2, 0.2, 0.2, 1.0, 8,  20),
        ('ring',     _ring,       8, 0.2, 0.2, 0.1, 0.1, 0.3, 0.5, 8,  25),
        ('star',     _star,       7, 0.05,0.05, 0.3, 0.3, 1.0, -0.5, 8,  25),
        ('chain',    _chain,      8, 0.3, 0.15,0.3, 0.05, 0.3, -1.0, 10, 25),
    ]       

    def test_topology_sweep(self):
        for name, topo_fn, n, l_soc, l_rep, l_bias, l_init, theta_rep, bias, n_runs, horizon in self.CASES:
            with self.subTest(topology=name):
                rng = np.random.default_rng(hash(name) % (2**31))
                nbrs = topo_fn(n)
                Abar = build_expected_message_matrix(nbrs, n)
                run_traj = _sim_tanh_repulsion_no_homophily(
                    Abar,
                    lambda_self = 1.0 - l_soc - l_rep - l_bias - l_init,
                    lambda_social=l_soc,
                    lambda_repulsion=l_rep,
                    lambda_bias=l_bias,
                    lambda_init=l_init,
                    theta_repulsion=theta_rep,
                    bias=bias,
                    n_runs=n_runs,
                    horizon=horizon,
                    rng=rng,
                )
                run_neighbors = {rn:nbrs for rn in run_traj.keys()}
                fit = fit_friedkin_johnsen_bias_tanh_repulsion_joint_add(run_traj, run_neighbors)
                self.assertAlmostEqual(fit['lambda_social'], l_soc, delta=1e-04)
                self.assertAlmostEqual(fit['lambda_repulsion'], l_rep, delta=1e-04)
                self.assertAlmostEqual(fit['lambda_bias'], l_bias, delta=1e-04)
                self.assertAlmostEqual(fit['lambda_init'], l_init, delta=1e-04)
                self.assertAlmostEqual(fit['lambda_self'], 1.0 - l_soc - l_rep - l_bias - l_init, delta=1e-04)

                self.assertLess(fit['mse_pool'], 1e-04)


    def test_random_sparse(self):
        rng = np.random.default_rng(42)
        n, in_degree = 16, 5

        for _ in range(5):
            nbrs = _random_sparse(n, in_degree, rng)
            Abar = build_expected_message_matrix(nbrs, n)
            l_soc, l_rep, l_bias, l_init, theta_rep = 0.2, 0.3, 0.1, 0.1, 0.5
            bias = rng.uniform(-1.0, 1.0)

            run_traj = _sim_tanh_repulsion_no_homophily(
                Abar = Abar,
                lambda_self = 1.0 - l_soc - l_rep - l_bias - l_init,
                lambda_social = l_soc,
                lambda_repulsion= l_rep,
                lambda_bias=l_bias,
                lambda_init=l_init,
                theta_repulsion=theta_rep,
                bias=bias,
                n_runs = 8,
                horizon = 10,
                rng=rng
            )

            run_neighbors = {rn:nbrs for rn in run_traj.keys()}
            fit = fit_friedkin_johnsen_bias_tanh_repulsion_joint_add(run_traj, run_neighbors)

            self.assertAlmostEqual(fit['lambda_social'], l_soc, delta=1e-04)
            self.assertAlmostEqual(fit['lambda_repulsion'], l_rep, delta=1e-04)
            self.assertAlmostEqual(fit['lambda_bias'], l_bias, delta=1e-04)
            self.assertAlmostEqual(fit['lambda_init'], l_init, delta=1e-04)
            self.assertAlmostEqual(fit['lambda_self'], 1.0 - l_soc - l_rep - l_bias - l_init, delta=1e-04)

            self.assertLess(fit['mse_pool'], 1e-04)

    def test_per_run_different_graph(self):
        """Half runs on ring, half on complete; """
        rng = np.random.default_rng(123)
        n, theta_rep, l_soc, l_rep, l_bias, l_init = 15, 0.4, 0.2, 0.3, 0.1, 0.1
        bias = -0.2

        run_traj ,run_neighbors = {},{}

        for r in range(16):
            if r % 2 == 0:
                nbrs = _ring(n)
            else:
                nbrs = _all_to_all(n)
            Abar = build_expected_message_matrix(nbrs, n)
            run_traj[f'run_{r:02d}'] = _sim_tanh_repulsion_no_homophily(
                Abar = Abar,
                lambda_self = 1.0 - l_soc - l_rep - l_bias - l_init,
                lambda_social = l_soc,
                lambda_repulsion= l_rep,
                lambda_bias=l_bias,
                lambda_init=l_init,
                theta_repulsion=theta_rep,
                bias=bias,
                n_runs = 1,
                horizon = 10,
                rng=rng
            )[f'run_00']
            run_neighbors[f'run_{r:02d}'] = nbrs
        fit = fit_friedkin_johnsen_bias_tanh_repulsion_joint_add(run_traj, run_neighbors)

        DELTA = 1e-04
        self.assertAlmostEqual(fit['lambda_social'], l_soc, delta=DELTA)
        self.assertAlmostEqual(fit['lambda_repulsion'], l_rep, delta=DELTA)
        self.assertAlmostEqual(fit['lambda_bias'], l_bias, delta=DELTA)
        self.assertAlmostEqual(fit['lambda_init'], l_init, delta=DELTA)
        self.assertAlmostEqual(fit['lambda_self'], 1.0 - l_soc - l_rep - l_bias - l_init, delta=DELTA)
        self.assertLessEqual(fit['mse_pool'], 1e-04)

    def test_varied_graph_random_params(self):

        """Generate 10 sets of runs with
                - randomly sampled generation parameters
                - 10 runs with different random graphs
        """

        rng = np.random.default_rng(456)
        n, in_degree = 12, 4

        # generate a random list of 10 sets of parameters

        param_set = []
        for _ in range(10):
            # choose 5 weights randomly from 0,1 and normalize
            weights = rng.uniform(0.0, 1.0, size=5)
            weights /= weights.sum()
            l_soc, l_rep, l_bias, l_init, _ = weights
            theta_rep = rng.uniform(0.0, 2.0)
            bias = rng.uniform(-1.0, 1.0)

            param_set.append((l_soc, l_rep, l_bias, l_init, theta_rep, bias))

        # also push some edge cases
        #  one weight is 0
        param_set.append((0.0, 0.5, 0.2, 0.2, 0.5, 0.5))
        param_set.append((0.5, 0.0, 0.2, 0.2, 0.5, -0.5))
        param_set.append((0.5, 0.2, 0.0, 0.2, 0.5, 0.0))
        param_set.append((0.3, 0.2, 0.2, 0.0, 1.0, 1.0))
        param_set.append((0.2, 0.3, 0.3, 0.2, 0.0, -1.0))
        
        # edge case theta_rep values
        param_set.append((0.2, 0.3, 0.2, 0.2, 0.0, 0.5))
        param_set.append((0.2, 0.3, 0.2, 0.2, 2.0, -0.5))

        # edge case bias values
        param_set.append((0.2, 0.3, 0.2, 0.2, 0.5, -1.0))
        param_set.append((0.2, 0.3, 0.2, 0.2, 0.5, 1.0))

        for p in param_set:

            l_soc, l_rep, l_bias, l_init, theta_rep, bias = p

            run_traj, run_neighbors = {}, {}
            for r in range(10):
                nbrs = _random_sparse(n, in_degree, rng)
                Abar = build_expected_message_matrix(nbrs, n)
                run_traj[f'run_{r:02d}'] = _sim_tanh_repulsion_no_homophily(
                    Abar = Abar,
                    lambda_self = 1.0 - l_soc - l_rep - l_bias - l_init,
                    lambda_social = l_soc,
                    lambda_repulsion= l_rep,
                    lambda_bias=l_bias,
                    lambda_init=l_init,
                    theta_repulsion=theta_rep,
                    bias=bias,
                    n_runs = 1,
                    horizon = 10,
                    rng=rng
                )[f'run_00']
                run_neighbors[f'run_{r:02d}'] = nbrs
            fit = fit_friedkin_johnsen_bias_tanh_repulsion_joint_add(run_traj, run_neighbors)

            DELTA = 1e-04
            if not abs(fit['lambda_repulsion'] - l_rep) < DELTA:
                # if we're not recovering the exact parameters, we need to check that, when we rollout 
                #   with recovered parameters, we get very close to the original trajectory
                # we should also check that we can at least do this when we add the ground truth theta_rep value to the search
                #   in case it's getting missed in the grid search

                fit_oracle = fit_friedkin_johnsen_bias_tanh_repulsion_joint_add(run_traj, run_neighbors, custom_search_vals=[theta_rep])

                for r in range(10):
                    nbrs = run_neighbors[f'run_{r:02d}']
                    Abar = build_expected_message_matrix(nbrs, n)
                    recovered_traj = _sim_tanh_repulsion_no_homophily(
                        Abar = Abar,
                        lambda_self = fit_oracle['lambda_self'],
                        lambda_social = fit_oracle['lambda_social'],
                        lambda_repulsion= fit_oracle['lambda_repulsion'],
                        lambda_bias=fit_oracle['lambda_bias'],
                        lambda_init=fit_oracle['lambda_init'],
                        theta_repulsion=fit_oracle['theta_rep'],
                        bias=fit_oracle['bias'],
                        n_runs = 1,
                        horizon = 10,
                        x0 = run_traj[f'run_{r:02d}'][0],
                    )[f'run_00']
                    
                    # check that recovered_traj is close to run_traj[f'run_{r:02d}'] within a tolerance of DELTA
                    self.assertTrue(np.allclose(recovered_traj, run_traj[f'run_{r:02d}'], atol=DELTA), 
                                    msg=f"Recovered trajectory does not match original trajectory for run {r} with parameters {p}. Max abs diff: {np.max(np.abs(recovered_traj - run_traj[f'run_{r:02d}']))}")
                    
                    self.assertLessEqual(fit_oracle['mse_pool'], 1e-04)
            else:
                self.assertAlmostEqual(fit['lambda_social'], l_soc, delta=DELTA)
                self.assertAlmostEqual(fit['lambda_repulsion'], l_rep, delta=DELTA)
                self.assertAlmostEqual(fit['lambda_bias'], l_bias, delta=DELTA)
                self.assertAlmostEqual(fit['lambda_init'], l_init, delta=DELTA)
                self.assertAlmostEqual(fit['lambda_self'], 1.0 - l_soc - l_rep - l_bias - l_init, delta=DELTA)

                self.assertLessEqual(fit['mse_pool'], 1e-04)

            

if __name__ == '__main__':
    unittest.main()