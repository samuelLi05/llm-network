import unittest
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.models.adjacency_based.repulsion_mult import (
    fit_repulsion_fj_bias_mult,
    _get_generic_social_kernel_term_weight_based
)

from modeling.models.data_prep import (
    build_expected_message_matrix
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

def _get_generic_social_kernel_term_weight_based_legacy(x, neighbors, F):

    # Get the 'social term' for opinion dynamics models where a force term is used to weight opinion updates
    #  (generalizing )

    # : x : np.array of shape (n_agents, ) representing the opinions of agents
    # : neighbors : list of lists, where neighbors[i] is a list of indices of agents who influence agent i
    # : F : a kernel function that takes a non-negative float and returns a float
    #           which will be used to weight neighbors' opinions

    n_agents = len(x)
    social_x = np.zeros(n_agents)

    for i in range(n_agents):

        nbh = list(neighbors[i])

        weights = [F(abs(x[j] - x[i])) for j in nbh]
        nbh_op = [x[j] for j in nbh]

        if len(weights) == 0:
            raise ValueError(f"Agent {i} has no neighbors, cannot compute social kernel term.")

        weights = np.array(weights)
        weights /= np.sum(np.abs(weights))

        social_x[i] = np.sum([weights[j] * nbh_op[j] for j in range(len(nbh))])

    return np.array(social_x)


def _sim_weight_based_mult_repulsion(Abar,
                                     lambda_self = None,
                                     lambda_social = None,
                                     lambda_init = None,
                                     lambda_bias = None,
                                     bias = None,
                                     beta_rep = None,
                                     n_runs = None,
                                     horizon = None,
                                     rng = None,
                                     x0 = None,
                                     noise_std=0.0,):
    
    # check lambdas form a convex combination
    if not abs(lambda_self + lambda_social + lambda_init + lambda_bias - 1.0) < 1e-8:
        raise ValueError("Sum of lambda_self, lambda_social, lambda_init, lambda_bias, and lambda_repulsion must equal 1.0")
    if not (lambda_self >= 0 and lambda_social >= 0 and lambda_init >= 0 and lambda_bias >= 0):
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

            # compute social (with repulsion) term
            social_term = np.zeros(n)
            for i in range(n):
                weights_i = (1 - beta_rep * np.abs(x[i] - x)) * Abar[i,:]
                weights_i /= np.sum(np.abs(weights_i))
                social_term[i] = np.sum(weights_i * x)

            x = lambda_self * x + lambda_social * social_term \
                + lambda_init * states[0] + \
                lambda_bias * bias
            if noise_std > 0.0:
                noise = noise_std * rng.normal(size = n)
                x += noise
                x = np.clip(x, -1.0, 1.0)
            else:
                # check that x lies in [-1,1]
                if not (np.all(np.logical_and(x <= 1, x>= -1))):
                    raise ValueError("Something went wrong: opinions have left [-1,1] ")
                
            states.append(x.copy())
        run_traj[f'run_{r:02d}'] = np.asarray(states)

    return run_traj

def consistency_check_weight_based_mult(run_traj, run_neighbors, l_self, l_soc, l_bias, l_init, bias, beta_rep):

    errors = {}
    for key, traj in run_traj.items():

        # pull t=0 opinions, and push forward with input parameters (usually generated from fitting)
        #  we want to see that these match run_traj and run_neighbors

        n = np.shape(traj)[1]
        nbrs = run_neighbors[key]
        Abar = build_expected_message_matrix(nbrs,n)

        recovered_traj = _sim_weight_based_mult_repulsion(
                    Abar,
                    lambda_self = l_self,
                    lambda_social=l_soc,
                    lambda_bias=l_bias,
                    lambda_init=l_init,
                    beta_rep=beta_rep,
                    bias=bias,
                    n_runs=1,
                    horizon=np.shape(traj)[0] - 1,
                    x0 = traj[0]
                )[f'run_00']
        
        max_err = np.max(np.abs(recovered_traj - traj))
        errors[key] = max_err
    return errors

class TestMultRepulsionWeightBasedRecovery(unittest.TestCase):

    CASES = [
        # (name, topo_fn, n, lambda_soc, lambda_bias, lambda_init, beta_rep, bias, n_runs, horizon)
        ('complete', _all_to_all, 6, 0.1, 0.2, 0.2, 0.2, 1.0, 8,  20),
        ('ring',     _ring,       8, 0.2, 0.2, 0.1, 1.0, 0.5, 8,  25),
        ('star',     _star,       7, 0.05,0.05, 0.3, 5.0, -0.5, 8,  25),
        ('chain',    _chain,      8, 0.3, 0.15,0.3, 0.4, -1.0, 10, 25),
    ]       

                
    def test_topology_sweep(self):
        seed_cnt = 1

        for name, topo_fn, n, l_soc, l_bias, l_init, beta_rep, bias, n_runs, horizon in self.CASES:
            with self.subTest(topology=name):
                rng = np.random.default_rng(seed_cnt)
                seed_cnt += 1
                nbrs = topo_fn(n)
                Abar = build_expected_message_matrix(nbrs, n)
                run_traj = _sim_weight_based_mult_repulsion(
                    Abar,
                    lambda_self = 1.0 - l_soc - l_bias - l_init,
                    lambda_social=l_soc,
                    lambda_bias=l_bias,
                    lambda_init=l_init,
                    beta_rep=beta_rep,
                    bias=bias,
                    n_runs=n_runs,
                    horizon=horizon,
                    rng=rng,
                )
                run_neighbors = {rn:nbrs for rn in run_traj.keys()}
                fit = fit_repulsion_fj_bias_mult(run_traj, run_neighbors, repulsion_version="weight-based")

                valid = abs(fit['lambda_social'] - l_soc) < 1e-04 \
                        and abs(fit['lambda_bias'] - l_bias) < 1e-04\
                        and abs(fit['lambda_init'] - l_init) < 1e-04\
                        and abs(fit['lambda_self'] - (1.0 - l_soc - l_bias - l_init)) < 1e-04\
                        and abs(fit['beta_rep'] - beta_rep) < 1e-04

                if not (valid):
                    fit_oracle  = fit_repulsion_fj_bias_mult(run_traj, run_neighbors, repulsion_version="weight-based",
                                                     custom_search_values=[beta_rep])
                    max_errors = consistency_check_weight_based_mult(run_traj, run_neighbors, 
                                                                     fit_oracle['lambda_self'],
                                                                     fit_oracle['lambda_social'], 
                                                                     fit_oracle['lambda_bias'], 
                                                                     fit_oracle['lambda_init'], 
                                                                     fit_oracle['bias'], 
                                                                     fit_oracle['beta_rep'])
                    for _,error in max_errors.items():
                        self.assertLess(error, 1e-04)

                else:

                    self.assertAlmostEqual(fit['lambda_social'], l_soc, delta=1e-04)
                    self.assertAlmostEqual(fit['lambda_bias'], l_bias, delta=1e-04)
                    self.assertAlmostEqual(fit['lambda_init'], l_init, delta=1e-04)
                    self.assertAlmostEqual(fit['lambda_self'], 1.0 - l_soc - l_bias - l_init, delta=1e-04)
                    self.assertAlmostEqual(fit['beta_rep'], beta_rep, delta = 1e-04)

                self.assertLess(fit['mse_pool'], 1e-04) 

    def test_random_sparse(self):
        rng = np.random.default_rng(42)
        n, in_degree = 16, 5

        for _ in range(5):
            nbrs = _random_sparse(n, in_degree, rng)
            Abar = build_expected_message_matrix(nbrs, n)
            l_soc, l_bias, l_init, bias, beta_rep = 0.2, 0.3, 0.1, -0.1, 1.1
            bias = rng.uniform(-1.0,1.0)

            run_traj = _sim_weight_based_mult_repulsion(
                    Abar,
                    lambda_self = 1.0 - l_soc - l_bias - l_init,
                    lambda_social=l_soc,
                    lambda_bias=l_bias,
                    lambda_init=l_init,
                    beta_rep=beta_rep,
                    bias=bias,
                    n_runs=9,
                    horizon=10,
                    rng=rng,
                )

            run_neighbors = {rn:nbrs for rn in run_traj.keys()}
            fit = fit_repulsion_fj_bias_mult(run_traj, run_neighbors, repulsion_version="weight-based")

            self.assertAlmostEqual(fit['lambda_social'], l_soc, delta=1e-04)
            self.assertAlmostEqual(fit['lambda_bias'], l_bias, delta=1e-04)
            self.assertAlmostEqual(fit['lambda_init'], l_init, delta=1e-04)
            self.assertAlmostEqual(fit['lambda_self'], 1.0 - l_soc - l_bias - l_init, delta=1e-04)
            self.assertAlmostEqual(fit['beta_rep'], beta_rep, delta =1e-04)

            self.assertLess(fit['mse_pool'], 1e-04)     

    def test_per_run_different_graph(self):
        """Half runs on ring, half on complete; """
        rng = np.random.default_rng(123)
        n, beta_rep, l_soc, l_bias, l_init = 15, 1.5, 0.2, 0.3, 0.1
        bias = -0.2

        run_traj ,run_neighbors = {},{}

        for r in range(16):
            if r % 2 == 0:
                nbrs = _ring(n)
            else:
                nbrs = _all_to_all(n)
            Abar = build_expected_message_matrix(nbrs, n)
            run_traj[f'run_{r:02d}'] = _sim_weight_based_mult_repulsion(
                    Abar,
                    lambda_self = 1.0 - l_soc - l_bias - l_init,
                    lambda_social=l_soc,
                    lambda_bias=l_bias,
                    lambda_init=l_init,
                    beta_rep=beta_rep,
                    bias=bias,
                    n_runs=1,
                    horizon=10,
                    rng=rng,
                )[f'run_00']
            run_neighbors[f'run_{r:02d}'] = nbrs
        fit = fit_repulsion_fj_bias_mult(run_traj, run_neighbors, repulsion_version="weight-based")

        DELTA = 1e-04
        self.assertAlmostEqual(fit['lambda_social'], l_soc, delta=DELTA)
        self.assertAlmostEqual(fit['lambda_bias'], l_bias, delta=DELTA)
        self.assertAlmostEqual(fit['lambda_init'], l_init, delta=DELTA)
        self.assertAlmostEqual(fit['lambda_self'], 1.0 - l_soc - l_bias - l_init, delta=DELTA)
        self.assertAlmostEqual(fit['beta_rep'], beta_rep, delta=DELTA)

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
            # choose 4 weights randomly from 0,1 and normalize
            weights = rng.uniform(0.0, 1.0, size=4)
            weights /= weights.sum()
            l_soc, l_bias, l_init, _ = weights
            beta_rep = rng.uniform(0.0, 5.0)
            bias = rng.uniform(-1.0, 1.0)

            param_set.append((l_soc, l_bias, l_init, beta_rep, bias))

        
        #  one weight is 0
        param_set.append((0.0, 0.5, 0.2, 1.5, 0.5))
        param_set.append((0.5, 0.0, 0.2, 2.5, -0.5))
        param_set.append((0.5, 0.2, 0.0, 0.5, 0.0))
        param_set.append((0.3, 0.2, 0.2, 1.0, 1.0))
        
        # edge case beta_rep values
        param_set.append((0.2, 0.3, 0.2, 0.0, 0.5))
        param_set.append((0.2, 0.3, 0.2, 20.0, -0.5))

        # edge case bias values
        param_set.append((0.2, 0.3, 0.2, 0.5, -1.0))
        param_set.append((0.2, 0.3, 0.2, 1.0, 1.0))


        NUM_TRAJECTORIES = 8

        for p in tqdm(param_set, desc = "Iterating over random graphs and parameters"):

            l_soc, l_bias, l_init, beta_rep, bias = p

            run_traj, run_neighbors = {}, {}
            for r in range(NUM_TRAJECTORIES):
                nbrs = _random_sparse(n, in_degree, rng)
                Abar = build_expected_message_matrix(nbrs, n)
                run_traj[f'run_{r:02d}'] = _sim_weight_based_mult_repulsion(
                    Abar,
                    lambda_self = 1.0 - l_soc - l_bias - l_init,
                    lambda_social=l_soc,
                    lambda_bias=l_bias,
                    lambda_init=l_init,
                    beta_rep=beta_rep,
                    bias=bias,
                    n_runs=1,
                    horizon=10,
                    rng=rng,
                )[f'run_00']
                run_neighbors[f'run_{r:02d}'] = nbrs
            fit = fit_repulsion_fj_bias_mult(run_traj, run_neighbors, repulsion_version="weight-based")

            valid = abs(fit['lambda_social'] - l_soc) < 1e-04 \
                    and abs(fit['lambda_bias'] - l_bias) < 1e-04\
                    and abs(fit['lambda_init'] - l_init) < 1e-04\
                    and abs(fit['lambda_self'] - (1.0 - l_soc - l_bias - l_init)) < 1e-04\
                    and abs(fit['beta_rep'] - beta_rep) < 1e-04
            
            if not (valid):
                print("Testing via traj. regeneration")
                print(f"lambda_social: {fit['lambda_social']} ; {l_soc} ; {fit['lambda_social'] - l_soc}" )
                print(f"lambda_bias: {fit['lambda_bias']} ; {l_bias} ; {fit['lambda_bias'] - l_bias}" )
                print(f"lambda_init: {fit['lambda_init']} ; {l_init} ; {fit['lambda_init'] - l_init}" )
                print(f"lambda_self: {fit['lambda_self']} ; {1.0 - l_soc - l_bias - l_init} ; {fit['lambda_self'] - (1.0 - l_soc - l_bias - l_init)}" )
                print(f"beta_rep: {fit['beta_rep']} ; {beta_rep} ; {fit['beta_rep'] - beta_rep}")
                fit_oracle  = fit_repulsion_fj_bias_mult(run_traj, run_neighbors, repulsion_version="weight-based",
                                                    custom_search_values=[beta_rep])
                max_errors = consistency_check_weight_based_mult(run_traj, run_neighbors, 
                                                                    fit_oracle['lambda_self'],
                                                                    fit_oracle['lambda_social'], 
                                                                    fit_oracle['lambda_bias'], 
                                                                    fit_oracle['lambda_init'], 
                                                                    fit_oracle['bias'], 
                                                                    fit_oracle['beta_rep'])
                
                for _,error in max_errors.items():
                    self.assertLess(error, 1e-04)

            else:

                self.assertAlmostEqual(fit['lambda_social'], l_soc, delta=1e-04)
                self.assertAlmostEqual(fit['lambda_bias'], l_bias, delta=1e-04)
                self.assertAlmostEqual(fit['lambda_init'], l_init, delta=1e-04)
                self.assertAlmostEqual(fit['lambda_self'], 1.0 - l_soc - l_bias - l_init, delta=1e-04)
                self.assertAlmostEqual(fit['beta_rep'], beta_rep, delta = 1e-04)

    def test_social_kernel_term_implementation(self):

        NUM_GRAPHS = 10
        NUM_OPINION_SAMPLES = 10
        kernel_list = [
            lambda x: 1 - 0.5 * x,
            lambda x: 1 - 2.0 * x,
            lambda x: np.exp(-2.0 * x),
            lambda x: 1 - 2* np.exp(-3.0 * x),
        ]

        n = 20
        in_degree = 5

        rng = np.random.default_rng(42)
        for _ in range(NUM_GRAPHS):
            g = _random_sparse(n, in_degree, rng)

            for _ in range(NUM_OPINION_SAMPLES):

                for kernel in kernel_list:
                    x = rng.uniform(-1.0, 1.0, size=n)

                    social_term_legacy = _get_generic_social_kernel_term_weight_based_legacy(x, g, kernel)
                    Abar = build_expected_message_matrix(g, n)
                    social_term_new = _get_generic_social_kernel_term_weight_based(x, Abar, kernel)

                    self.assertTrue(np.allclose(social_term_legacy, social_term_new, atol=1e-08))



        

if __name__ == "__main__":

    unittest.main()