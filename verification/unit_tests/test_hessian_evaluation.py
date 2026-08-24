import unittest
from pathlib import Path
import sys
import csv
from tqdm import tqdm

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# ensure project imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.models.adjacency_based.friedkin_johnsen import (
    select_friedkin_johnsen_adjacency_lambdas
)
from modeling.models.adjacency_based.homophily import (
    fit_homophily_stubborness
)

from modeling.models.adjacency_based.bias_only import (
    fit_bias_init_only_model
)

from verification.evaluate_param_sensitivity import (
    evaluate_mse,
    get_perturbation_function,
    num_hessian
)

def _random_sparse(n, in_degree, rng):
    in_degree = max(1, min(in_degree, n - 1))
    nbrs = {}
    for i in range(n):
        pool = [j for j in range(n) if j != i]
        nbrs[i] = rng.choice(pool, size=in_degree, replace=False).tolist()
    return nbrs

class TestHessianEval(unittest.TestCase):

    def test_mse_pool_comp_match_random(self):

        # check that the computation of mses matches between optimization
        #  scripting and the eval in evaluate_param_sensitivity.py

        num_cases = 5
        T = 10
        n = 10
        num_runs = 10

        rng = np.random.default_rng(10)
        for _ in tqdm(range(num_cases), desc="Testing MSE pool computation match"):
            run_traj = {}
            run_nbrs = {}
            for run in range(num_runs):
                run_traj[f"run_{run}"] = rng.uniform(-1.0, 1.0, size=(T,n))
                nbrs = _random_sparse(10, 3, rng)
                run_nbrs[f"run_{run}"] = nbrs

            result_fj = select_friedkin_johnsen_adjacency_lambdas(run_traj, run_nbrs)[0]
            mse_pool_fj = result_fj['mse_pool']

            lambda_soc = result_fj['gamma']*(1 - result_fj['lambda1'] - result_fj['lambda2'])
            lambda_self = (1.0 - result_fj['gamma'])*(1 - result_fj['lambda1'] - result_fj['lambda2'])
            lambda_soc = max(lambda_soc, 0.0)
            lambda_self = max(lambda_self, 0.0)
            lambda_init = result_fj['lambda1']
            lambda_bias = result_fj['lambda2']
            bias = result_fj['bias']

            mse_recomputed = evaluate_mse(
                run_traj, run_nbrs,
                lambda_self = lambda_self,
                lambda_soc=lambda_soc,
                lambda_init=lambda_init,
                lambda_bias=lambda_bias,
                bias=bias,
                gamma = 0.0)
            
            self.assertAlmostEqual(mse_pool_fj, mse_recomputed, places=6)

            # repeat for homophily model
            result_hom_stub = fit_homophily_stubborness(run_traj, run_nbrs)
            mse_pool_hom_stub = result_hom_stub['mse_pool']

            lambda_self = result_hom_stub['lambda_self']
            lambda_init = result_hom_stub['lambda1']
            lambda_bias = result_hom_stub['lambda2']
            lambda_soc = max(1 - lambda_self - lambda_init - lambda_bias, 0.0)
            bias = result_hom_stub['bias']
            gamma = result_hom_stub['gamma']

            mse_recomputed_hom_stub = evaluate_mse(
                run_traj, run_nbrs,
                lambda_self = lambda_self,
                lambda_soc=lambda_soc,
                lambda_init=lambda_init,
                lambda_bias=lambda_bias,
                bias=bias,
                gamma = gamma)
            
            self.assertAlmostEqual(mse_pool_hom_stub, mse_recomputed_hom_stub, places=6)

            # repeat for bias_init_only model
            result_bias_init_only = fit_bias_init_only_model(run_traj)
            mse_bias_init_only = result_bias_init_only['mse_pool']

            lambda_self = result_bias_init_only['lambda_self']
            lambda_bias = result_bias_init_only['lambda_bias']
            lambda_init = result_bias_init_only['lambda_init']
            bias = result_bias_init_only['bias']
            lambda_soc = 0.0
            gamma = 0.0

            mse_recomputed_bias_init_only = evaluate_mse(
                run_traj, run_nbrs,
                lambda_self = lambda_self,
                lambda_soc=lambda_soc,
                lambda_init=lambda_init,
                lambda_bias=lambda_bias,
                bias=bias,
                gamma = gamma)
            
            self.assertAlmostEqual(mse_bias_init_only, mse_recomputed_bias_init_only, places=6)

    def test_perturbation(self):

        lambda_self_base = 0.3
        lambda_soc_base = 0.2
        lambda_init_base = 0.3
        lambda_bias_base = 0.2
        bias_base = -0.5 
        gamma = 0.2

        base_params = {
            'lambda_self': lambda_self_base,
            'lambda_soc': lambda_soc_base,
            'lambda_init': lambda_init_base,
            'lambda_bias': lambda_bias_base,
            'bias': bias_base,
            'gamma': gamma
        }

        num_cases = 5
        T = 10
        n = 10
        num_runs = 10

        rng = np.random.default_rng(10)
        for _ in tqdm(range(num_cases), desc="Testing MSE pool computation match"):
            run_traj = {}
            run_nbrs = {}
            for run in range(num_runs):
                run_traj[f"run_{run}"] = rng.uniform(-1.0, 1.0, size=(T,n))
                nbrs = _random_sparse(10, 3, rng)
                run_nbrs[f"run_{run}"] = nbrs

            # lambda_soc
            f = get_perturbation_function(run_traj, run_nbrs, 'lambda_soc', base_params)

            mse_recomputed = f(0.05)

            mse_recomputed_direct = evaluate_mse(
                run_traj, run_nbrs,
                lambda_self = 0.25,
                lambda_soc = 0.25,
                lambda_init = lambda_init_base,
                lambda_bias = lambda_bias_base,
                bias = bias_base,
                gamma = gamma,)
            
            self.assertAlmostEqual(mse_recomputed, mse_recomputed_direct, places=6)

            # lambda_init
            f = get_perturbation_function(run_traj, run_nbrs, 'lambda_init', base_params)

            mse_recomputed = f(-0.05)

            mse_recomputed_direct = evaluate_mse(
                run_traj, run_nbrs,
                lambda_self = 0.35,
                lambda_soc = lambda_soc_base,
                lambda_init = 0.25,
                lambda_bias = lambda_bias_base,
                bias = bias_base,
                gamma = gamma,)
            
            self.assertAlmostEqual(mse_recomputed, mse_recomputed_direct, places=6)
            
            # lambda_bias
            f = get_perturbation_function(run_traj, run_nbrs, 'lambda_bias', base_params)

            mse_recomputed = f(0.1)

            mse_recomputed_direct = evaluate_mse(
                run_traj, run_nbrs,
                lambda_self = 0.2,
                lambda_soc = lambda_soc_base,
                lambda_init = lambda_init_base,
                lambda_bias = 0.3,
                bias = bias_base,   
                gamma = gamma,)
            
            self.assertAlmostEqual(mse_recomputed, mse_recomputed_direct, places=6)
            
            # bias
            f = get_perturbation_function(run_traj, run_nbrs, 'bias', base_params)

            mse_recomputed = f(-0.1)

            mse_recomputed_direct = evaluate_mse(
                run_traj, run_nbrs,
                lambda_self = lambda_self_base,
                lambda_soc = lambda_soc_base,
                lambda_init = lambda_init_base,
                lambda_bias = lambda_bias_base,
                bias = -0.6,
                gamma = gamma,)
            
            self.assertAlmostEqual(mse_recomputed, mse_recomputed_direct, places=6)

            # gamma
            f = get_perturbation_function(run_traj, run_nbrs, 'gamma', base_params)

            mse_recomputed = f(0.15)

            mse_recomputed_direct = evaluate_mse(
                run_traj, run_nbrs,
                lambda_self = lambda_self_base,
                lambda_soc = lambda_soc_base,
                lambda_init = lambda_init_base,
                lambda_bias = lambda_bias_base,
                bias = bias_base,
                gamma = 0.35,)
            
            self.assertAlmostEqual(mse_recomputed, mse_recomputed_direct, places=6) 

    def test_hessian_estimation(self):

        f_quad = lambda x: 2.0 * x**2 + 3.0 * x + 5.0

        hess_estimate = num_hessian(f_quad)
        self.assertAlmostEqual(hess_estimate, 4.0, places=3)

        f_cub = lambda x: x**3
        self.assertAlmostEqual(num_hessian(f_cub), 0.0, places=3)

        f_cub_2 = lambda x: (x - 1.0)**3 + 2.0
        self.assertAlmostEqual(num_hessian(f_cub_2), -6.0, places=3)

        f_lin = lambda x: 5.0 * x + 2.0
        self.assertAlmostEqual(num_hessian(f_lin), 0.0, places=3)

        f_sin = lambda x: np.sin(x)
        self.assertAlmostEqual(num_hessian(f_sin), 0.0, places=3)



if __name__ == '__main__':
    unittest.main()