import unittest
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.models.adjacency_based.bias_only import (
    fit_bias_init_only_model,
    bias_init_only_rollout
)

def _simulate_bias_init_only(lambda_self, lambda_init, bias_tilde, n_runs, horizon, n,
                             x0 = None, rng = None):
    
    run_traj = {}

    if not(x0 is None):
        if not (len(np.shape(x0)) == 1 and np.shape(x0)[0] == n):
            raise ValueError(f"Incorrect dimensions {np.shape(x0)} for x0")

    for r in range(n_runs):
        if x0 is None:
            x0 = rng.uniform(-1.0, 1.0, size=n)

        x = x0.copy()
        states = [x.copy()]

        for _ in range(horizon):
            x = lambda_self * x + lambda_init * x0 + bias_tilde
            states.append(x.copy())
        run_traj[f'run_{r:02d}'] = np.asarray(states)
    return run_traj

def _grid_search_bias_init(traj, grid_points):
    # helper to fit parameters using a brute force search 
    #  from a single run

    T,n = np.shape(traj)
    
    x = traj[:T-1,:]
    y = traj[1:, :]
    x0 = np.repeat(x[0].reshape(1, -1), x.shape[0], axis = 0)

    best_mse = np.inf
    best_sol = {"lambda_self": np.nan,
                "lambda_init": np.nan,
                "bias_tilde" : np.nan}

    lambda_self_search_values = np.linspace(0.0, 1.0, grid_points)
    lambda_init_search_values = np.linspace(0.0, 1.0, grid_points)
    bias_tilde_search_values = np.linspace(-1.0, 1.0, 2*grid_points)

    for l_self in lambda_self_search_values:
        for l_init in lambda_init_search_values:
            for bias_tilde in bias_tilde_search_values:
                if l_self + l_init + abs(bias_tilde) <= 1.0:

                    mse = np.sum((x * l_self + bias_tilde + x0 * l_init - y)**2)

                    if mse < best_mse:
                        best_mse = mse
                        best_sol = {"lambda_self":l_self,
                                    "lambda_init": l_init,
                                    "bias_tilde" : bias_tilde}

    if np.isinf(best_mse):
        raise ValueError("No non-inifte solution found")
    
    return best_sol


class TestBiasRecovery(unittest.TestCase):

    def test_bias_init_recovery(self):

        param_list = []

        # specifying cases with convention (l_self, l_init, bias_tilde)

        # all cases with only one effect
        param_list.append((1.0, 0.0, 0.0))
        param_list.append((0.0, 1.0, 0.0))
        param_list.append((0.0, 0.0, 0.0))  # no bias
        param_list.append((0.0, 0.0, 1.0))  # + bias
        param_list.append((0.0, 0.0, -1.0)) # - bias

        # all cases with only two effects
        param_list.append((0.4, 0.5, 0.0))
        param_list.append((0.3, 0.0, 0.2))  # + bias
        param_list.append((0.3, 0.0, -0.2)) # - bias
        param_list.append((0.0, 0.5, 0.5))  # + bias
        param_list.append((0.0, 0.6, -0.4)) # - bias

        rng = np.random.default_rng(42)

        for _ in range(20):
            lam_self = rng.uniform(0.0, 1.0)
            lam_bias = rng.uniform(0.0, 1.0)
            lam_init = rng.uniform(0.0, 1.0)

            total = lam_self + lam_bias + lam_init
            lam_self /= total
            lam_bias /= total
            lam_init /= total

            bias_tilde = lam_bias * rng.uniform(-1.0, 1.0)
            param_list.append((lam_self, lam_init, bias_tilde))

        n_runs = 10
        horizon = 10
        n = 10

        for p in param_list:
            l_self, l_init, bias_tilde = p

            run_traj = _simulate_bias_init_only(lambda_self=l_self, lambda_init=l_init, bias_tilde=bias_tilde,
                                                n_runs=n_runs, horizon=horizon, n=n, rng=rng)
            
            fit_result = fit_bias_init_only_model(run_traj_map=run_traj)

            if abs(bias_tilde) < 1e-04:
                # in this event self-weight and init-weight effects won't generally be identifiable
                self.assertAlmostEqual(fit_result["lambda_self"] + fit_result["lambda_init"], l_self + l_init, places=4)
                self.assertLessEqual(fit_result["mse_pool"], 1e-04)

            else:

                self.assertAlmostEqual(fit_result["lambda_self"], l_self, places=4)
                self.assertAlmostEqual(fit_result["lambda_init"], l_init, places=4)
                self.assertAlmostEqual(fit_result["lambda_bias"] * fit_result["bias"], bias_tilde, places=4)
                self.assertLessEqual(fit_result["mse_pool"], 1e-04)
            
    def test_against_brute_force(self):


        num_cases = 5
        T = 10
        n = 10

        rng = np.random.default_rng(10)

        for _ in range(num_cases):
            traj = rng.uniform(-1.0, 1.0, size=(T,n))
            run_traj = {'run_00': traj}

            gs_solution = _grid_search_bias_init(traj, grid_points=21)
            fit_result = fit_bias_init_only_model(run_traj_map=run_traj)

            l_self_gs = gs_solution["lambda_self"]
            l_init_gs = gs_solution["lambda_init"]
            b_tilde_gs = gs_solution["bias_tilde"]

            l_self_cvx = fit_result["lambda_self"]
            l_init_cvx = fit_result["lambda_init"]
            b_tilde_cvx = fit_result["lambda_bias"]*fit_result["bias"]

            self.assertTrue(abs(l_self_gs - l_self_cvx) < 0.1)
            self.assertTrue(abs(l_init_gs - l_init_cvx) < 0.1)
            self.assertTrue(abs(b_tilde_gs - b_tilde_cvx) < 0.1)


        # repeat with data generated from sim rollouts with added noise
        for _ in range(num_cases):


            l_self_gen = rng.uniform(0.0,1.0)
            l_init_gen = rng.uniform(0.0,1.0)
            l_bias_gen = rng.uniform(0.0, 1.0)

            total = l_self_gen + l_init_gen + l_bias_gen

            l_self_gen /= total
            l_init_gen /= total
            l_bias_gen /= total
            bias_tilde_gen = rng.uniform(-1.0,1.0)*l_bias_gen

            traj = _simulate_bias_init_only(lambda_self=l_self_gen, bias_tilde=bias_tilde_gen, lambda_init=l_init_gen,n_runs=1,
                                          horizon=T - 1, n=n, rng=rng)['run_00']
            traj = traj + rng.uniform(-0.1, 0.1, size=(T,n))
            traj = np.clip(traj, -1.0, 1.0)


            run_traj = {'run_00': traj}

            gs_solution = _grid_search_bias_init(traj, grid_points=41)
            fit_result = fit_bias_init_only_model(run_traj_map=run_traj)

            l_self_gs = gs_solution["lambda_self"]
            l_init_gs = gs_solution["lambda_init"]
            b_tilde_gs = gs_solution["bias_tilde"]

            l_self_cvx = fit_result["lambda_self"]
            l_init_cvx = fit_result["lambda_init"]
            b_tilde_cvx = fit_result["lambda_bias"]*fit_result["bias"]

            self.assertTrue(abs(l_self_gs - l_self_cvx) < 0.1)
            self.assertTrue(abs(l_init_gs - l_init_cvx) < 0.1)
            self.assertTrue(abs(b_tilde_gs - b_tilde_cvx) < 0.1)

class TestClosedFormSolution(unittest.TestCase):

    def test_closed_form_solution(self):

        run_traj = {
            'run_00': np.array([[1, -1.0],
                                [0.0, 0.0],
                                [0.0, 0.0]])
        }

        fit_result = fit_bias_init_only_model(run_traj)
        self.assertAlmostEqual(fit_result["lambda_self"], 0.0, places=4)
        self.assertAlmostEqual(fit_result["lambda_init"], 0.0, places=4)
        self.assertAlmostEqual(fit_result["lambda_bias"]* fit_result["bias"], 0.0, places=4)
        self.assertAlmostEqual(fit_result["mse_pool"], 0.0, places=4)

        run_traj = {
            'run_00': np.array([[1.0, 0.0, -1.0],
                                [0.0, 0.0, 0.0],])
        }
        fit_result = fit_bias_init_only_model(run_traj)
        self.assertAlmostEqual(fit_result["lambda_self"], 0.0, places=4)
        self.assertAlmostEqual(fit_result["lambda_init"], 0.0, places=4)
        self.assertAlmostEqual(fit_result["lambda_bias"]* fit_result["bias"], 0.0, places=4)
        self.assertAlmostEqual(fit_result["mse_pool"], 0.0, places=4)

        run_traj = {
            'run_00': np.array([[1.0, 0.0, -1.0],
                                [1.0, 0.0, 0.0],])
        }
        fit_result = fit_bias_init_only_model(run_traj)
        self.assertAlmostEqual(fit_result["lambda_self"] + fit_result["lambda_init"], 1/2, places=4)
        self.assertAlmostEqual(fit_result["lambda_bias"]* fit_result["bias"], 1/3, places=4)
        self.assertAlmostEqual(fit_result["mse_pool"], 1/18, places=4)

        run_traj = {
            'run_00': np.array([[1, -1.0],
                                [0.0, 0.0],
                                [0.0, 0.0],
                                [1.0, -1.0]])
        }
        fit_result = fit_bias_init_only_model(run_traj)
        self.assertAlmostEqual(fit_result["lambda_self"], 0.0, places=4)
        self.assertAlmostEqual(fit_result["lambda_init"], 1/3, places=4)
        self.assertAlmostEqual(fit_result["lambda_bias"]* fit_result["bias"], 0.0, places=4)
        self.assertAlmostEqual(fit_result["mse_pool"], 2/9, places=4)

class TestRolloutFunction(unittest.TestCase):


    def test_hand_computed_solution(self):

        # test cases from no-init scripts (check they work when l_init = 0)

        # check that rollout produces solutions that follow basic 
        #  geometric decay
        x0 = np.array([-1.0, 0.5])

        traj = bias_init_only_rollout(lambda_self=0.5, lambda_bias=0.5, bias = 0.0, lambda_init=0.0,
                                 x0 = x0, horizon =10)
        self.assertEqual(np.shape(traj), (11, 2))

        self.assertTrue(np.allclose(traj[:3,:],
                        np.array([[-1.0, 0.5],
                                  [-0.5, 0.25],
                                  [-0.25,0.125]])))

        # pure bias
        traj = bias_init_only_rollout(lambda_self=0.0, lambda_bias=1.0, bias = -0.5, lambda_init=0.0,
                                 x0 = x0, horizon =5)

        self.assertTrue(np.allclose(traj[:3,:],
                        np.array([[-1.0, 0.5],
                                  [-0.5, -0.5],
                                  [-0.5,-0.5]])))
        
        traj = bias_init_only_rollout(lambda_self=0.0, lambda_bias=1.0, bias = 0.5, lambda_init=0.0,
                                 x0 = x0, horizon =5)

        self.assertTrue(np.allclose(traj[:3,:],
                        np.array([[-1.0, 0.5],
                                  [0.5, 0.5],
                                  [0.5,0.5]])))
        
        # non-zero bias with decay
        x0 = np.array([-1.0, 0.0, 1.0])
        traj = bias_init_only_rollout(lambda_self=2/3, lambda_bias=1/3, bias = 1.0, lambda_init=0.0,
                                 x0 = x0, horizon =10)
        self.assertTrue(np.allclose(traj[:3,:],
                        np.array([[-1.0, 0.0, 1.0],
                                  [-1/3, 1/3, 1.0],
                                  [1/9,5/9, 1.0]])))
        
        # pure init weight
        x0 = np.array([-1.0, 0.0, 1.0])
        traj = bias_init_only_rollout(lambda_self=0, lambda_bias=0, bias = 0, lambda_init=1.0,
                                 x0 = x0, horizon =10)
        self.assertTrue(np.allclose(traj,
                                    np.repeat(np.array([[-1.0, 0.0, 1.0]]),11, axis=0)))

        # purely init and bias: 
        x0 = np.array([-1.0, 0.0, 1.0])
        traj = bias_init_only_rollout(lambda_self=0, lambda_bias=0.5, bias = 0.5, lambda_init=0.5,
                                 x0 = x0, horizon =10)
        
        expected_traj = np.vstack((x0, np.repeat(np.array([[-0.25, 0.25, 0.75]]),10, axis=0)))
        self.assertTrue(np.allclose(traj,
                                    expected_traj))

        # even mix of init and bias and self
        x0 = np.array([1.0, 0.0, -1.0])
        traj = bias_init_only_rollout(lambda_self=1/3, lambda_bias=1/3, bias = 0.5, lambda_init=1/3,
                                 x0 = x0, horizon =10)

        expected_traj_3 = np.array([[1.0,0.0,-1.0],
                                    [5/6,1/6,-1/2],
                                    [7/9,2/9,-1/3]])
        self.assertTrue(np.allclose(traj[:3,:],
                                    expected_traj_3))
        
        # init and bias cancel out
        x0 = np.array([1.0, -1.0])
        traj = bias_init_only_rollout(lambda_self=1/3, lambda_bias=1/3, bias = -1.0, lambda_init=1/3,
                                 x0 = x0, horizon =10)

        expected_traj_3 = np.array([[1.0,-1.0],
                                    [1/3,-1.0],
                                    [1/9,-1.0]])

    def test_matches_simulation(self):
        
        
        num_samples = 30
        n=10

        param_list = []
        # specifying cases with convention (l_self, l_init, bias_tilde)

        # all cases with only one effect
        param_list.append((1.0, 0.0, 0.0))
        param_list.append((0.0, 1.0, 0.0))
        param_list.append((0.0, 0.0, 0.0))  # no bias
        param_list.append((0.0, 0.0, 1.0))  # + bias
        param_list.append((0.0, 0.0, -1.0)) # - bias

        # all cases with only two effects
        param_list.append((0.4, 0.5, 0.0))
        param_list.append((0.3, 0.0, 0.2))  # + bias
        param_list.append((0.3, 0.0, -0.2)) # - bias
        param_list.append((0.0, 0.5, 0.5))  # + bias
        param_list.append((0.0, 0.6, -0.4)) # - bias

        rng = np.random.default_rng(42)

        for _ in range(num_samples):
            lam_self = rng.uniform(0.0, 1.0)
            lam_bias = rng.uniform(0.0, 1.0)
            lam_init = rng.uniform(0.0, 1.0)

            total = lam_self + lam_bias + lam_init
            lam_self /= total
            lam_bias /= total
            lam_init /= total

            bias_tilde = lam_bias * rng.uniform(-1.0, 1.0)
            param_list.append((lam_self, lam_init, bias_tilde))


        for p in param_list:
            l_self, l_init, bias_tilde = p

            l_bias = 1.0 - l_self - l_init
            if l_bias > 0:
                bias = bias_tilde / l_bias
            else:
                bias = 0

            x0 = rng.uniform(-1.0, 1.0, size = n)
            traj_map = _simulate_bias_init_only(lambda_init=l_init,
                                                lambda_self=l_self,
                                                bias_tilde = bias_tilde,
                                                n_runs=1,
                                                horizon=10,
                                                n=n,
                                                x0=x0)
            
            traj_rollout = bias_init_only_rollout(lambda_self=l_self,
                                                  lambda_bias=l_bias,
                                                  lambda_init=l_init,
                                                  bias=bias,
                                                  x0=x0,
                                                  horizon=10,
                                                  )
            
            self.assertTrue(np.allclose(traj_map['run_00'], traj_rollout))


if __name__ == "__main__":

    unittest.main()