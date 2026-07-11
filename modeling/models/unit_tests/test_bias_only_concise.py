import unittest
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from modeling.models.adjacency_based.bias_only import (
    fit_bias_only_model,
    bias_only_rollout
)

def _simulate_bias_only(lambda_self, bias_tilde, n_runs, horizon, n,
                        x0 = None, rng = None):
    
    run_traj = {}

    for r in range(n_runs):
        if x0 is None:
            x = rng.uniform(-1.0,1.0, size=n)
        else:
            x = x0.copy()

        states = [x.copy()]
        for _ in range(horizon):
            x = lambda_self * x + bias_tilde
            states.append(x.copy())
        run_traj[f'run_{r:02d}'] = np.asarray(states)
    return run_traj

def _grid_search(traj, grid_points):
    # helper to fit parameters using a brute force search from a single trajectory

    T,n = np.shape(traj)

    x = traj[:T-1, :]
    y = traj[1:, :]

    best_mse = np.inf
    best_sol = (np.nan, np.nan)
    
    lambda_self_search_values = np.linspace(0.0, 1.0, grid_points)
    bias_tilde_search_values = np.linspace(-1.0, 1.0, 2*grid_points)

    for l_self in lambda_self_search_values:
        for bias_tilde in bias_tilde_search_values:
            if l_self + abs(bias_tilde) <= 1.0:

                mse = np.sum((x * l_self + bias_tilde - y)**2)

                if mse < best_mse:
                    best_sol = (l_self, bias_tilde)
                    best_mse = mse

    if np.isinf(best_mse):
        raise ValueError("No non-infinite solution found")
    
    return best_sol





class TestBiasRecovery(unittest.TestCase):

    def test_bias_recovery(self):

        param_list = []

        param_list.append((1.0, 0.0))
        param_list.append((0.5, 0.5))
        param_list.append((0.5, -0.5))
        param_list.append((0.0, 0.0))
        param_list.append((0.0, -1.0))
        param_list.append((0.0, 1.0))

        rng = np.random.default_rng(42)

        for _ in range(10):
            lam_self = rng.uniform(0.0,1.0)
            lam_bias = 1.0 - lam_self
            bias_tilde = lam_bias * rng.uniform(-1.0,1.0)
            param_list.append((lam_self,bias_tilde))

        n_runs = 10
        horizon = 10
        n = 10

        for p in param_list:
            l_self, bias_tilde = p

            run_traj= _simulate_bias_only(lambda_self=l_self, bias_tilde=bias_tilde,n_runs=n_runs,
                                          horizon=horizon, n=n, rng=rng)
            
            fit_result = fit_bias_only_model(run_traj_map=run_traj)

            self.assertAlmostEqual(fit_result["lambda_self"], l_self, places=4)
            self.assertAlmostEqual(fit_result["lambda_bias"] * fit_result["bias"], bias_tilde, places=4)
            self.assertLessEqual(fit_result["mse_pool"], 1e-04)

    def test_against_brute_force(self):

        num_cases = 5
        T = 10
        n = 5

        rng = np.random.default_rng(10)

        # generate random fitting data and check that 
        for _ in range(num_cases):
            traj = rng.uniform(-1.0,1.0, size = (T, n))

            run_traj = {'run_00': traj}

            grid_search_soln = _grid_search(traj, grid_points = 11)
            
            fit_result = fit_bias_only_model(run_traj_map=run_traj)

            l_self_gs, b_tilde_gs = grid_search_soln
            l_self_cvx = fit_result["lambda_self"]
            b_tilde_cvx = fit_result["lambda_bias"]*fit_result["bias"]

            self.assertTrue(abs(l_self_gs - l_self_cvx) < 0.1)
            self.assertTrue(abs(b_tilde_gs - b_tilde_cvx) < 0.1)

        # repeat with data generated from sim rollouts with added noise
        for _ in range(num_cases):
            l_self_gen = rng.uniform(0.0,1.0)
            bias_tilde_gen = rng.uniform(-1.0,1.0)*(1.0 - l_self_gen)

            traj = _simulate_bias_only(lambda_self=l_self_gen, bias_tilde=bias_tilde_gen,n_runs=1,
                                          horizon=T - 1, n=n, rng=rng)['run_00']
            traj = traj + rng.uniform(-0.1, 0.1, size=(T,n))
            traj = np.clip(traj, -1.0, 1.0)



            run_traj = {'run_00': traj}

            grid_search_soln = _grid_search(traj, grid_points = 21)
            
            fit_result = fit_bias_only_model(run_traj_map=run_traj)

            l_self_gs, b_tilde_gs = grid_search_soln
            l_self_cvx = fit_result["lambda_self"]
            b_tilde_cvx = fit_result["lambda_bias"]*fit_result["bias"]

            self.assertTrue(abs(l_self_gs - l_self_cvx) < 0.1)
            self.assertTrue(abs(b_tilde_gs - b_tilde_cvx) < 0.1)



class TestClosedFormSolution(unittest.TestCase):

    def test_closed_form_solution(self):

        # construct a manual run (2 agents, 2 steps)
        run_traj = {
            'run_00': np.array([[1, -0.5],
                                [0.0, 0.0]])
        }

        fit_result = fit_bias_only_model(run_traj)
        self.assertAlmostEqual(fit_result["lambda_self"], 0.0)
        self.assertAlmostEqual(fit_result["lambda_bias"]* fit_result["bias"], 0.0)

        # 3 agents, 2 time steps
        run_traj = {
            'run_00': np.array([[1, 1, -1],
                                [1, 0.0, 0.0]])
        }

        fit_result = fit_bias_only_model(run_traj)
        self.assertAlmostEqual(fit_result["lambda_self"], 1/4)
        self.assertAlmostEqual(fit_result["lambda_bias"]* fit_result["bias"], 1/4)
        self.assertAlmostEqual(fit_result["lambda_bias"], 3/4)

        # 2 agents, 3 timesteps
        run_traj = {
            'run_00': np.array([[1, 0.0],
                                [1, 0.0],
                                [0.0, 0.0]])
        }

        fit_result = fit_bias_only_model(run_traj)
        self.assertAlmostEqual(fit_result["lambda_self"], 1/2)
        self.assertAlmostEqual(fit_result["lambda_bias"]* fit_result["bias"], 0.0)
        self.assertAlmostEqual(fit_result["lambda_bias"], 1/2)
        self.assertAlmostEqual(fit_result["bias"], 0.0)
        self.assertAlmostEqual(fit_result["mse_pool"], 1/8)

        # 1 agent, 4 time steps
        run_traj = {
            'run_00': np.array([[0],
                                [0],
                                [1],
                                [1],])
        }

        fit_result = fit_bias_only_model(run_traj)
        self.assertAlmostEqual(fit_result["lambda_self"], 1/2)
        self.assertAlmostEqual(fit_result["lambda_bias"]* fit_result["bias"], 1/2)
        self.assertAlmostEqual(fit_result["lambda_bias"], 1/2)
        self.assertAlmostEqual(fit_result["bias"], 1.0)
        self.assertAlmostEqual(fit_result["mse_pool"], 1/6)

        # 2 agents, 1 timestep, constraint active at OPT
        run_traj = {
            'run_00': np.array([[0.0, -1.0],
                                [1.0, 0.0]])
        }
        fit_result = fit_bias_only_model(run_traj)
        self.assertAlmostEqual(fit_result["lambda_self"], 0.4)
        self.assertAlmostEqual(fit_result["lambda_bias"]* fit_result["bias"], 0.6)
        self.assertAlmostEqual(fit_result["lambda_bias"], 0.6)
        self.assertAlmostEqual(fit_result["bias"], 1.0)
        self.assertAlmostEqual(fit_result["mse_pool"], 0.1)

        run_traj = {
            'run_00': np.array([[0.0, 1.0],
                                [-1.0, 0.0]])
        }
        fit_result = fit_bias_only_model(run_traj)
        self.assertAlmostEqual(fit_result["lambda_self"], 0.4)
        self.assertAlmostEqual(fit_result["lambda_bias"]* fit_result["bias"], -0.6)
        self.assertAlmostEqual(fit_result["lambda_bias"], 0.6)
        self.assertAlmostEqual(fit_result["bias"], -1.0)
        self.assertAlmostEqual(fit_result["mse_pool"], 0.1)



class TestRolloutFunction(unittest.TestCase):

    def test_hand_computed_solution(self):
        # check that rollout produces solutions that follow basic 
        #  geometric decay
        x0 = np.array([-1.0, 0.5])

        traj = bias_only_rollout(lambda_self=0.5, lambda_bias=0.5, bias = 0.0,
                                 x0 = x0, horizon =10)
        self.assertEqual(np.shape(traj), (11, 2))

        self.assertTrue(np.allclose(traj[:3,:],
                        np.array([[-1.0, 0.5],
                                  [-0.5, 0.25],
                                  [-0.25,0.125]])))

        # pure bias
        traj = bias_only_rollout(lambda_self=0.0, lambda_bias=1.0, bias = -0.5,
                                 x0 = x0, horizon =5)

        self.assertTrue(np.allclose(traj[:3,:],
                        np.array([[-1.0, 0.5],
                                  [-0.5, -0.5],
                                  [-0.5,-0.5]])))
        
        traj = bias_only_rollout(lambda_self=0.0, lambda_bias=1.0, bias = 0.5,
                                 x0 = x0, horizon =5)

        self.assertTrue(np.allclose(traj[:3,:],
                        np.array([[-1.0, 0.5],
                                  [0.5, 0.5],
                                  [0.5,0.5]])))
        
        # non-zero bias with decay
        x0 = np.array([-1.0, 0.0, 1.0])
        traj = bias_only_rollout(lambda_self=2/3, lambda_bias=1/3, bias = 1.0,
                                 x0 = x0, horizon =10)
        self.assertTrue(np.allclose(traj[:3,:],
                        np.array([[-1.0, 0.0, 1.0],
                                  [-1/3, 1/3, 1.0],
                                  [1/9,5/9, 1.0]])))
        


    def test_matches_simulation(self):

        num_samples = 10
        rng = np.random.default_rng(10)
        n = 10

        for _ in range(num_samples):

            l_self = rng.uniform(0.0,1.0)
            l_bias = 1 - l_self
            bias = rng.uniform(-1.0,1.0)* l_bias

            x0 = rng.uniform(-1.0, 1.0, size = n)

            traj_map = _simulate_bias_only(l_self, l_bias*bias, n_runs=1,horizon=10,n=n,x0=x0)
            traj_rollout = bias_only_rollout(lambda_self=l_self,lambda_bias=l_bias, bias=bias, x0=x0, horizon = 10)
            
            self.assertTrue(np.allclose(traj_map['run_00'], traj_rollout))



if __name__ == "__main__":
    unittest.main()
