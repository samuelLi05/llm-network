import cvxpy as cp
import numpy as np

from modeling.models.data_prep import build_dataset_from_run, build_expected_message_matrix, _grid_search_with_refinement
from modeling.models.adjacency_based.repulsion import _prepare_pooled_blocks_rep
# TODO: pick a repulsion model (possibly may need to move _prepare_pooled_blocks_rep around for that)

def _get_generic_social_kernel_term_force_based(x, neighbors, F, bound_type = 'clip'):

    # : x : np.array of shape (n_agents, ) representing the opinions of agents
    # : neighbors : list of lists, where neighbors[i] is a list of indices of agents who influence agent i
    # : F : a kernel function that takes a non-negative float and returns a float
    #           which will be used to weight neighbors' opinions
    # : bound_type : str, either 'clip' or 'tanh', which determines how to bound the upda

    n_agents = len(x)


    for i in range(n_agents):

        nbh = list(neighbors[i])

        weights = [F(abs(x[j] - x[i])) for j in nbh]
        nbh_delta = [x[j] - x[i] for j in nbh]

        # normalize the weights so the absolute values sum to 1, if there are any weights
        if len(weights) == 0:
            raise ValueError(f"Agent {i} has no neighbors, cannot compute social kernel term.")
        
        weights = np.array(weights)
        weights /= np.sum(np.abs(weights))

        x[i] = np.sum([weights[j] * nbh_delta[j] for j in range(len(nbh))])


def _get_generic_social_kernel_term_weight_based(x, Abar, F):
    
    # Get the 'social term' for opinion dynamics models where a force term is used to weight opinion updates
    #  (generalizing )

    # : x : np.array of shape (n_agents, ) representing the opinions of agents
    # : Abar : row normalized adjacency matrix
    # : F : a kernel function that takes a non-negative float and returns a float
    #           which will be used to weight neighbors' opinions

    n_agents = len(x)
    x_social = np.zeros(n_agents)
    for i in range(n_agents):

        weights = Abar[i,:] * np.array([F(abs(x[j] - x[i])) for j in range(n_agents)])
        if np.sum(weights) == 0:
            raise ValueError(f"Agent {i} has no neighbors, cannot compute social kernel term.")

        weights /= np.sum(np.abs(weights))

        x_social[i] = np.sum(weights * x)

    return np.array(x_social)

def fit_repulsion_fj_bias_mult(run_traj_map, run_neighbors, repulsion_version: str,
                               opt_eps = 1e-09, custom_search_values = None):
    
    # Solve for the optimal model with a repulsion term.
    #  repulsion term type depends on the repulsion_version flag which should come 
    #  from ("weight-based", "force-based")
    # 
    # Repulsion effects basically weight each neighbor 
    #  proportional to 1- (\beta)|x(j) - x(i)|, with a sign for that 

    if not (repulsion_version in ("weight-based", "force_based")):
        raise ValueError("Invalid setting for fitting method")
    
    data_prepped = _prepare_pooled_blocks_rep(run_traj_map, run_neighbors)
    x_pool = data_prepped["x_pool"]
    y_pool = data_prepped["y_pool"]
    x0_pool = data_prepped["x0_pool"]
    x_blocks = data_prepped["x_blocks"]
    abar_blocks = data_prepped["abar_blocks"]

    def _F_kernel(beta_rep , d: float):
        if d < 0:
            raise ValueError("Negative value sent to kernel")
        return 1 - beta_rep * d
    # TODO: implement more kernels

    def _solve_for_rep(beta_rep_fixed: float):

        _F_kernel_beta = lambda d : _F_kernel(beta_rep_fixed, d)

        beta_rep_fixed = float(beta_rep_fixed)

        if repulsion_version == "weight-based":
            rep_blocks = [
                np.asarray([_get_generic_social_kernel_term_weight_based(x_blocks[i][t], abar_blocks[i], _F_kernel_beta) for t in range(x_blocks[i].shape[0])], dtype=float)
                for i in range(len(x_blocks))
            ]
        elif repulsion_version == "force_based":
            raise NotImplementedError("Force-based repulsion not implemented yet")
        
        rep_pool = np.vstack(rep_blocks)

        lambda_self_var = cp.Variable(nonneg=True)
        lambda_social_var = cp.Variable(nonneg=True)
        lambda_init_var = cp.Variable(nonneg=True)
        lambda_bias_var = cp.Variable(nonneg=True)

        bias_tilde_var = cp.Variable()  # variable representing lambda_bias * bias, in reformulation

        pred_pool = (lambda_self_var * x_pool + 
                     lambda_social_var * rep_pool + 
                     lambda_init_var * x0_pool + 
                     bias_tilde_var)
    
        objective = cp.Minimize(cp.sum_squares(pred_pool - y_pool))
        constraints = [
            lambda_self_var + lambda_social_var + lambda_init_var + lambda_bias_var == 1,
            bias_tilde_var <= lambda_bias_var,
            bias_tilde_var >= - lambda_bias_var
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, eps_abs=opt_eps, eps_rel=opt_eps, verbose=False)

        if (
            lambda_self_var.value is None or
            lambda_social_var.value is None or
            lambda_init_var.value is None or
            lambda_bias_var.value is None or
            bias_tilde_var.value is None
        ):
            raise RuntimeError("Optimization failure")
        
        # clamp all values to [0,1] and renormalize
        lambda_self = max(0, min(1, lambda_self_var.value))
        lambda_social = max(0, min(1, lambda_social_var.value))
        lambda_init = max(0, min(1, lambda_init_var.value))
        lambda_bias = max(0, min(1, lambda_bias_var.value))

        total = lambda_self + lambda_social + lambda_init + lambda_bias

        if total > 0:
            lambda_self /= total
            lambda_social /= total
            lambda_init /= total
            lambda_bias /= total
        else:
            raise ValueError("All lambda values are zero after clamping, cannot normalize.")
        
        bias_tilde = max(-lambda_bias,  min(lambda_bias, bias_tilde_var.value))
 
        bias = bias_tilde / lambda_bias if lambda_bias > 0 else 0.0 

        fitted_pool = lambda_self * x_pool + lambda_social * rep_pool + lambda_init * x0_pool + lambda_bias * bias

        mse_pool = float(np.mean((fitted_pool - y_pool) ** 2))
        mse_pool_sum = float(np.sum((fitted_pool - y_pool) ** 2))

        if abs(mse_pool_sum - problem.value > 1e-06):
            raise ValueError(f"Computed mse_pool_sum {mse_pool_sum} is not close to the optimization value {problem.value}")
        
        solver_iters = -1
        if problem.solver_stats is not None and problem.solver_stats.num_iters is not None:
            solver_iters = int(problem.solver_stats.num_iters)

        candidate = {
            "beta_rep": beta_rep_fixed,
            "lambda_self": lambda_self,
            "lambda_social": lambda_social,
            "lambda_init": lambda_init,
            "lambda_bias": lambda_bias,
            "bias": bias,
            "mse_pool": mse_pool,
            "status": str(problem.status),
            "success": bool(problem.status in (cp.OPTIMAL)),
            "nit": solver_iters,
            "objective": float(problem.value) if problem.value is not None else mse_pool
        }

        return candidate

    def beta_objective(beta_fixed:float) -> float:
        candidate = _solve_for_rep(beta_fixed)
        return candidate["mse_pool"]
    
    beta_hat, beta_objective_map = _grid_search_with_refinement(beta_objective, 
                                 bounds = (0.0,2.0),
                                 coarse_points=100,
                                 fine_points = 100,
                                 fine_search_region_width=0.1,
                                 bisection_init_region_width=0.01,
                                 bisection_eps=1e-06,
                                 geom_search_bnds=(1e-03, 1e02),
                                 custom_search_vals=custom_search_values)
    
    best_result = _solve_for_rep(beta_hat)
    if best_result is None:
        raise ValueError("Failed to find a valid solution for the best theta.")

    total_points = np.shape(y_pool)[0] * np.shape(y_pool)[1]

    return {
        "beta_rep": float(best_result["beta_rep"]),
        "lambda_self": float(best_result["lambda_self"]),
        "lambda_social": float(best_result["lambda_social"]),
        "lambda_init": float(best_result["lambda_init"]),
        "lambda_bias": float(best_result["lambda_bias"]),
        "bias": float(best_result["bias"]),
        "mse_pool": float(best_result["mse_pool"]),
        "status": str(best_result["status"]),
        "success": bool(best_result["success"]),
        "nit": int(best_result["nit"]),
        "objective": float(best_result["objective"]),
        "beta_objective_map": beta_objective_map,
        "total_points": int(total_points)
    }
