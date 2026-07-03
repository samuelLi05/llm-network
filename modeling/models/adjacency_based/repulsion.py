import cvxpy as cp
import numpy as np

from modeling.models.data_prep import build_dataset_from_run, build_expected_message_matrix, _grid_search_with_refinement

def _get_repulsion_term_tanh(x, neighbors, theta_rep):

    # x : np.array of shape (n_agents, ) representing the opinions of agents
    # neighbors : list of lists, where neighbors[i] is a list of indices of neighbors of agent i, who influence agent i
    # theta_rep : float, the repulsion threshold

    n_agents = len(x)
    pre_tanh_repulsion_term = np.zeros((n_agents))

    for i in range(n_agents):
        for j in neighbors[i]:
            if abs(x[j] - x[i]) > theta_rep:
                pre_tanh_repulsion_term[i] += x[j] - x[i]
    
    return np.tanh(pre_tanh_repulsion_term)


def _get_generic_social_kernel_term(x, neighbors, F):

    # : x : np.array of shape (n_agents, ) representing the opinions of agents
    # : neighbors : list of lists, where neighbors[i] is a list of indices of neighbors of agent i, who influence agent i
    # : F : a kernel function that takes a non-negative float and returns a float. 
    #           which will be used to weight neighbors' opinions

    n_agents = len(x)

    for i in range(n_agents):

        nbh = list(neighbors[i])

        weights = [F(abs(x[j] - x[i])) for j in nbh]
        nbh_op = [x[j] for j in nbh]

        # normalize the weights so the absolute values sum to 1, if there are any weights
        if len(weights) == 0:
            raise ValueError(f"Agent {i} has no neighbors, cannot compute social kernel term.")
        
        weights = np.array(weights)
        weights /= np.sum(np.abs(weights))

        x[i] = np.sum([weights[j] * nbh_op[j] for j in range(len(nbh))])

    



def _prepare_pooled_blocks_rep(run_traj_map, run_neighbors):
    run_names = sorted(run_traj_map.keys())
    x_blocks = []
    y_blocks = []
    x0_blocks = []

    for rn in run_names:
        traj = np.asarray(run_traj_map[rn], dtype=float)
        x, y = build_dataset_from_run(traj)
        x_blocks.append(x)
        y_blocks.append(y)
        x0_blocks.append(np.repeat(traj[0].reshape(1, -1), x.shape[0], axis=0))
    
    x_pool = np.vstack(x_blocks)
    y_pool = np.vstack(y_blocks)
    x0_pool = np.vstack(x0_blocks)

    n = x_pool.shape[1]
    abar_blocks = [build_expected_message_matrix(run_neighbors.get(rn, {}), n) for rn in run_names]
    nbrs_list = [run_neighbors.get(rn, {}) for rn in run_names]
    xa_blocks = [x_blocks[i] @ abar_blocks[i].T for i in range(len(x_blocks))]


    
    return {
        "run_names": run_names,
        "x_blocks": x_blocks,
        "y_blocks": y_blocks,
        "x0_blocks": x0_blocks,
        "x_pool": x_pool,
        "y_pool": y_pool,
        "x0_pool": x0_pool,
        "abar_blocks": abar_blocks,
        "xa_blocks": xa_blocks,
        "n": n,
        "nbrs_list": nbrs_list,
        "xa_pool": np.vstack(xa_blocks)
    }


def fit_repulsion_joint_add(run_traj_map, run_neighbors, eps=1e-09, opt_eps=1e-09):

    # Solve for the optimal model with a repulsion term
    #  where the repulsion term is tanh(\sum_{j \in N(i) : |x(j) - x(i)| > theta_rep} (x(j) - x(i)))

    data_prepped = _prepare_pooled_blocks_rep(run_traj_map, run_neighbors)
    x_pool = data_prepped["x_pool"]
    y_pool = data_prepped["y_pool"]
    xa_pool = data_prepped["xa_pool"]
    x_blocks = data_prepped["x_blocks"]
    nbrs_list = data_prepped["nbrs_list"]


    def _solve_for_rep(theta_rep_fixed: float):
        theta_rep_fixed = float(theta_rep_fixed)

        rep_blocks = [
            np.asarray([_get_repulsion_term_tanh(x_blocks[i][t], nbrs_list[i], theta_rep_fixed) for t in range(x_blocks[i].shape[0])],dtype=float)
            for i in range(len(x_blocks))
        ]
        rep_pool = np.vstack(rep_blocks)

        lambda_self_var = cp.Variable(nonneg=True)
        lambda_social_var = cp.Variable(nonneg=True)
        lambda_repulsion_var = cp.Variable(nonneg=True)

        pred_pool = lambda_self_var * x_pool + lambda_social_var * xa_pool + lambda_repulsion_var * rep_pool

        objective = cp.Minimize(cp.sum_squares(pred_pool - y_pool))
        constraints = [lambda_self_var + lambda_social_var + lambda_repulsion_var == 1]
        problem = cp.Problem(objective, constraints)

        problem.solve(solver=cp.OSQP, eps_abs=opt_eps, eps_rel=opt_eps, verbose=False)

        if lambda_self_var.value is None or lambda_social_var.value is None or lambda_repulsion_var.value is None:
            raise ValueError("Optimization failed to find a solution.")
        
        # clamp all values to [0, 1] and renormalize
        lambda_self = max(0, min(1, lambda_self_var.value))
        lambda_social = max(0, min(1, lambda_social_var.value))
        lambda_repulsion = max(0, min(1, lambda_repulsion_var.value))

        total = lambda_self + lambda_social + lambda_repulsion
        if total > 0:
            lambda_self /= total
            lambda_social /= total
            lambda_repulsion /= total
        else:
            raise ValueError("All lambda values are zero after clamping, cannot normalize.")
        
        fitted_pool = lambda_self * x_pool + lambda_social * xa_pool + lambda_repulsion * rep_pool
        mse_pool = float(np.mean((fitted_pool - y_pool) ** 2))

        solver_iters = -1
        if problem.solver_stats is not None and problem.solver_stats.num_iters is not None:
            solver_iters = int(problem.solver_stats.num_iters)

        candidate = {
            "theta_rep": theta_rep_fixed,
            "lambda_self": lambda_self,
            "lambda_social": lambda_social,
            "lambda_repulsion": lambda_repulsion,
            "mse_pool": mse_pool,
            "status":  str(problem.status),
            "success": bool(problem.status in (cp.OPTIMAL)),
            "nit": solver_iters,
            "objective": float(problem.value) if problem.value is not None else mse_pool
        }

        return candidate

    def theta_objective(theta_fixed: float) -> float:
        candidate = _solve_for_rep(theta_fixed)
        return candidate["mse_pool"]
    
    theta_hat, theta_objective_map = _grid_search_with_refinement(
        theta_objective, 
        bounds = (0.0, 2.0))

    best_result = _solve_for_rep(theta_hat)
    if best_result is None:
        raise ValueError("Failed to find a valid solution for the best theta.")
    
    total_points = np.shape(y_pool)[0] * np.shape(y_pool)[1]

    return {
        "theta_rep": float(best_result["theta_rep"]),
        "lambda_self": float(best_result["lambda_self"]),
        "lambda_social": float(best_result["lambda_social"]),
        "lambda_repulsion": float(best_result["lambda_repulsion"]),
        "mse_pool": float(best_result["mse_pool"]),
        "status": str(best_result["status"]),
        "success": bool(best_result["success"]),
        "nit": int(best_result["nit"]),
        "objective": float(best_result["objective"]),
        "theta_objective_map": theta_objective_map,
        "total_points": int(total_points)
    }

def fit_friedkin_johnsen_bias_tanh_repulsion_joint(run_traj_map, run_neighbors, eps=1e-09, opt_eps=1e-09):

    # Solve for the optimal model with a repulsion term
    #  where the repulsion term is tanh(\sum_{j \in N(i) : |x(j) - x(i)| > theta_rep} (x(j) - x(i)))

    data_prepped = _prepare_pooled_blocks_rep(run_traj_map, run_neighbors)
    x_pool = data_prepped["x_pool"]
    y_pool = data_prepped["y_pool"]
    x0_pool = data_prepped["x0_pool"]
    xa_pool = data_prepped["xa_pool"]
    x_blocks = data_prepped["x_blocks"]
    nbrs_list = data_prepped["nbrs_list"]

    def _solve_for_rep(theta_rep_fixed: float):
        theta_rep_fixed = float(theta_rep_fixed)

        rep_blocks = [
            np.asarray([_get_repulsion_term_tanh(x_blocks[i][t], nbrs_list[i], theta_rep_fixed) for t in range(x_blocks[i].shape[0])],dtype=float)
            for i in range(len(x_blocks))
        ]
        rep_pool = np.vstack(rep_blocks)

        lambda_self_var = cp.Variable(nonneg=True)
        lambda_social_var = cp.Variable(nonneg=True)
        lambda_init_var = cp.Variable(nonneg=True)
        lambda_bias_var = cp.Variable(nonneg=True)
        lambda_repulsion_var = cp.Variable(nonneg=True)

        bias_tilde_var = cp.Variable()  # variable representing lambda_bias * bias, in reformulation
        
        pred_pool = (lambda_self_var * x_pool + 
                     lambda_social_var * xa_pool + 
                     lambda_init_var * x0_pool + 
                     bias_tilde_var + 
                     lambda_repulsion_var * rep_pool)
        
        objective = cp.Minimize(cp.sum_squares(pred_pool - y_pool))
        constraints = [lambda_self_var + lambda_social_var + lambda_init_var + lambda_bias_var + lambda_repulsion_var == 1,
                       bias_tilde_var <= lambda_bias_var,
                       bias_tilde_var >= -lambda_bias_var]
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, eps_abs=opt_eps, eps_rel=opt_eps, verbose=False)

        if (
            lambda_self_var.value is None or
            lambda_social_var.value is None or
            lambda_init_var.value is None or
            lambda_bias_var.value is None or
            lambda_repulsion_var.value is None or
            bias_tilde_var.value is None
        ):
            raise RuntimeError("Optimization failed to find a solution.")
        
        # clamp all the lambdas to [0, 1] and renormalize
        lambda_self = max(0, min(1, lambda_self_var.value))
        lambda_social = max(0, min(1, lambda_social_var.value))
        lambda_init = max(0, min(1, lambda_init_var.value))
        lambda_bias = max(0, min(1, lambda_bias_var.value))
        lambda_repulsion = max(0, min(1, lambda_repulsion_var.value))

        total = lambda_self + lambda_social + lambda_init + lambda_bias + lambda_repulsion
        if total > 0:
            lambda_self /= total
            lambda_social /= total
            lambda_init /= total
            lambda_bias /= total
            lambda_repulsion /= total
        else:
            raise ValueError("All lambda values are zero after clamping, cannot normalize.")

        # also clip bias_tilde to be in -lambda_bias, lambda_bias
        bias_tilde = max(-lambda_bias, min(lambda_bias, bias_tilde_var.value))

        bias = bias_tilde / lambda_bias if lambda_bias > 0 else 0.0 # if lambda_bias is 0, bias has no effect, so we can set it to 0

        fitted_pool = lambda_self * x_pool + lambda_social * xa_pool + lambda_init * x0_pool + bias * lambda_bias + lambda_repulsion * rep_pool

        mse_pool = float(np.mean((fitted_pool - y_pool) ** 2))

        mse_pool_sum = float(np.sum((fitted_pool - y_pool) ** 2))
        # check that mse_pool_sum is close to the opt value
        if abs(mse_pool_sum - problem.value) > 1e-6:
            raise ValueError(f"Computed mse_pool_sum {mse_pool_sum} is not close to the optimization value {problem.value}")

        solver_iters = -1
        if problem.solver_stats is not None and problem.solver_stats.num_iters is not None:
            solver_iters = int(problem.solver_stats.num_iters)

        candidate = {
            "theta_rep": theta_rep_fixed,
            "lambda_self": lambda_self,
            "lambda_social": lambda_social,
            "lambda_init": lambda_init,
            "lambda_bias": lambda_bias,
            "lambda_repulsion": lambda_repulsion,
            "bias": bias,
            "mse_pool": mse_pool,
            "status":  str(problem.status),
            "success": bool(problem.status in (cp.OPTIMAL)),
            "nit": solver_iters,
            "objective": float(problem.value) if problem.value is not None else mse_pool
        }

        return candidate
    
    def theta_objective(theta_fixed: float) -> float:
        candidate = _solve_for_rep(theta_fixed)
        return candidate["mse_pool"]
    
    theta_hat, theta_objective_map = _grid_search_with_refinement(
        theta_objective, 
        bounds = (0.0, 2.0))
    
    best_result = _solve_for_rep(theta_hat)
    if best_result is None:
        raise ValueError("Failed to find a valid solution for the best theta.")

    total_points = np.shape(y_pool)[0] * np.shape(y_pool)[1]

    return {
        "theta_rep": float(best_result["theta_rep"]),
        "lambda_self": float(best_result["lambda_self"]),
        "lambda_social": float(best_result["lambda_social"]),
        "lambda_init": float(best_result["lambda_init"]),
        "lambda_bias": float(best_result["lambda_bias"]),
        "lambda_repulsion": float(best_result["lambda_repulsion"]),
        "bias": float(best_result["bias"]),
        "mse_pool": float(best_result["mse_pool"]),
        "status": str(best_result["status"]),
        "success": bool(best_result["success"]),
        "nit": int(best_result["nit"]),
        "objective": float(best_result["objective"]),
        "theta_objective_map": theta_objective_map,
        "total_points": int(total_points)
    }

    raise NotImplementedError("This function is not yet implemented. It will solve for the optimal model with a repulsion term using cvxpy.")
