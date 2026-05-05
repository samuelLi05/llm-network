from data_prep import load_run_data, build_global_init_map, build_run_trajectory, build_neighbors_index, _numeric_agent_key

import cvxpy as cp
import numpy as np

from plot_utils import (
    compute_mean_per_timestep,
    compute_variance_per_timestep,
    compute_wasserstein_distance_per_timestep,
)

def build_dataset_from_run(run):
    X = []
    Y = []
    for t in range(len(run) - 1):
        X.append(run[t])
        Y.append(run[t + 1])
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    return X, Y

def fit_row_stochastic_W_from_pooled_runs(run_traj_map, run_neighbors):
    run_names = sorted(run_traj_map.keys())
    ref_neighbors = run_neighbors[run_names[0]]

    for run_name in run_names[1:]:
        if run_neighbors[run_name] != ref_neighbors:
            raise ValueError('RUN_NEIGHBORS must be identical across runs for pooled fitting.')

    X_blocks = []
    Y_blocks = []
    for run_name in run_names:
        X, Y = build_dataset_from_run(np.asarray(run_traj_map[run_name], dtype=float))
        X_blocks.append(X)
        Y_blocks.append(Y)

    X_pool = np.vstack(X_blocks)
    Y_pool = np.vstack(Y_blocks)

    n = X_pool.shape[1]
    W = np.zeros((n, n), dtype=float)

    for i in range(n):
        ns = ref_neighbors[i]
        X_ns = X_pool[:, ns]
        y = Y_pool[:, i]

        w_ns = cp.Variable(len(ns))
        objective = cp.Minimize(cp.sum_squares(X_ns @ w_ns - y))
        constraints = [w_ns >= 0, cp.sum(w_ns) == 1]
        problem = cp.Problem(objective, constraints)
        problem.solve()

        row = np.zeros(n, dtype=float)
        row[ns] = np.asarray(w_ns.value, dtype=float).ravel()
        W[i] = row

    return W, X_pool, Y_pool


def degroot_rollout_prediction(W, x0, horizon):
    predictions = [x0]
    current_x = x0.copy()
    for t in range(horizon):
        current_x = W @ current_x
        predictions.append(current_x.copy())
    return predictions


def fit_friedkin_johnsen(run_traj_map, run_neighbors, lambda1, lambda2, agent_inits):
    if lambda1 < 0 or lambda2 < 0 or lambda1 + lambda2 > 1:
        raise ValueError('lambda1 and lambda2 must be nonnegative and satisfy lambda1 + lambda2 <= 1')

    run_names = sorted(run_traj_map.keys())
    ref_neighbors = run_neighbors[run_names[0]]

    # Ensure same graph across runs
    for rn in run_names[1:]:
        if run_neighbors[rn] != ref_neighbors:
            raise ValueError('RUN_NEIGHBORS must be identical across runs')
        

    # Build pooled dataset
    X_blocks, Y_blocks, X0_blocks = [], [], []
    for rn in run_names:
        traj = np.asarray(run_traj_map[rn], dtype=float)
        X, Y = build_dataset_from_run(traj)
        X_blocks.append(X)
        Y_blocks.append(Y)

    X_pool = np.vstack(X_blocks)
    Y_pool = np.vstack(Y_blocks)
    n = X_pool.shape[1]
    alpha = 1.0 - lambda1 - lambda2

    b = cp.Variable()  # scalar global bias
    W_vars = []        # store per-row variables
    objective_terms = []
    constraints = []

    x0_init = build_x0_from_agent_inits(agent_inits, n)

    for i in range(n):
        ns = ref_neighbors[i]
        if len(ns) == 0:
            continue  # skip isolated nodes safely

        w_ns = cp.Variable(len(ns))
        W_vars.append((i, ns, w_ns))

        X_ns = X_pool[:, ns]
        y = Y_pool[:, i]
        x0i = float(x0_init[i])

        pred = lambda1 * x0i + lambda2 * b + alpha * (X_ns @ w_ns)
        objective_terms.append(cp.sum_squares(y - pred))

        constraints += [
            w_ns >= 0,
            cp.sum(w_ns) == 1,
        ]

    # bias constraint
    constraints += [b >= -1, b <= 1]

    objective = cp.Minimize(cp.sum(objective_terms))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    if b.value is None:
        raise RuntimeError('Solver failed')

    # ---- RECONSTRUCT W ----
    W = np.zeros((n, n), dtype=float)
    for (i, ns, w_ns) in W_vars:
        W[i, ns] = np.asarray(w_ns.value).ravel()

    return W, float(b.value), X_pool, Y_pool

def build_x0_from_agent_inits(agent_inits, n):
    x0 = np.full((n,), np.nan, dtype=float)
    for aid, val in agent_inits.items():
        idx = int(aid.split('_', 1)[1]) - 1
        x0[idx] = float(val)
    if np.isnan(x0).any():
        missing = np.where(np.isnan(x0))[0].tolist()
        raise ValueError(f'missing init values for indices: {missing}')
    return x0

def friedkin_johnsen_rollout_prediction(W, bias, x0, horizon, lambda1, lambda2):
    alpha = 1.0 - lambda1 - lambda2
    x0 = np.asarray(x0, dtype=float)
    current_x = x0.copy()
    predictions = [current_x.copy()]
    bias_vec = np.full_like(current_x, float(bias), dtype=float)

    for _ in range(horizon):
        current_x = lambda1 * x0 + lambda2 * bias_vec + alpha * (W @ current_x)
        predictions.append(current_x.copy())

    return predictions


def select_friedkin_johnsen_lambdas(run_traj_map, run_neighbors, lambda_grid, agent_inits):
    best_result = None
    all_results = []

    for lambda1 in lambda_grid:
        for lambda2 in lambda_grid:
            if lambda1 + lambda2 > 1:
                continue

            W_hat, b_hat, X_pool, Y_pool = fit_friedkin_johnsen(
                run_traj_map,
                run_neighbors,
                lambda1,
                lambda2,
                agent_inits,
            )

            n = X_pool.shape[1]
            x0 = build_x0_from_agent_inits(agent_inits, n)
            alpha = 1.0 - lambda1 - lambda2
            pred_pool = lambda1 * x0[None, :] + lambda2 * b_hat + alpha * (X_pool @ W_hat.T)
            mse_pool = float(np.mean((Y_pool - pred_pool) ** 2))

            result = {
                'lambda1': float(lambda1),
                'lambda2': float(lambda2),
                'mse_pool': mse_pool,
            }
            all_results.append(result)

            if best_result is None or mse_pool < best_result['mse_pool']:
                best_result = result

    return best_result, all_results


# Joint FJ fit with reparameterization: W_tilde = alpha * W, b_tilde = lambda2 * b

def fit_friedkin_johnsen_joint(run_traj_map, run_neighbors, agent_inits, eps=1e-4):
    run_names = sorted(run_traj_map.keys())
    ref_neighbors = run_neighbors[run_names[0]]

    for rn in run_names[1:]:
        if run_neighbors[rn] != ref_neighbors:
            raise ValueError('RUN_NEIGHBORS must be identical across runs for pooled fitting.')

    # Build pooled transitions with per-transition run-level initial condition x0.
    X_blocks = []
    Y_blocks = []

    for rn in run_names:
        traj = np.asarray(run_traj_map[rn], dtype=float)
        X, Y = build_dataset_from_run(traj)
        X_blocks.append(X)
        Y_blocks.append(Y)

    X_pool = np.vstack(X_blocks)
    Y_pool = np.vstack(Y_blocks)
    _, n = X_pool.shape
    x0_init = build_x0_from_agent_inits(agent_inits, n)
    X0_pool = np.repeat(x0_init.reshape(1, -1), X_pool.shape[0], axis=0)
    # Decision variables.
    lambda1 = cp.Variable(nonneg=True)
    lambda2 = cp.Variable(nonneg=True)
    b_tilde = cp.Variable()  # equals lambda2 * b
    alpha = 1.0 - lambda1 - lambda2
    W_tilde = cp.Variable((n, n))

    # Least-squares objective on pooled transitions.
    ones_n = np.ones((n,), dtype=float)
    residual = Y_pool - (lambda1 * X0_pool + b_tilde * ones_n[None, :] + X_pool @ W_tilde.T)
    objective = cp.Minimize(cp.sum_squares(residual))
    constraints = []

    # Strict-like constraints via epsilon to support identifiability/bijection:
    # lambda1 + lambda2 < 1  ->  lambda1 + lambda2 <= 1 - eps
    # lambda2 != 0           ->  lambda2 >= eps
    constraints += [
        lambda2 >= eps,
        lambda1 >= 0.2,
        lambda1 + lambda2 <= 1.0 - eps,
        lambda1 <= 1.0,
    ]

    # Graph-structured nonnegative row constraints on W_tilde.
    for i in range(n):
        ns = ref_neighbors[i]
        allowed = np.zeros((n,), dtype=float)
        allowed[np.asarray(ns, dtype=int)] = 1.0

        constraints.append(W_tilde[i, :] >= 0)
        constraints.append(cp.sum(W_tilde[i, :]) == alpha)
        constraints.append(cp.multiply(1.0 - allowed, W_tilde[i, :]) == 0)

    # Since b in [-1, 1], b_tilde = lambda2 * b implies b_tilde in [-lambda2, lambda2].
    constraints += [
        b_tilde <= lambda2,
        b_tilde >= -lambda2,

    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    if lambda1.value is None or lambda2.value is None or W_tilde.value is None or b_tilde.value is None:

        raise RuntimeError('Joint FJ optimization failed to produce a solution.')

    lambda1_hat = float(lambda1.value)
    lambda2_hat = float(lambda2.value)
    alpha_hat = 1.0 - lambda1_hat - lambda2_hat
    b_tilde_hat = float(b_tilde.value)
    W_tilde_hat = np.asarray(W_tilde.value, dtype=float)

    if alpha_hat <= eps:

        raise RuntimeError(f'Estimated alpha too small for stable W recovery: alpha={alpha_hat}')

    W_hat = W_tilde_hat / alpha_hat
    b_hat = b_tilde_hat / lambda2_hat

    # Build fitted pooled prediction and MSE in the same parameterization used for training.
    fitted_pool = lambda1_hat * X0_pool + b_tilde_hat * ones_n[None, :] + X_pool @ W_tilde_hat.T
    mse_pool = float(np.mean((Y_pool - fitted_pool) ** 2))

    return {
        'lambda1': lambda1_hat,
        'lambda2': lambda2_hat,
        'alpha': alpha_hat,
        'b_tilde': b_tilde_hat,
        'W_tilde': W_tilde_hat,
        'W': W_hat,
        'b': float(b_hat),
        'X_pool': X_pool,
        'Y_pool': Y_pool,
        'X0_pool': X0_pool,
        'mse_pool': mse_pool,
        'status': prob.status,
        'objective': float(prob.value) if prob.value is not None else np.nan,
    }

def fit_friedkin_johnsen_joint_traj0(run_traj_map,run_neighbors,eps=1e-4):
    run_names=sorted(run_traj_map.keys())
    ref_neighbors=run_neighbors[run_names[0]]

    for rn in run_names[1:]:
        if run_neighbors[rn]!=ref_neighbors:
            raise ValueError('RUN_NEIGHBORS must be identical across runs for pooled fitting.')
        
    X_blocks=[]
    Y_blocks=[]
    X0_blocks=[]
    for rn in run_names:
        traj=np.asarray(run_traj_map[rn],dtype=float)
        X,Y=build_dataset_from_run(traj)
        X_blocks.append(X)
        Y_blocks.append(Y)
        X0_blocks.append(np.repeat(traj[0].reshape(1,-1),X.shape[0],axis=0))

    X_pool=np.vstack(X_blocks)
    Y_pool=np.vstack(Y_blocks)
    X0_pool=np.vstack(X0_blocks)
    _,n=X_pool.shape

    lambda1=cp.Variable(nonneg=True)
    lambda2=cp.Variable(nonneg=True)
    b_tilde=cp.Variable()
    alpha=1.0-lambda1-lambda2
    W_tilde=cp.Variable((n,n))
    ones_n=np.ones((n,),dtype=float)
    residual=Y_pool-(lambda1*X0_pool+b_tilde*ones_n[None,:]+X_pool@W_tilde.T)
    objective=cp.Minimize(cp.sum_squares(residual))
    constraints=[lambda2>=eps, lambda1+lambda2<=1.0-eps,lambda1<=1.0]

    for i in range(n):
        ns=ref_neighbors[i]
        allowed=np.zeros((n,),dtype=float)
        allowed[np.asarray(ns,dtype=int)]=1.0
        constraints.append(W_tilde[i,:]>=0)
        constraints.append(cp.sum(W_tilde[i,:])==alpha)
        constraints.append(cp.multiply(1.0-allowed,W_tilde[i,:])==0)
    constraints+=[b_tilde<=lambda2,b_tilde>=-lambda2]
    prob=cp.Problem(objective,constraints)
    prob.solve(solver=cp.OSQP)

    if lambda1.value is None or lambda2.value is None or W_tilde.value is None or b_tilde.value is None:
        raise RuntimeError('Joint FJ traj0 optimization failed to produce a solution.')
    lambda1_hat=float(lambda1.value)
    lambda2_hat=float(lambda2.value)
    alpha_hat=1.0-lambda1_hat-lambda2_hat
    b_tilde_hat=float(b_tilde.value)
    W_tilde_hat=np.asarray(W_tilde.value,dtype=float)
    if alpha_hat<=eps:
        raise RuntimeError(f'Estimated alpha too small for stable W recovery: alpha={alpha_hat}')
    
    W_hat=W_tilde_hat/alpha_hat
    b_hat=b_tilde_hat/lambda2_hat
    fitted_pool=lambda1_hat*X0_pool+b_tilde_hat*ones_n[None,:]+X_pool@W_tilde_hat.T
    mse_pool=float(np.mean((Y_pool-fitted_pool)**2))
    return {'lambda1':lambda1_hat,'lambda2':lambda2_hat,'alpha':alpha_hat,'b_tilde':b_tilde_hat,'W_tilde':W_tilde_hat,'W':W_hat,'b':float(b_hat),'X_pool':X_pool,'Y_pool':Y_pool,'X0_pool':X0_pool,'mse_pool':mse_pool,'status':prob.status,'objective':float(prob.value) if prob.value is not None else np.nan}


def build_row_normalized_adjacency(neighbors, n):
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        row_neighbors = list(neighbors[i])
        if len(row_neighbors) == 0:
            A[i, i] = 1.0
            continue
        row_neighbors = [j for j in row_neighbors if 0 <= j < n]
        if len(row_neighbors) == 0:
            A[i, i] = 1.0
            continue
        A[i, row_neighbors] = 1.0 / len(row_neighbors)
    return A

def fit_degroot_adjacency_scalar(run_traj_map, run_neighbors):
    run_names = sorted(run_traj_map.keys())
    ref_neighbors = run_neighbors[run_names[0]]

    for rn in run_names[1:]:
        if run_neighbors[rn] != ref_neighbors:
            raise ValueError('RUN_NEIGHBORS must be identical across runs for pooled fitting.')

    X_blocks = []
    Y_blocks = []

    for rn in run_names:
        traj = np.asarray(run_traj_map[rn], dtype=float)
        X, Y = build_dataset_from_run(traj)
        X_blocks.append(X)
        Y_blocks.append(Y)

    X_pool = np.vstack(X_blocks)
    Y_pool = np.vstack(Y_blocks)

    n = X_pool.shape[1]
    Abar = build_row_normalized_adjacency(ref_neighbors, n)

    gamma = cp.Variable()
    pred_pool = gamma * (X_pool @ Abar.T) + (1.0 - gamma) * X_pool

    objective = cp.Minimize(cp.sum_squares(Y_pool - pred_pool))
    constraints = [
        gamma >= 0,
        gamma <= 1,
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)

    if gamma.value is None:
        raise RuntimeError('Adjacency-scalar DeGroot optimization failed to produce a solution.')

    gamma_hat = float(gamma.value)
    W_hat = gamma_hat * Abar + (1.0 - gamma_hat) * np.eye(n, dtype=float)
    fitted_pool = X_pool @ W_hat.T
    mse_pool = float(np.mean((Y_pool - fitted_pool) ** 2))

    return {
        'gamma': gamma_hat,
        'Abar': Abar,
        'W': W_hat,
        'X_pool': X_pool,
        'Y_pool': Y_pool,
        'mse_pool': mse_pool,
        'status': problem.status,
        'objective': float(problem.value) if problem.value is not None else np.nan,
    }

def fit_friedkin_johnsen_adjacency(run_traj_map, run_neighbors, lambda1, lambda2):
    if lambda1 < 0 or lambda2 < 0 or lambda1 + lambda2 > 1:
        raise ValueError('lambda1 and lambda2 must be nonnegative and satisfy lambda1 + lambda2 <= 1')

    run_names = sorted(run_traj_map.keys())
    ref_neighbors = run_neighbors[run_names[0]]

    for rn in run_names[1:]:
        if run_neighbors[rn] != ref_neighbors:
            raise ValueError('RUN_NEIGHBORS must be identical across runs for pooled fitting.')

    X_blocks = []
    Y_blocks = []
    X0_blocks = []

    for rn in run_names:
        traj = np.asarray(run_traj_map[rn], dtype=float)
        X, Y = build_dataset_from_run(traj)
        X_blocks.append(X)
        Y_blocks.append(Y)
        X0_blocks.append(np.repeat(traj[0].reshape(1, -1), X.shape[0], axis=0))

    X_pool = np.vstack(X_blocks)
    Y_pool = np.vstack(Y_blocks)
    X0_pool = np.vstack(X0_blocks)

    n = X_pool.shape[1]
    Abar = build_row_normalized_adjacency(ref_neighbors, n)
    alpha = 1.0 - lambda1 - lambda2

    gamma = cp.Variable()
    bias = cp.Variable()

    bias_vec = bias * np.ones((n,), dtype=float)
    pred_pool = (
        lambda1 * X0_pool
        + lambda2 * bias_vec[None, :]
        + alpha * (gamma * (X_pool @ Abar.T) + (1.0 - gamma) * X_pool)
    )

    objective = cp.Minimize(cp.sum_squares(Y_pool - pred_pool))
    constraints = [
        gamma >= 0,
        gamma <= 1,
        bias >= -1,
        bias <= 1,
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP)

    if gamma.value is None or bias.value is None:
        raise RuntimeError('Adjacency-based FJ optimization failed to produce a solution.')

    gamma_hat = float(gamma.value)
    bias_hat = float(bias.value)
    W_hat = gamma_hat * Abar + (1.0 - gamma_hat) * np.eye(n, dtype=float)
    fitted_pool = (
        lambda1 * X0_pool
        + lambda2 * bias_hat * np.ones((1, n), dtype=float)
        + alpha * (X_pool @ W_hat.T)
    )
    mse_pool = float(np.mean((Y_pool - fitted_pool) ** 2))

    return {
        'gamma': gamma_hat,
        'bias': bias_hat,
        'Abar': Abar,
        'W': W_hat,
        'X_pool': X_pool,
        'Y_pool': Y_pool,
        'X0_pool': X0_pool,
        'mse_pool': mse_pool,
        'status': problem.status,
        'objective': float(problem.value) if problem.value is not None else np.nan,
    }

def friedkin_johnsen_adjacency_rollout(W, bias, x0, horizon, lambda1, lambda2):
    alpha = 1.0 - lambda1 - lambda2
    x0 = np.asarray(x0, dtype=float)
    current_x = x0.copy()
    predictions = [current_x.copy()]
    bias_vec = np.full_like(current_x, float(bias), dtype=float)

    for _ in range(horizon):
        current_x = lambda1 * x0 + lambda2 * bias_vec + alpha * (W @ current_x)
        predictions.append(current_x.copy())

    return predictions


def select_friedkin_johnsen_adjacency_lambdas(run_traj_map, run_neighbors, lambda_grid):
    """
    Grid search over lambda1, lambda2 pairs for adjacency-based FJ model.
    Calls fit_friedkin_johnsen_adjacency() for each valid pair and returns best result.
    """
    best_result = None
    all_results = []

    for lambda1 in lambda_grid:
        for lambda2 in lambda_grid:
            if lambda1 + lambda2 > 1:
                continue

            adj_result = fit_friedkin_johnsen_adjacency(
                run_traj_map,
                run_neighbors,
                lambda1,
                lambda2,
            )

            mse_pool = adj_result['mse_pool']
            
            result = {
                'lambda1': float(lambda1),
                'lambda2': float(lambda2),
                'mse_pool': mse_pool,
                'gamma': adj_result['gamma'],
                'bias': adj_result['bias'],
            }
            all_results.append(result)

            if best_result is None or mse_pool < best_result['mse_pool']:
                best_result = result

    return best_result, all_results


def align_rollout_pair(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    T = min(observed.shape[0], predicted.shape[0])
    return observed[:T], predicted[:T]


def stack_curves(curves):
    curves = [np.asarray(curve, dtype=float).ravel() for curve in curves if len(curve) > 0]
    if not curves:
        return np.empty((0, 0), dtype=float)
    common_T = min(curve.shape[0] for curve in curves)
    return np.stack([curve[:common_T] for curve in curves], axis=0)

def evaluate_validation_model(validation_traj_map, rollout_fn):
    run_names = sorted(validation_traj_map.keys())
    per_run = {}
    mean_true_curves = []
    mean_pred_curves = []
    var_true_curves = []
    var_pred_curves = []
    wasserstein_curves = []
    transition_mses = []

    for rn in run_names:
        observed = np.asarray(validation_traj_map[rn], dtype=float)
        predicted = np.asarray(rollout_fn(observed), dtype=float)
        observed, predicted = align_rollout_pair(observed, predicted)

        mean_true, mean_pred = compute_mean_per_timestep(observed, predicted)
        var_true, var_pred = compute_variance_per_timestep(observed, predicted)
        wasserstein_curve = compute_wasserstein_distance_per_timestep(observed, predicted)

        transition_mses.append(float(np.mean((observed - predicted) ** 2)))
        mean_true_curves.append(mean_true)
        mean_pred_curves.append(mean_pred)
        var_true_curves.append(var_true)
        var_pred_curves.append(var_pred)
        wasserstein_curves.append(wasserstein_curve)

        per_run[rn] = {
            'observed': observed,
            'predicted': predicted,
            'mean_true': mean_true,
            'mean_pred': mean_pred,
            'var_true': var_true,
            'var_pred': var_pred,
            'wasserstein': wasserstein_curve,
            'transition_mse': float(np.mean((observed - predicted) ** 2)),
        }

    mean_true_stack = stack_curves(mean_true_curves)
    mean_pred_stack = stack_curves(mean_pred_curves)
    var_true_stack = stack_curves(var_true_curves)
    var_pred_stack = stack_curves(var_pred_curves)
    wasserstein_stack = stack_curves(wasserstein_curves)

    return {
        'per_run': per_run,
        'mean_true_stack': mean_true_stack,
        'mean_pred_stack': mean_pred_stack,
        'var_true_stack': var_true_stack,
        'var_pred_stack': var_pred_stack,
        'wasserstein_stack': wasserstein_stack,
        'transition_mse_mean': float(np.mean(transition_mses)),
        'mean_curve_abs_error': float(np.mean(np.abs(mean_true_stack - mean_pred_stack))) if mean_true_stack.size else np.nan,
        'var_curve_abs_error': float(np.mean(np.abs(var_true_stack - var_pred_stack))) if var_true_stack.size else np.nan,
        'wasserstein_curve_mean': float(np.mean(wasserstein_stack)) if wasserstein_stack.size else np.nan,
    }
