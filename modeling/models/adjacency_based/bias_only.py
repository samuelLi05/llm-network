import cvxpy as cp
import numpy as np

from modeling.models.data_prep import build_dataset_from_run

def fit_bias_only_model(run_traj_map, opt_eps = 1e-09):

    run_names = sorted(run_traj_map.keys())

    x_blocks = []
    y_blocks = []

    for rn in run_names:
        traj = np.asarray(run_traj_map[rn], dtype=float)
        x,y = build_dataset_from_run(traj)
        x_blocks.append(x)
        y_blocks.append(y)

    x_pool = np.vstack(x_blocks)
    y_pool = np.vstack(y_blocks)

    n = x_pool.shape[1]

    bias_tilde = cp.Variable()
    lambda_bias = cp.Variable(nonneg=True)
    lambda_self = cp.Variable(nonneg=True)

    pred_pool = lambda_self * x_pool + bias_tilde
    objective = cp.Minimize(cp.sum_squares(y_pool - pred_pool))
    constraints = [lambda_bias + lambda_self == 1.0,
                   bias_tilde <= lambda_bias,
                   bias_tilde >= -lambda_bias]
    
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, eps_abs=opt_eps, eps_rel=opt_eps, verbose=False)

    if bias_tilde.value is None or lambda_bias.value is None or lambda_self.value is None:
        raise RuntimeError("Bias-only model optimization failed.")
    
    lambda_bias_hat = float(lambda_bias.value)
    lambda_self_hat = float(lambda_self.value)
    bias_tilde_hat = float(bias_tilde.value)
    
    lambda_bias_hat = max(0.0, lambda_bias_hat)
    lambda_self_hat = max(0.0, lambda_self_hat)
    total = lambda_bias_hat + lambda_self_hat
    lambda_bias_hat /= total
    lambda_self_hat /= total

    bias_tilde_hat = float(np.clip(bias_tilde_hat, -lambda_bias_hat, lambda_bias_hat))
    bias_hat = bias_tilde_hat/lambda_bias_hat if lambda_bias_hat > 0 else 0.0


    fitted_pool = bias_tilde_hat + lambda_self_hat * x_pool
    mse_pool = float(np.mean((y_pool - fitted_pool)**2))
    mse_pool_sum = float(np.sum((y_pool - fitted_pool)**2))

    if not (abs(problem.value - mse_pool_sum) < 1e-06):
        raise RuntimeError("Error after cleaning does not match OPT value. ")

    total_points = np.shape(y_pool)[0] * np.shape(y_pool)[1]

    return {
        "lambda_self": float(lambda_self_hat),
        "lambda_bias": float(lambda_bias_hat),
        "bias": float(bias_hat),
        "mse_pool": float(mse_pool),
        "total_points": int(total_points)
    }

def bias_only_rollout(
    lambda_self: float, 
    lambda_bias: float,
    bias: float,
    x0: np.ndarray,
    horizon: int
) -> np.ndarray:
    
    if lambda_self < 0 or lambda_bias < 0:
        raise ValueError("Lambda params must be non-negative")
    if not np.isclose(lambda_bias + lambda_self, 1.0):
        raise ValueError("Lambda params must sum to one")
    if bias < -1 or bias > 1:
        raise ValueError("Bias must be in the range [-1, 1].")


    # check that x0 is a 1D array
    if x0.ndim != 1:
        raise ValueError("x0 must be a 1D array.")
    n = x0.shape[0]

    current = x0.copy()
    predictions = [current.copy()]

    for _ in range(int(horizon)):
        current = lambda_self * current + lambda_bias * bias
        predictions.append(current.copy())

    traj = np.asarray(predictions, dtype=float)
    if not (traj.shape[0] == horizon + 1 and traj.shape[1] == n):
        raise ValueError(f"Trajectory shape {traj.shape} is not as expected ({horizon + 1}, {n})")
    
    return traj

def fit_bias_init_only_model(run_traj_map, opt_eps = 1e-09):

    run_names = sorted(run_traj_map.keys())

    x_blocks = []
    y_blocks = []
    x0_blocks = []

    for rn in run_names:
        traj = np.asarray(run_traj_map[rn], dtype=float)
        x,y = build_dataset_from_run(traj)

        x_blocks.append(x)
        y_blocks.append(y)
        x0_blocks.append(np.repeat(traj[0].reshape(1, -1), x.shape[0], axis=0))

    x_pool = np.vstack(x_blocks)
    y_pool = np.vstack(y_blocks)
    x0_pool = np.vstack(x0_blocks)

    n = x_pool.shape[1]

    bias_tilde = cp.Variable()
    lambda_bias = cp.Variable(nonneg=True)
    lambda_self = cp.Variable(nonneg=True)
    lambda_init = cp.Variable(nonneg=True)

    pred_pool = lambda_self * x_pool + bias_tilde + lambda_init * x0_pool
    objective = cp.Minimize(cp.sum_squares(y_pool - pred_pool))
    constraints = [lambda_bias + lambda_self + lambda_init == 1.0,
                   bias_tilde <= lambda_bias,
                   bias_tilde >= -lambda_bias]
    
    problem = cp.Problem(objective, constraints)
    problem.solve(solver = cp.OSQP, eps_abs=opt_eps, eps_rel=opt_eps, verbose=False)

    if bias_tilde.value is None or lambda_bias.value is None or lambda_self.value is None or lambda_init.value is None:
        raise RuntimeError("Bias-init-only model optimization failed.")
    
    lambda_bias_hat = float(lambda_bias.value)
    lambda_self_hat = float(lambda_self.value)
    lambda_init_hat = float(lambda_init.value)
    bias_tilde_hat = float(bias_tilde.value)

    lambda_bias_hat = max(0.0, lambda_bias_hat)
    lambda_self_hat = max(0.0, lambda_self_hat)
    lambda_init_hat = max(0.0, lambda_init_hat)

    total = lambda_bias_hat + lambda_self_hat + lambda_init_hat
    lambda_bias_hat /= total
    lambda_self_hat /= total
    lambda_init_hat /= total

    bias_tilde_hat = float(np.clip(bias_tilde_hat, -lambda_bias_hat, lambda_bias_hat))
    bias_hat = bias_tilde_hat/lambda_bias_hat if lambda_bias_hat > 0 else 0.0

    fitted_pool = bias_tilde_hat + lambda_self_hat * x_pool + lambda_init_hat * x0_pool

    mse_pool = float(np.mean((y_pool - fitted_pool)**2))
    mse_pool_sum = float(np.sum((y_pool - fitted_pool)**2))

    if not (abs(problem.value - mse_pool_sum) < 1e-06):
        raise RuntimeError("Error after cleaning does not match OPT value. ")

    total_points = np.shape(y_pool)[0] * np.shape(y_pool)[1]

    return {
        "lambda_self": float(lambda_self_hat),
        "lambda_bias": float(lambda_bias_hat),
        "lambda_init": float(lambda_init_hat),
        "bias": float(bias_hat),
        "mse_pool": float(mse_pool),
        "total_points": int(total_points)
    }

def bias_init_only_rollout(
    lambda_self: float, 
    lambda_bias: float,
    lambda_init: float,
    bias: float,
    x0: np.ndarray,
    horizon: int       
):
    
    if lambda_self < 0 or lambda_bias < 0 or lambda_init < 0:
        raise ValueError("Lambda params must be non-negative")
    if not np.isclose(lambda_bias + lambda_self + lambda_init, 1.0):
        raise ValueError("Lambda params must sum to one")
    if bias < -1 or bias > 1:
        raise ValueError("Bias must be in the range [-1, 1].")

    # check that x0 is a 1D array
    if x0.ndim != 1:
        raise ValueError("x0 must be a 1D array.")
    n = x0.shape[0]

    current = x0.copy()
    predictions = [current.copy()]

    for _ in range(int(horizon)):
        current = lambda_self * current + lambda_bias * bias + lambda_init * x0
        predictions.append(current.copy())

    traj = np.asarray(predictions, dtype=float)
    if not (traj.shape[0] == horizon + 1 and traj.shape[1] == n):
        raise ValueError(f"Trajectory shape {traj.shape} is not as expected ({horizon + 1}, {n})")
    
    return traj