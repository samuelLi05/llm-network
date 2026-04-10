from pathlib import Path
import sys
import numpy as np
import cvxpy as cp


THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR.parent
if str(MODELS_DIR) not in sys.path:
	sys.path.insert(0, str(MODELS_DIR))

from plot_utils import compute_eigenvalue


def random_row_stochastic_matrix(n, rng):
	W = np.zeros((n, n), dtype=float)
	for i in range(n):
		W[i] = rng.dirichlet(np.ones(n))
	return W


def build_transitions(W, steps, shock_std, rng):
	n = W.shape[0]
	x = rng.uniform(-1.0, 1.0, size=n)
	X, Y = [], []
	for _ in range(steps):
		X.append(x.copy())
		x = W @ x + shock_std * rng.normal(size=n)
		Y.append(x.copy())
	return np.asarray(X, dtype=float), np.asarray(Y, dtype=float)


def build_stacked_independent_transitions(W, n_trajectories, steps_per_trajectory, shock_std, rng):
	X_blocks, Y_blocks = [], []
	for _ in range(n_trajectories):
		X_t, Y_t = build_transitions(W, steps=steps_per_trajectory, shock_std=shock_std, rng=rng)
		X_blocks.append(X_t)
		Y_blocks.append(Y_t)
	return np.vstack(X_blocks), np.vstack(Y_blocks)


def min_positive_eigvals(values, tol=1e-12):
	vals = np.asarray(values, dtype=float)
	pos = vals[vals > tol]
	if pos.size == 0:
		return 0.0
	return float(np.min(pos))


def build_x0_from_agent_inits(agent_inits, n):
	x0 = np.full((n,), np.nan, dtype=float)
	for aid, val in agent_inits.items():
		idx = int(aid.split('_', 1)[1]) - 1
		x0[idx] = float(val)
	if np.isnan(x0).any():
		raise ValueError('Agent init map is missing values')
	return x0


def fit_friedkin_johnsen(run_traj_map, run_neighbors, lambda1, lambda2, agent_inits):
	if lambda1 < 0 or lambda2 < 0 or lambda1 + lambda2 > 1:
		raise ValueError('lambda1 and lambda2 must satisfy nonnegativity and lambda1 + lambda2 <= 1')

	run_names = sorted(run_traj_map.keys())
	ref_neighbors = run_neighbors[run_names[0]]
	for rn in run_names[1:]:
		if run_neighbors[rn] != ref_neighbors:
			raise ValueError('RUN_NEIGHBORS must be identical across runs')

	X_blocks, Y_blocks = [], []
	for rn in run_names:
		X, Y = build_dataset_from_run(np.asarray(run_traj_map[rn], dtype=float))
		X_blocks.append(X)
		Y_blocks.append(Y)

	X_pool = np.vstack(X_blocks)
	Y_pool = np.vstack(Y_blocks)
	n = X_pool.shape[1]
	alpha = 1.0 - lambda1 - lambda2

	b = cp.Variable()
	W_rows = []
	objective_terms = []
	constraints = [b >= -1.0, b <= 1.0]

	x0_init = build_x0_from_agent_inits(agent_inits, n)
	for i in range(n):
		ns = ref_neighbors[i]
		w_ns = cp.Variable(len(ns))
		W_rows.append((i, ns, w_ns))

		X_ns = X_pool[:, ns]
		y = Y_pool[:, i]
		x0i = float(x0_init[i])
		pred = lambda1 * x0i + lambda2 * b + alpha * (X_ns @ w_ns)
		objective_terms.append(cp.sum_squares(y - pred))
		constraints += [w_ns >= 0, cp.sum(w_ns) == 1]

	objective = cp.Minimize(cp.sum(objective_terms))
	prob = cp.Problem(objective, constraints)
	prob.solve(solver=cp.OSQP)

	if b.value is None:
		raise RuntimeError('FJ fit failed')

	W = np.zeros((n, n), dtype=float)
	for i, ns, w_ns in W_rows:
		W[i, ns] = np.asarray(w_ns.value, dtype=float).ravel()

	return W, float(b.value), X_pool, Y_pool


def select_friedkin_johnsen_lambdas_by_mse(run_traj_map, run_neighbors, lambda_grid, agent_inits):
	best_result = None
	all_results = []

	for lambda1 in lambda_grid:
		for lambda2 in lambda_grid:
			if lambda1 + lambda2 > 1:
				continue

			W_hat, b_hat, X_pool, Y_pool = fit_friedkin_johnsen(
				run_traj_map,
				run_neighbors,
				float(lambda1),
				float(lambda2),
				agent_inits,
			)

			n = X_pool.shape[1]
			x0 = build_x0_from_agent_inits(agent_inits, n)
			alpha = 1.0 - float(lambda1) - float(lambda2)
			pred_pool = float(lambda1) * x0[None, :] + float(lambda2) * b_hat + alpha * (X_pool @ W_hat.T)
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


def fit_friedkin_johnsen_joint(run_traj_map, run_neighbors, agent_inits, eps=1e-4):
	run_names = sorted(run_traj_map.keys())
	ref_neighbors = run_neighbors[run_names[0]]
	for rn in run_names[1:]:
		if run_neighbors[rn] != ref_neighbors:
			raise ValueError('RUN_NEIGHBORS must be identical across runs')

	X_blocks, Y_blocks = [], []
	for rn in run_names:
		X, Y = build_dataset_from_run(np.asarray(run_traj_map[rn], dtype=float))
		X_blocks.append(X)
		Y_blocks.append(Y)

	X_pool = np.vstack(X_blocks)
	Y_pool = np.vstack(Y_blocks)
	_, n = X_pool.shape

	x0 = build_x0_from_agent_inits(agent_inits, n)
	X0_pool = np.repeat(x0.reshape(1, -1), X_pool.shape[0], axis=0)

	lambda1 = cp.Variable(nonneg=True)
	lambda2 = cp.Variable(nonneg=True)
	b_tilde = cp.Variable()
	alpha = 1.0 - lambda1 - lambda2
	W_tilde = cp.Variable((n, n))

	ones_n = np.ones((n,), dtype=float)
	residual = Y_pool - (lambda1 * X0_pool + b_tilde * ones_n[None, :] + X_pool @ W_tilde.T)
	objective = cp.Minimize(cp.sum_squares(residual))
	constraints = [
		lambda2 >= eps,
		lambda1 >= 0.0,
		lambda1 + lambda2 <= 1.0 - eps,
		lambda1 <= 1.0,
	]

	for i in range(n):
		ns = ref_neighbors[i]
		allowed = np.zeros((n,), dtype=float)
		allowed[np.asarray(ns, dtype=int)] = 1.0
		constraints.append(W_tilde[i, :] >= 0)
		constraints.append(cp.sum(W_tilde[i, :]) == alpha)
		constraints.append(cp.multiply(1.0 - allowed, W_tilde[i, :]) == 0)

	constraints += [b_tilde <= lambda2, b_tilde >= -lambda2]

	prob = cp.Problem(objective, constraints)
	prob.solve(solver=cp.OSQP)

	if lambda1.value is None or lambda2.value is None or W_tilde.value is None or b_tilde.value is None:
		raise RuntimeError('Joint FJ optimization failed')

	lambda1_hat = float(lambda1.value)
	lambda2_hat = float(lambda2.value)
	alpha_hat = 1.0 - lambda1_hat - lambda2_hat
	if alpha_hat <= eps:
		raise RuntimeError('Estimated alpha too small for stable W recovery')

	b_tilde_hat = float(b_tilde.value)
	W_tilde_hat = np.asarray(W_tilde.value, dtype=float)
	W_hat = W_tilde_hat / alpha_hat
	b_hat = b_tilde_hat / lambda2_hat

	fitted_pool = lambda1_hat * X0_pool + b_tilde_hat * ones_n[None, :] + X_pool @ W_tilde_hat.T
	mse_pool = float(np.mean((Y_pool - fitted_pool) ** 2))

	return {
		'lambda1': lambda1_hat,
		'lambda2': lambda2_hat,
		'alpha': alpha_hat,
		'b': float(b_hat),
		'W': W_hat,
		'mse_pool': mse_pool,
		'X_pool': X_pool,
		'Y_pool': Y_pool,
	}


def build_dataset_from_run(run):
	X, Y = [], []
	for t in range(len(run) - 1):
		X.append(run[t])
		Y.append(run[t + 1])
	return np.asarray(X, dtype=float), np.asarray(Y, dtype=float)


def build_synthetic_fj_runs(rng, n, n_runs, horizon, neighbors, lambda1, lambda2, b, x0_prior, noise_std):
	W_true = np.zeros((n, n), dtype=float)
	for i in range(n):
		ns = neighbors[i]
		W_true[i, ns] = rng.dirichlet(np.ones(len(ns)))

	alpha = 1.0 - lambda1 - lambda2
	bias_vec = np.full((n,), float(b), dtype=float)

	run_traj = {}
	for r in range(n_runs):
		x = rng.uniform(-1.0, 1.0, size=n)
		states = [x.copy()]
		for _ in range(horizon):
			x = lambda1 * x0_prior + lambda2 * bias_vec + alpha * (W_true @ x) + noise_std * rng.normal(size=n)
			states.append(x.copy())
		run_traj[f'run_{r:02d}'] = np.asarray(states, dtype=float)

	run_neighbors = {rn: neighbors for rn in run_traj}
	return W_true, run_traj, run_neighbors


def test_compute_eigenvalue_full_equals_reduced_for_all_neighbors():
	rng = np.random.default_rng(7)
	n = 8
	W = random_row_stochastic_matrix(n, rng)
	X, Y = build_transitions(W, steps=80, shock_std=0.05, rng=rng)
	neighbors = {i: list(range(n)) for i in range(n)}

	out = compute_eigenvalue(X, Y, neighbors, intercepts=False)

	assert out['gram_full_shape'] == (n * n, n * n)
	assert out['eigvals_full'].shape == (n * n,)


def test_compute_eigenvalue_min_reduced_eig_increases_with_richer_transitions():
	rng = np.random.default_rng(11)
	n = 10
	W = random_row_stochastic_matrix(n, rng)
	neighbors = {i: list(range(n)) for i in range(n)}
	shock_std = 0.08
	steps_per_trajectory = 80

	X_poor, Y_poor = build_stacked_independent_transitions(
		W,
		n_trajectories=2,
		steps_per_trajectory=steps_per_trajectory,
		shock_std=shock_std,
		rng=rng,
	)
	X_rich, Y_rich = build_stacked_independent_transitions(
		W,
		n_trajectories=8,
		steps_per_trajectory=steps_per_trajectory,
		shock_std=shock_std,
		rng=rng,
	)

	poor = compute_eigenvalue(X_poor, Y_poor, neighbors, intercepts=False)
	rich = compute_eigenvalue(X_rich, Y_rich, neighbors, intercepts=False)

	poor_min = min_positive_eigvals(poor['eigvals_full'])
	rich_min = min_positive_eigvals(rich['eigvals_full'])

	assert rich_min > poor_min


def test_compute_eigenvalue_full_is_neighbor_invariant():
	rng = np.random.default_rng(19)
	n = 10
	neighbors = {
		i: sorted({i, (i - 1) % n, (i + 1) % n, (i + 3) % n})
		for i in range(n)
	}
	all_neighbors = {i: list(range(n)) for i in range(n)}

	W = np.zeros((n, n), dtype=float)
	for i in range(n):
		ns = neighbors[i]
		W[i, ns] = rng.dirichlet(np.ones(len(ns)))

	X, Y = build_transitions(W, steps=180, shock_std=0.08, rng=rng)
	out_sparse = compute_eigenvalue(X, Y, neighbors, intercepts=False)
	out_all = compute_eigenvalue(X, Y, all_neighbors, intercepts=False)

	assert out_sparse['gram_full_shape'] == (n * n, n * n)
	assert out_all['gram_full_shape'] == (n * n, n * n)
	np.testing.assert_allclose(out_sparse['eigvals_full'], out_all['eigvals_full'], rtol=1e-9, atol=1e-11)


def test_fj_line_search_and_joint_fit_are_similar_on_synthetic_data():
	rng = np.random.default_rng(101)
	n = 8
	n_runs = 12
	horizon = 16
	neighbors = {i: list(range(n)) for i in range(n)}

	true_lambda1 = 0.2
	true_lambda2 = 0.3
	true_b = -0.15
	noise_std = 0.00

	x0_prior = rng.uniform(-0.8, 0.8, size=n)
	agent_inits = {f'agent_{i + 1}': float(x0_prior[i]) for i in range(n)}

	W_true, run_traj, run_neighbors = build_synthetic_fj_runs(
		rng=rng,
		n=n,
		n_runs=n_runs,
		horizon=horizon,
		neighbors=neighbors,
		lambda1=true_lambda1,
		lambda2=true_lambda2,
		b=true_b,
		x0_prior=x0_prior,
		noise_std=noise_std,
	)

	lambda_grid = np.linspace(0.05, 0.5, 10)
	best, _ = select_friedkin_johnsen_lambdas_by_mse(
		run_traj,
		run_neighbors,
		lambda_grid=lambda_grid,
		agent_inits=agent_inits,
	)

	W_ls, b_ls, X_pool_ls, Y_pool_ls = fit_friedkin_johnsen(
		run_traj,
		run_neighbors,
		best['lambda1'],
		best['lambda2'],
		agent_inits,
	)
	joint = fit_friedkin_johnsen_joint(run_traj, run_neighbors, agent_inits, eps=1e-4)

	assert abs(best['lambda1'] - joint['lambda1']) < 0.08
	assert abs(best['lambda2'] - joint['lambda2']) < 0.08
	assert abs(b_ls - joint['b']) < 0.15

	W_rel = float(np.linalg.norm(W_ls - joint['W']) / max(np.linalg.norm(W_ls), 1e-12))
	assert W_rel < 0.15

	x0 = build_x0_from_agent_inits(agent_inits, n)
	alpha_ls = 1.0 - best['lambda1'] - best['lambda2']
	pred_ls = best['lambda1'] * x0[None, :] + best['lambda2'] * b_ls + alpha_ls * (X_pool_ls @ W_ls.T)
	alpha_joint = 1.0 - joint['lambda1'] - joint['lambda2']
	pred_joint = joint['lambda1'] * x0[None, :] + joint['lambda2'] * joint['b'] + alpha_joint * (X_pool_ls @ joint['W'].T)
	pred_mse = float(np.mean((pred_ls - pred_joint) ** 2))

	# True-parameter recovery metrics for each method.
	W_rel_ls_true = float(np.linalg.norm(W_ls - W_true) / max(np.linalg.norm(W_true), 1e-12))
	W_rel_joint_true = float(np.linalg.norm(joint['W'] - W_true) / max(np.linalg.norm(W_true), 1e-12))
	b_abs_ls_true = float(abs(b_ls - true_b))
	b_abs_joint_true = float(abs(joint['b'] - true_b))
	l1_abs_ls_true = float(abs(best['lambda1'] - true_lambda1))
	l2_abs_ls_true = float(abs(best['lambda2'] - true_lambda2))
	l1_abs_joint_true = float(abs(joint['lambda1'] - true_lambda1))
	l2_abs_joint_true = float(abs(joint['lambda2'] - true_lambda2))

	assert pred_mse < 5e-4
	assert abs(best['mse_pool'] - joint['mse_pool']) < 5e-4

	assert W_rel_ls_true < 0.20
	assert W_rel_joint_true < 0.20
	assert b_abs_ls_true < 0.20
	assert b_abs_joint_true < 0.20
	assert l1_abs_ls_true < 0.12
	assert l2_abs_ls_true < 0.12
	assert l1_abs_joint_true < 0.12
	assert l2_abs_joint_true < 0.12
