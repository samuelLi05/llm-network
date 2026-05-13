from pathlib import Path
import sys
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR.parent
if str(MODELS_DIR) not in sys.path:
	sys.path.insert(0, str(MODELS_DIR))

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

def build_x0_from_agent_inits(agent_inits, n):
	x0 = np.full((n,), np.nan, dtype=float)
	for aid, val in agent_inits.items():
		idx = int(aid.split('_', 1)[1]) - 1
		x0[idx] = float(val)
	if np.isnan(x0).any():
		raise ValueError('Agent init map is missing values')
	return x0


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


def build_synthetic_linear_runs(rng, W_true, n_runs, horizon, noise_std):
	n = int(W_true.shape[0])
	run_traj = {}
	for r in range(n_runs):
		x = rng.uniform(-1.0, 1.0, size=n)
		states = [x.copy()]
		for _ in range(horizon):
			x = W_true @ x + noise_std * rng.normal(size=n)
			states.append(x.copy())
		run_traj[f'run_{r:02d}'] = np.asarray(states, dtype=float)
	return run_traj


def build_synthetic_homophily_runs(
	rng,
	n,
	n_runs,
	horizon,
	Abar,
	gamma,
	noise_std,
	lambda_self=0.0,
	lambda1=0.0,
	lambda2=0.0,
	bias=0.0,
):
	if lambda_self < 0 or lambda1 < 0 or lambda2 < 0 or (lambda_self + lambda1 + lambda2) > 1:
		raise ValueError('invalid lambda values')

	alpha = 1.0 - float(lambda_self) - float(lambda1) - float(lambda2)
	Abar = np.asarray(Abar, dtype=float)
	run_traj = {}
	for r in range(n_runs):
		x_init = rng.uniform(-1.0, 1.0, size=n)
		x = x_init.copy()
		states = [x.copy()]
		for _ in range(horizon):
			diff = np.abs(x[:, None] - x[None, :])
			raw = Abar * np.exp(-float(gamma) * diff)
			row_sums = raw.sum(axis=1, keepdims=True)
			W_t = np.zeros_like(raw)
			valid = row_sums[:, 0] > 0.0
			W_t[valid] = raw[valid] / row_sums[valid]
			h = W_t @ x
			x = (
				float(lambda_self) * x
				+ float(lambda1) * x_init
				+ float(lambda2) * float(bias)
				+ alpha * h
				+ noise_std * rng.normal(size=n)
			)
			states.append(x.copy())
		run_traj[f'run_{r:02d}'] = np.asarray(states, dtype=float)
	return run_traj


def build_neighbors_all_to_all(n):
	return {i: list(range(n)) for i in range(n)}