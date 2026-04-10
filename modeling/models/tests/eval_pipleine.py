from pathlib import Path
import sys
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR.parent
if str(MODELS_DIR) not in sys.path:
	sys.path.insert(0, str(MODELS_DIR))

from plot_utils import compute_eigenvalue


def _random_row_stochastic_matrix(n, rng):
	W = np.zeros((n, n), dtype=float)
	for i in range(n):
		W[i] = rng.dirichlet(np.ones(n))
	return W


def _build_transitions(W, steps, shock_std, rng):
	n = W.shape[0]
	x = rng.uniform(-1.0, 1.0, size=n)
	X, Y = [], []
	for _ in range(steps):
		X.append(x.copy())
		x = W @ x + shock_std * rng.normal(size=n)
		Y.append(x.copy())
	return np.asarray(X, dtype=float), np.asarray(Y, dtype=float)


def _build_stacked_independent_transitions(W, n_trajectories, steps_per_trajectory, shock_std, rng):
	X_blocks, Y_blocks = [], []
	for _ in range(n_trajectories):
		X_t, Y_t = _build_transitions(W, steps=steps_per_trajectory, shock_std=shock_std, rng=rng)
		X_blocks.append(X_t)
		Y_blocks.append(Y_t)
	return np.vstack(X_blocks), np.vstack(Y_blocks)


def _min_positive_eigvals(values, tol=1e-12):
	vals = np.asarray(values, dtype=float)
	pos = vals[vals > tol]
	if pos.size == 0:
		return 0.0
	return float(np.min(pos))


def test_compute_eigenvalue_full_equals_reduced_for_all_neighbors():
	rng = np.random.default_rng(7)
	n = 8
	W = _random_row_stochastic_matrix(n, rng)
	X, Y = _build_transitions(W, steps=80, shock_std=0.05, rng=rng)
	neighbors = {i: list(range(n)) for i in range(n)}

	out = compute_eigenvalue(X, Y, neighbors, intercepts=False)

	assert out['gram_full_shape'] == (n * n, n * n)
	assert out['eigvals_full'].shape == (n * n,)


def test_compute_eigenvalue_min_reduced_eig_increases_with_richer_transitions():
	rng = np.random.default_rng(11)
	n = 10
	W = _random_row_stochastic_matrix(n, rng)
	neighbors = {i: list(range(n)) for i in range(n)}
	shock_std = 0.08
	steps_per_trajectory = 80

	X_poor, Y_poor = _build_stacked_independent_transitions(
		W,
		n_trajectories=2,
		steps_per_trajectory=steps_per_trajectory,
		shock_std=shock_std,
		rng=rng,
	)
	X_rich, Y_rich = _build_stacked_independent_transitions(
		W,
		n_trajectories=8,
		steps_per_trajectory=steps_per_trajectory,
		shock_std=shock_std,
		rng=rng,
	)

	poor = compute_eigenvalue(X_poor, Y_poor, neighbors, intercepts=False)
	rich = compute_eigenvalue(X_rich, Y_rich, neighbors, intercepts=False)

	poor_min = _min_positive_eigvals(poor['eigvals_full'])
	rich_min = _min_positive_eigvals(rich['eigvals_full'])

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

	X, Y = _build_transitions(W, steps=180, shock_std=0.08, rng=rng)
	out_sparse = compute_eigenvalue(X, Y, neighbors, intercepts=False)
	out_all = compute_eigenvalue(X, Y, all_neighbors, intercepts=False)

	assert out_sparse['gram_full_shape'] == (n * n, n * n)
	assert out_all['gram_full_shape'] == (n * n, n * n)
	np.testing.assert_allclose(out_sparse['eigvals_full'], out_all['eigvals_full'], rtol=1e-9, atol=1e-11)
