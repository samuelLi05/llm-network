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


def _min_positive_eigvals(values, tol=1e-12):
	vals = np.asarray(values, dtype=float)
	pos = vals[vals > tol]
	if pos.size == 0:
		return 0.0
	return float(np.min(pos))


def _assert_multiset_subset_with_tolerance(sub_vals, full_vals, rtol=1e-8, atol=1e-10):
	sub_vals = np.asarray(sub_vals, dtype=float)
	full_vals = np.asarray(full_vals, dtype=float)
	used = np.zeros(full_vals.shape[0], dtype=bool)

	for sv in sub_vals:
		idx = np.where(np.isclose(full_vals, sv, rtol=rtol, atol=atol) & (~used))[0]
		if idx.size == 0:
			raise AssertionError(f'No matching full-spectrum eigenvalue found for reduced eigenvalue {sv}')
		used[idx[0]] = True


def test_compute_eigenvalue_full_equals_reduced_for_all_neighbors():
	rng = np.random.default_rng(7)
	n = 8
	W = _random_row_stochastic_matrix(n, rng)
	X, Y = _build_transitions(W, steps=80, shock_std=0.05, rng=rng)
	neighbors = {i: list(range(n)) for i in range(n)}

	out = compute_eigenvalue(X, Y, neighbors, intercepts=False)

	assert out['gram_full_shape'] == (n * n, n * n)
	assert out['gram_reduced_shape'] == (n * n, n * n)
	np.testing.assert_allclose(out['eigvals_full'], out['eigvals_reduced'], rtol=1e-9, atol=1e-11)


def test_compute_eigenvalue_min_reduced_eig_increases_with_richer_transitions():
	rng = np.random.default_rng(11)
	n = 10
	W = _random_row_stochastic_matrix(n, rng)
	neighbors = {i: list(range(n)) for i in range(n)}

	X_poor, Y_poor = _build_transitions(W, steps=120, shock_std=1e-9, rng=rng)
	X_rich, Y_rich = _build_transitions(W, steps=320, shock_std=0.20, rng=rng)

	poor = compute_eigenvalue(X_poor, Y_poor, neighbors, intercepts=False)
	rich = compute_eigenvalue(X_rich, Y_rich, neighbors, intercepts=False)

	poor_min = _min_positive_eigvals(poor['eigvals_reduced'])
	rich_min = _min_positive_eigvals(rich['eigvals_reduced'])

	assert rich_min > poor_min


def test_compute_eigenvalue_reduced_is_subset_of_full_with_sparse_neighbors():
	rng = np.random.default_rng(19)
	n = 10
	neighbors = {
		i: sorted({i, (i - 1) % n, (i + 1) % n, (i + 3) % n})
		for i in range(n)
	}

	W = np.zeros((n, n), dtype=float)
	for i in range(n):
		ns = neighbors[i]
		W[i, ns] = rng.dirichlet(np.ones(len(ns)))

	X, Y = _build_transitions(W, steps=180, shock_std=0.08, rng=rng)
	out = compute_eigenvalue(X, Y, neighbors, intercepts=False)

	assert out['active_param_count'] < out['full_param_count']
	_assert_multiset_subset_with_tolerance(
		out['eigvals_reduced'],
		out['eigvals_full'],
		rtol=1e-7,
		atol=1e-9,
	)
