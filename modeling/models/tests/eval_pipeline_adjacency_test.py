from pathlib import Path
import sys

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR.parent
if str(MODELS_DIR) not in sys.path:
	sys.path.insert(0, str(MODELS_DIR))
if str(THIS_DIR) not in sys.path:
	sys.path.insert(0, str(THIS_DIR))

from data_prep import build_expected_message_matrix
from adjacency_based.degroot import fit_degroot_adjacency_scalar
from adjacency_based.friedkin_johnsen import fit_friedkin_johnsen_adjacency
from adjacency_based.homophily import fit_homophily, fit_homophily_friedkin_johnsen, fit_homophily_stubborness
from synthetic_data import (
	build_neighbors_all_to_all,
	build_synthetic_fj_runs,
	build_synthetic_fj_runs_from_neighbors,
	build_synthetic_homophily_runs,
	build_synthetic_linear_runs,
	build_synthetic_linear_runs_from_neighbors,
	build_synthetic_homophily_runs_from_neighbors,
)


def _row_normalize(mat):
	mat = np.asarray(mat, dtype=float)
	row_sums = mat.sum(axis=1, keepdims=True)
	out = np.zeros_like(mat)
	valid = row_sums[:, 0] > 0.0
	out[valid] = mat[valid] / row_sums[valid]
	zero_idx = np.where(~valid)[0]
	for i in zero_idx:
		out[i, i] = 1.0
	return out


def _assert_rows_stochastic(w, atol=1e-6):
	w = np.asarray(w, dtype=float)
	np.testing.assert_allclose(w.sum(axis=1), np.ones((w.shape[0],), dtype=float), atol=atol, rtol=0.0)
	assert np.all(w >= -1e-10)


def test_degroot_adjacency_scalar_fit_returns_row_stochastic_w_and_low_mse():
	rng = np.random.default_rng(2026)
	n = 6
	neighbors = build_neighbors_all_to_all(n)
	run_neighbors = {f'run_{i:02d}': neighbors for i in range(8)}

	Abar = build_expected_message_matrix(neighbors, n)
	gamma_true = 0.65
	W_true = _row_normalize(gamma_true * Abar + (1.0 - gamma_true) * np.eye(n))
	run_traj_map = build_synthetic_linear_runs_from_neighbors(rng, neighbors, n_runs=8, horizon=24, noise_std=0.0)

	fit = fit_degroot_adjacency_scalar(run_traj_map, run_neighbors)

	assert np.isfinite(fit['mse_pool'])
	X_pool = np.asarray(fit['X_pool'], dtype=float)
	Y_pool = np.asarray(fit['Y_pool'], dtype=float)
	baseline_mse = float(np.mean((Y_pool - X_pool) ** 2))
	assert fit['mse_pool'] < baseline_mse
	assert 0.0 <= float(fit['gamma']) <= 1.0
	for w in fit['W_blocks'].values():
		_assert_rows_stochastic(w)


def test_fj_adjacency_fit_returns_row_stochastic_w_and_low_mse():
	rng = np.random.default_rng(2027)
	n = 6
	neighbors = build_neighbors_all_to_all(n)
	x0_prior = rng.uniform(-0.8, 0.8, size=n)
	lambda1 = 0.2
	lambda2 = 0.25
	bias = -0.15

	_, run_traj_map, run_neighbors = build_synthetic_fj_runs_from_neighbors(
		rng=rng,
		neighbors=neighbors,
		n_runs=10,
		horizon=24,
		lambda1=lambda1,
		lambda2=lambda2,
		b=bias,
		x0_prior=x0_prior,
		noise_std=0.0,
	)

	fit = fit_friedkin_johnsen_adjacency(run_traj_map, run_neighbors, lambda1=lambda1, lambda2=lambda2)

	assert np.isfinite(fit['mse_pool'])
	X_pool = np.asarray(fit['X_pool'], dtype=float)
	Y_pool = np.asarray(fit['Y_pool'], dtype=float)
	X0_pool = np.asarray(fit['X0_pool'], dtype=float)
	alpha = 1.0 - lambda1 - lambda2
	baseline = lambda1 * X0_pool + alpha * X_pool
	baseline_mse = float(np.mean((Y_pool - baseline) ** 2))
	assert fit['mse_pool'] < baseline_mse
	assert 0.0 <= float(fit['gamma']) <= 1.0
	assert -1.0 <= float(fit['bias']) <= 1.0
	for w in fit['W_blocks'].values():
		_assert_rows_stochastic(w)


def test_homophily_adjacency_variants_fit_low_mse():
	rng = np.random.default_rng(2028)
	n = 5
	neighbors = build_neighbors_all_to_all(n)
	run_neighbors = {f'run_{i:02d}': neighbors for i in range(6)}
	Abar = build_expected_message_matrix(neighbors, n)

	# Use neighbors-based generator for adjacency-focused tests
	runs_plain = build_synthetic_homophily_runs_from_neighbors(
		rng=rng,
		n=n,
		n_runs=6,
		horizon=18,
		neighbors=neighbors,
		gamma=1.0,
		noise_std=0.0,
		poisson_mean=None,
		lambda_self=0.2,
	)
	fit_plain = fit_homophily(runs_plain, run_neighbors, gamma0=1.0)
	assert fit_plain['success']
	assert fit_plain['mse_pool'] < 1e-9

	runs_fj = build_synthetic_homophily_runs_from_neighbors(
		rng=rng,
		n=n,
		n_runs=6,
		horizon=18,
		neighbors=neighbors,
		gamma=0.9,
		noise_std=0.0,
		poisson_mean=None,
		lambda_self=0.2,
		lambda1=0.25,
	)
	fit_fj = fit_homophily_friedkin_johnsen(runs_fj, run_neighbors, gamma0=1.0)
	assert fit_fj['success']
	assert fit_fj['mse_pool'] < 1e-5

	runs_bias = build_synthetic_homophily_runs_from_neighbors(
		rng=rng,
		n=n,
		n_runs=6,
		horizon=18,
		neighbors=neighbors,
		gamma=1.1,
		noise_std=0.0,
		poisson_mean=None,
		lambda_self=0.2,
		lambda1=0.2,
		lambda2=0.2,
		bias=-0.1,
	)
	fit_bias = fit_homophily_stubborness(runs_bias, run_neighbors, gamma0=1.0)
	assert fit_bias['success']
	assert fit_bias['mse_pool'] < 1e-9
