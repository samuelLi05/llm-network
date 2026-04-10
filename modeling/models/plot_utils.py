import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from matplotlib.lines import Line2D

def plot_observed_trajectories(run_name, observed, agent_ids, horizon):
    observed = np.asarray(observed)
    if observed.ndim == 1:
        observed = observed[:, None]

    T = min(int(horizon), observed.shape[0] - 1)
    t = np.arange(T + 1)

    palette = np.array(
        list(plt.cm.tab20.colors) + list(plt.cm.Set3.colors) + list(plt.cm.Dark2.colors),
    )[: len(agent_ids)]

    plt.figure(figsize=(10.2, 5.6))
    for i, _ in enumerate(agent_ids):
        obs_rgb = 0.60 * np.array([0.72, 0.72, 0.72]) + 0.40 * np.asarray(palette[i][:3], dtype=float)

        plt.plot(
            t,
            observed[: T + 1, i],
            color=(*obs_rgb, 0.42),
            linewidth=0.9,
            marker='o',
            markersize=1.9,
        )

    plt.title(f'{run_name}: observed rollout (first {T} slices)')
    plt.xlabel('time slice')
    plt.ylabel('stance score')
    plt.tight_layout()
    plt.show()

def plot_predicted_vs_observed(run_name, observed, predicted, agent_ids, horizon):
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)

    if observed.ndim == 1:
        observed = observed[:, None]
    if predicted.ndim == 1:
        predicted = predicted[:, None]

    T = min(int(horizon), observed.shape[0] - 1, predicted.shape[0] - 1)
    t = np.arange(T + 1)

    palette = np.array(
        list(plt.cm.tab20.colors) + list(plt.cm.Set3.colors) + list(plt.cm.Dark2.colors),
    )[: len(agent_ids)]

    plt.figure(figsize=(10.2, 5.6))
    for i, _ in enumerate(agent_ids):
        base_rgb = np.asarray(palette[i][:3], dtype=float)
        obs_rgb = 0.60 * np.array([0.72, 0.72, 0.72]) + 0.40 * base_rgb

        plt.plot(
            t,
            observed[: T + 1, i],
            color=(*obs_rgb, 0.42),
            linewidth=0.9,
            marker='o',
            markersize=1.9,
        )
        plt.plot(
            t,
            predicted[: T + 1, i],
            color=(*base_rgb, 0.92),
            linewidth=1.35,
        )

    legend_handles = [
        Line2D([0], [0], color=(0.45, 0.45, 0.45, 0.45), linewidth=1.0, marker='o', markersize=3, label='observed'),
        Line2D([0], [0], color=(0.20, 0.20, 0.20, 0.95), linewidth=1.6, label='predicted'),
    ]
    plt.title(f'{run_name}: predicted vs observed rollout (first {T} slices)')
    plt.xlabel('time slice')
    plt.ylabel('stance score')
    plt.legend(handles=legend_handles, loc='upper right', frameon=False)
    plt.tight_layout()
    plt.show()


def calculate_mean_and_variance(y_true, y_pred, last_n=3):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_pred must have the same shape')

    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]

    T = y_true.shape[0]
    if T <= 0:
        return {
            'steady_mean_obs': np.nan,
            'steady_mean_pred': np.nan,
            'steady_var_obs': np.nan,
            'steady_var_pred': np.nan,
        }

    start = max(0, T - last_n)
    y_true_ss = y_true[start:]
    y_pred_ss = y_pred[start:]

    steady_mean_obs = float(np.mean(y_true_ss))
    steady_mean_pred = float(np.mean(y_pred_ss))
    steady_var_obs = float(np.var(y_true_ss))
    steady_var_pred = float(np.var(y_pred_ss))

    return {
        'steady_mean_obs': steady_mean_obs,
        'steady_mean_pred': steady_mean_pred,
        'steady_var_obs': steady_var_obs,
        'steady_var_pred': steady_var_pred,
    }


def compute_wasserstein_distance(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0 or y_pred.size == 0:
        return np.nan

    return float(wasserstein_distance(y_true, y_pred))


def compute_eigenvalue(X, Y, neighbors, intercepts):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape != Y.shape:
        raise ValueError('X and Y must be 2D arrays with identical shape (num_samples, n_agents)')

    m, n = X.shape

    # Start from the exact least-squares construction.
    # For one transition z_t, W z_t produces [w_1^T z_t, ..., w_n^T z_t]^T.
    # Define Q(z_t) = I_n ⊗ z_t^T, so
    #   Q(z_t) w = W z_t
    # with w flattened row-wise from W.
    # Stacking all transitions gives the full design matrix A.
    # If mask entries are inactive, those columns become exact zeros in A.
    # Then A has the exact structure:
    #   A = [A_active | 0]
    # and the Gram is:
    #   A^T A = [[A_active^T A_active, 0], [0, 0]].
    # Therefore the eigenvalues of A_active^T A_active are contained in the
    # eigenvalues of A^T A, independent of column order.

    # Build mask for neighbor structure over flattened W (size n^2)
    mask = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in neighbors[i]:
            mask[i, j] = 1.0
    mask_flat = mask.reshape(-1)  # shape (n^2,)

    q_blocks = []
    for t in range(m):
        zt = X[t].reshape(1, -1)  # (1, n)
        Q_t = np.kron(np.eye(n, dtype=float), zt)
        q_blocks.append(Q_t)

    A = np.vstack(q_blocks)
    y_vec = Y.reshape(-1)

    print (A)

    if intercepts:
        A = np.hstack([A, np.ones((A.shape[0], 1), dtype=float)])

    gram_full = A.T @ A
    eigvals_full = np.linalg.eigvalsh(gram_full)

    active_cols = mask_flat != 0
    if intercepts:
        active_cols = np.concatenate([active_cols, [True]])

    return {
        'eigvals_full': eigvals_full,
        'gram_full_shape': gram_full.shape,
    }


