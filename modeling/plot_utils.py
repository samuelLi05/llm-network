"""
Publication-ready plotting utilities based on analysis_utils.

Provides styled versions of metric plotting functions:
- plot_mean_per_timestep(): Observed vs Predicted means
- plot_variance_per_timestep(): Observed vs Predicted variances
- plot_wasserstein_distance_per_timestep(): Wasserstein distance metric

All functions follow publication-quality styling with SVG output.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from typing import Optional, Tuple

Array = np.ndarray

# Publication styling - compact, professional
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['lines.linewidth'] = 1.8
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3


def compute_mean_per_timestep(y_true: Array, y_pred: Array) -> Tuple[Array, Array]:
    """Compute mean across agents for each timestep.
    
    Parameters:
    -----------
    y_true : array of shape (T, n_agents)
    y_pred : array of shape (T, n_agents)
    
    Returns:
    --------
    mean_true, mean_pred : arrays of shape (T,)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    mean_true = np.mean(y_true, axis=1)
    mean_pred = np.mean(y_pred, axis=1)

    return mean_true, mean_pred


def compute_variance_per_timestep(y_true: Array, y_pred: Array) -> Tuple[Array, Array]:
    """Compute variance across agents for each timestep.
    
    Parameters:
    -----------
    y_true : array of shape (T, n_agents)
    y_pred : array of shape (T, n_agents)
    
    Returns:
    --------
    var_true, var_pred : arrays of shape (T,)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    var_true = np.var(y_true, axis=1)
    var_pred = np.var(y_pred, axis=1)

    return var_true, var_pred


def compute_wasserstein_distance_per_timestep(y_true: Array, y_pred: Array) -> Array:
    """Compute Wasserstein distance between distributions at each timestep.
    
    Parameters:
    -----------
    y_true : array of shape (T, n_agents)
    y_pred : array of shape (T, n_agents)
    
    Returns:
    --------
    wasserstein_per_timestep : array of shape (T,)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    T = min(y_true.shape[0], y_pred.shape[0])
    wasserstein_per_timestep = np.zeros(T)
    for t in range(T):
        wasserstein_per_timestep[t] = wasserstein_distance(y_true[t, :], y_pred[t, :])

    return wasserstein_per_timestep


def plot_mean_per_timestep(
    mean_true: Array,
    mean_pred: Array,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot observed vs predicted mean per timestep.
    
    Publication-ready styling with SVG output.
    """
    mean_true = np.asarray(mean_true, dtype=float).ravel()
    mean_pred = np.asarray(mean_pred, dtype=float).ravel()
    
    t = np.arange(len(mean_true))
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(t, mean_true, label='Observed', color='#1f77b4', linewidth=1.8, marker='o', markersize=3)
    ax.plot(t, mean_pred, label='Predicted', color='#ff7f0e', linewidth=1.8, marker='s', markersize=3)
    
    ax.set_xlabel('Time (steps)', fontsize=9, fontweight='normal')
    ax.set_ylabel('Mean', fontsize=9, fontweight='normal')
    if title:
        ax.set_title(title, fontsize=10, fontweight='normal', pad=8)
    else:
        ax.set_title('Mean Per Timestep', fontsize=10, fontweight='normal', pad=8)
    
    ax.legend(loc='best', frameon=False, fontsize=8)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.4)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format='svg', bbox_inches='tight')
    
    return fig, ax


def plot_variance_per_timestep(
    var_true: Array,
    var_pred: Array,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot observed vs predicted variance per timestep.
    
    Publication-ready styling with SVG output.
    """
    var_true = np.asarray(var_true, dtype=float).ravel()
    var_pred = np.asarray(var_pred, dtype=float).ravel()
    
    t = np.arange(len(var_true))
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(t, var_true, label='Observed', color='#1f77b4', linewidth=1.8, marker='o', markersize=3)
    ax.plot(t, var_pred, label='Predicted', color='#ff7f0e', linewidth=1.8, marker='s', markersize=3)
    
    ax.set_xlabel('Time (steps)', fontsize=9, fontweight='normal')
    ax.set_ylabel('Variance', fontsize=9, fontweight='normal')
    if title:
        ax.set_title(title, fontsize=10, fontweight='normal', pad=8)
    else:
        ax.set_title('Variance Per Timestep', fontsize=10, fontweight='normal', pad=8)
    
    ax.legend(loc='best', frameon=False, fontsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.4)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format='svg', bbox_inches='tight')
    
    return fig, ax


def plot_wasserstein_distance_per_timestep(
    wasserstein_per_timestep: Array,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot Wasserstein distance per timestep.
    
    Publication-ready styling with SVG output.
    """
    wasserstein_per_timestep = np.asarray(wasserstein_per_timestep, dtype=float).ravel()
    
    t_arr = np.arange(len(wasserstein_per_timestep))
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(t_arr, wasserstein_per_timestep, label='Wasserstein Distance', 
            color='#2ca02c', linewidth=1.8, marker='o', markersize=3)
    ax.fill_between(t_arr, wasserstein_per_timestep, alpha=0.15, color='#2ca02c')
    
    ax.set_xlabel('Time (steps)', fontsize=9, fontweight='normal')
    ax.set_ylabel('Wasserstein Distance W₂', fontsize=9, fontweight='normal')
    if title:
        ax.set_title(title, fontsize=10, fontweight='normal', pad=8)
    else:
        ax.set_title('Wasserstein Distance Per Timestep', fontsize=10, fontweight='normal', pad=8)
    
    ax.legend(loc='best', frameon=False, fontsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.4)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format='svg', bbox_inches='tight')
    
    return fig, ax


def plot_observed_trajectories(
    observed: Array,
    agent_ids: Optional[list] = None,
    title: Optional[str] = None,
    horizon: Optional[int] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot individual agent trajectories for observed data.
    
    Publication-ready styling with SVG output.
    No legend shown for many agents (use as-is for visualization).
    
    Parameters:
    -----------
    observed : array of shape (T, n_agents)
    agent_ids : list of agent identifiers (optional, ignored for legend)
    title : plot title
    horizon : max timesteps to plot (default: all)
    save_path : path to save figure
    """
    observed = np.asarray(observed, dtype=float)
    if observed.ndim == 1:
        observed = observed[:, None]
    
    n_agents = observed.shape[1]
    
    if horizon is None:
        horizon = observed.shape[0] - 1
    T = min(int(horizon), observed.shape[0] - 1)
    t = np.arange(T + 1)
    
    # Generate distinct colors for agents
    palette = np.array(
        list(plt.cm.tab20.colors) + list(plt.cm.Set3.colors) + list(plt.cm.Dark2.colors)
    )[: n_agents]
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    
    for i in range(n_agents):
        color = palette[i][:3]
        ax.plot(t, observed[: T + 1, i], color=color, linewidth=1.2, alpha=0.7)
    
    ax.set_xlabel('Time (steps)', fontsize=9, fontweight='normal')
    ax.set_ylabel('Value', fontsize=9, fontweight='normal')
    if title:
        ax.set_title(title, fontsize=10, fontweight='normal', pad=8)
    else:
        ax.set_title(f'Observed Trajectories', fontsize=10, fontweight='normal', pad=8)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.4)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format='svg', bbox_inches='tight')
    
    return fig, ax


def plot_predicted_vs_observed(
    observed: Array,
    predicted: Array,
    agent_ids: Optional[list] = None,
    title: Optional[str] = None,
    horizon: Optional[int] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot observed vs predicted trajectories for agents.
    
    Publication-ready styling with SVG output.
    Observed as solid lines, predicted as dashed lines, no legend.
    
    Parameters:
    -----------
    observed : array of shape (T, n_agents)
    predicted : array of shape (T, n_agents)
    agent_ids : list of agent identifiers (optional, unused)
    title : plot title
    horizon : max timesteps to plot
    save_path : path to save figure
    """
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    
    if observed.ndim == 1:
        observed = observed[:, None]
    if predicted.ndim == 1:
        predicted = predicted[:, None]
    
    n_agents = min(observed.shape[1], predicted.shape[1])
    
    if horizon is None:
        horizon = min(observed.shape[0], predicted.shape[0]) - 1
    T = min(int(horizon), observed.shape[0] - 1, predicted.shape[0] - 1)
    t = np.arange(T + 1)
    
    palette = np.array(
        list(plt.cm.tab20.colors) + list(plt.cm.Set3.colors) + list(plt.cm.Dark2.colors)
    )[: n_agents]
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    
    for i in range(n_agents):
        color = palette[i][:3]
        # Observed: solid line
        ax.plot(t, observed[: T + 1, i], color=color, linewidth=1.2, alpha=0.7, linestyle='-')
        # Predicted: dashed line
        ax.plot(t, predicted[: T + 1, i], color=color, linewidth=1.2, alpha=0.9, linestyle='--')
    
    ax.set_xlabel('Time (steps)', fontsize=9, fontweight='normal')
    ax.set_ylabel('Value', fontsize=9, fontweight='normal')
    if title:
        ax.set_title(title, fontsize=10, fontweight='normal', pad=8)
    else:
        ax.set_title('Predicted vs Observed', fontsize=10, fontweight='normal', pad=8)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.4)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format='svg', bbox_inches='tight')
    
    return fig, ax


def plot_violin_per_timestep(
    y_true: Array,
    y_pred: Array,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot violin plots showing distribution of values at each timestep.
    
    Publication-ready styling with SVG output.
    Intelligently spaces x-axis labels (max ~15 ticks).
    
    Parameters:
    -----------
    y_true : array of shape (T, n_agents)
    y_pred : array of shape (T, n_agents)
    title : plot title
    save_path : path to save figure
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    
    T = min(y_true.shape[0], y_pred.shape[0])
    
    fig, ax = plt.subplots(figsize=(10, 3.5))
    
    positions_obs = []
    positions_pred = []
    data_obs = []
    data_pred = []
    
    for t in range(T):
        pos_obs = 2 * t
        pos_pred = 2 * t + 1
        
        positions_obs.append(pos_obs)
        positions_pred.append(pos_pred)
        data_obs.append(y_true[t, :])
        data_pred.append(y_pred[t, :])
    
    vp_obs = ax.violinplot(data_obs, positions=positions_obs, widths=0.7, showmeans=True, 
                           showmedians=False, showextrema=False)
    vp_pred = ax.violinplot(data_pred, positions=positions_pred, widths=0.7, showmeans=True,
                            showmedians=False, showextrema=False)
    
    for pc in vp_obs['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.4)
    
    for pc in vp_pred['bodies']:
        pc.set_facecolor('#ff7f0e')
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.4)
    
    # Style the mean lines
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = vp_obs.get(partname)
        if vp is not None:
            vp.set_edgecolor('black')
            vp.set_linewidth(0.8)
    
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = vp_pred.get(partname)
        if vp is not None:
            vp.set_edgecolor('black')
            vp.set_linewidth(0.8)
    
    # Intelligently space x-axis labels (aim for ~15 max labels)
    max_labels = 15
    if T <= max_labels:
        tick_step = 1
    else:
        tick_step = int(np.ceil(T / max_labels))
    
    tick_positions = [2 * t + 0.5 for t in range(0, T, tick_step)]
    tick_labels = [str(t) for t in range(0, T, tick_step)]
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_xlabel('Time (steps)', fontsize=9, fontweight='normal')
    ax.set_ylabel('Value', fontsize=9, fontweight='normal')
    
    if title:
        ax.set_title(title, fontsize=10, fontweight='normal', pad=8)
    else:
        ax.set_title('Distribution Per Timestep', fontsize=10, fontweight='normal', pad=8)
    
    # Add custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', alpha=0.6, edgecolor='black', linewidth=0.4, label='Observed'),
        Patch(facecolor='#ff7f0e', alpha=0.6, edgecolor='black', linewidth=0.4, label='Predicted'),
    ]
    ax.legend(handles=legend_elements, loc='best', frameon=False, fontsize=8)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.4)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format='svg', bbox_inches='tight')
    
    return fig, ax
