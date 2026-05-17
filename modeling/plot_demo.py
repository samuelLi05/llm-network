"""
Simple demo of plot_utils.py publication-ready plotting functions.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, '/Users/samuelli/Documents/code/llm/llm-network/modeling')
import plot_utils

OUT_DIR = os.path.join("modeling", "figs")
os.makedirs(OUT_DIR, exist_ok=True)

# Generate synthetic data: one run with 15 timesteps and 30 agents
T = 15
n_agents = 30
rng = np.random.default_rng(42)

# Observed: smooth trajectory
observed = np.cumsum(0.015 * rng.normal(size=(T, n_agents)), axis=0) + 0.2
observed = np.clip(observed, -1, 1)

# Predicted: slightly noisy version
predicted = observed + 0.04 * rng.normal(size=(T, n_agents))
predicted = np.clip(predicted, -1, 1)

print("Generating publication-ready SVG plots...")

# ============================================================================
# Per-timestep metrics
# ============================================================================

# Plot 1: Mean per timestep
mean_true, mean_pred = plot_utils.compute_mean_per_timestep(observed, predicted)
plot_utils.plot_mean_per_timestep(
    mean_true, mean_pred,
    title='Mean Per Timestep',
    save_path=os.path.join(OUT_DIR, '01_mean_per_timestep.svg')
)
print(f"  Mean: max_obs={np.max(mean_true):.4f}, max_pred={np.max(mean_pred):.4f}")

# Plot 2: Variance per timestep
var_true, var_pred = plot_utils.compute_variance_per_timestep(observed, predicted)
plot_utils.plot_variance_per_timestep(
    var_true, var_pred,
    title='Variance Per Timestep',
    save_path=os.path.join(OUT_DIR, '02_variance_per_timestep.svg')
)
print(f"  Variance: max_obs={np.max(var_true):.4f}, max_pred={np.max(var_pred):.4f}")

# Plot 3: Wasserstein distance per timestep
wass = plot_utils.compute_wasserstein_distance_per_timestep(observed, predicted)
plot_utils.plot_wasserstein_distance_per_timestep(
    wass,
    title='Wasserstein Distance Per Timestep',
    save_path=os.path.join(OUT_DIR, '03_wasserstein_per_timestep.svg')
)
print(f"  Wasserstein: max={np.max(wass):.4f}, mean={np.mean(wass):.4f}")

# ============================================================================
# Trajectory plots (all 30 agents, no legend)
# ============================================================================

# Plot 4: Observed trajectories (all agents)
plot_utils.plot_observed_trajectories(
    observed,
    title='Observed Agent Trajectories (n=30)',
    horizon=15,
    save_path=os.path.join(OUT_DIR, '04_observed_trajectories.svg')
)
print(f"  Observed trajectories plotted (all 30 agents)")

# Plot 5: Predicted vs Observed (all agents)
plot_utils.plot_predicted_vs_observed(
    observed, predicted,
    title='Predicted vs Observed (solid=obs, dashed=pred)',
    horizon=15,
    save_path=os.path.join(OUT_DIR, '05_predicted_vs_observed.svg')
)
print(f"  Predicted vs Observed plotted (all 30 agents)")

# ============================================================================
# Distribution plots
# ============================================================================

# Plot 6: Violin plot showing distributions at each timestep
plot_utils.plot_violin_per_timestep(
    observed, predicted,
    title='Distribution Per Timestep: Observed vs Predicted',
    save_path=os.path.join(OUT_DIR, '06_violin_per_timestep.svg')
)
print(f"  Violin plot generated ({T} timesteps)")

print(f'\nDone! Figures saved to {OUT_DIR}')
for fname in sorted(os.listdir(OUT_DIR)):
    if fname.endswith('.svg'):
        print(f'  {fname}')
