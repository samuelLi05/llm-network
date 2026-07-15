"""
A script to go through the data and find training runs to which we can fit models with social effects and get improvements
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
print(f"Current working directory : {ROOT}")
# ensure project imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PARAMS = {
    'target_agent_fraction': 0.4,
    'constrain_messages': 150,
    'rollout_horizon_cap': 20,
}

from modeling.models.data_prep import load_run_data, build_run_trajectory, build_neighbors_index, _numeric_agent_key, build_row_normalized_adjacency  # type: ignore
from modeling.models.adjacency_based.friedkin_johnsen import (
    select_friedkin_johnsen_adjacency_lambdas,
)  # type: ignore
from modeling.models.adjacency_based.bias_only import fit_bias_init_only_model

LLM = "qwen-3.1"
TOPIC = "climate"
REVERSE = True

if __name__ == "__main__":

    RUNS_PATH_TRAIN = ROOT / 'modeling' / 'runs' / LLM / TOPIC / 'train'
    run_dirs = [file for file in RUNS_PATH_TRAIN.iterdir() if file.is_dir()]

    run_data = {r.name: load_run_data(r) for r in run_dirs} 
    global_agents = sorted({a for d in run_data.values() for a in d['agent_ids']}, key=_numeric_agent_key)
    n_agents = len(global_agents)
    
    traj_mask = {rn: build_run_trajectory(d, global_agents, target_agent_fraction=PARAMS['target_agent_fraction'], return_post_mask=True, constrain_messages=PARAMS['constrain_messages']) for rn, d in run_data.items()}
    run_traj = {rn: tm[0] for rn, tm in traj_mask.items()}
    run_neighbors = {rn: build_neighbors_index(d, global_agents, reverse=True) for rn, d in run_data.items()}

    # Fit FJ model, and bias-init only models to each trajectory
    
    individual_run_mse_pct_diff = {}

    for rn, traj in run_traj.items():

        run_traj_single = {rn:traj}
        run_nbh_single = {rn: run_neighbors[rn]}


        BEST_FJ_ADJ, _ = select_friedkin_johnsen_adjacency_lambdas(run_traj_single, run_nbh_single)
        FJ_ADJ_MSE = BEST_FJ_ADJ['mse_pool']

        BEST_BIAS_INIT_ONLY = fit_bias_init_only_model(run_traj_single)
        BIAS_INIT_ONLY_MSE = BEST_BIAS_INIT_ONLY['mse_pool']

        pct_improvement = 100*(BIAS_INIT_ONLY_MSE - FJ_ADJ_MSE) / BIAS_INIT_ONLY_MSE

        individual_run_mse_pct_diff[rn] = pct_improvement

print(individual_run_mse_pct_diff)

sorted_diff = sorted(individual_run_mse_pct_diff.items(), key = lambda kv: kv[1])

k = 5
for i in range(k):
    rn, pct = sorted_diff[-(i+1)]
    print(f" Run {rn} improvement is {individual_run_mse_pct_diff[rn]}")


labels, values = zip(*sorted_diff)

plt.bar(labels, values)
plt.xticks([])
plt.xlabel("Runs")
plt.ylabel("Pct improvement in MSE from \n fitting a model with social effects")
plt.show()

