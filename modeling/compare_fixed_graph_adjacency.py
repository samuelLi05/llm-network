from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
# ensure project imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.models.data_prep import load_run_data, build_run_trajectory, build_neighbors_index, _numeric_agent_key, build_row_normalized_adjacency  # type: ignore

from modeling.models.adjacency_based.friedkin_johnsen import(
    select_friedkin_johnsen_adjacency_lambdas,
    friedkin_johnsen_adjacency_rollout
)

from modeling.models.fixed_graph.friedkin_johnsen import(
    fit_friedkin_johnsen_joint_traj0
)

# Import evaluation code from generate_model_rankings.py
from modeling.generate_model_rankings import (
    evaluate_model,
)

MODEL_DISPLAY_NAMES = {
    'fj_adj': 'Friedkin-Johnsen (low-parameter)',
    'fj_fg': 'Friedkin-Johnsen (high-parameter)',
    'fj_fg_ngc': 'Friedkin-Johnsen (high-parameter, no graph constraints)',
}

PARAMS = {
    'target_agent_fraction': 0.4,
    'constrain_messages': 150,
    'rollout_horizon_cap': 20,
}


if __name__ == '__main__':
    RUNS_DIR = ROOT / 'modeling' / 'runs_fg_vs_adj'
    train_path = RUNS_DIR / 'fixed_graph' / 'train'
    test_path = RUNS_DIR / 'fixed_graph' / 'test'
    

    combo_dir = ROOT / 'fixed_graph_adjacency_comparison'
    combo_dir.mkdir(parents=True, exist_ok=True)

    llm_name = 'TBD'
    topic_name = 'vaccines'


    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Train or test path does not exist. Please run the fixed graph and adjacency model fitting scripts first.")
    
    train_run_dirs = sorted([p for p in train_path.iterdir() if p.is_dir()])
    try:
        run_data = {r.name: load_run_data(r) for r in train_run_dirs}
        global_agent_ids = sorted({a for d in run_data.values() for a in d['agent_ids']}, key=_numeric_agent_key)
        n_agents = len(global_agent_ids)
        traj_mask = {rn: build_run_trajectory(d, global_agent_ids, target_agent_fraction=PARAMS['target_agent_fraction'], return_post_mask=True, constrain_messages=PARAMS['constrain_messages']) for rn, d in run_data.items()}
        run_traj = {rn: tm[0] for rn, tm in traj_mask.items()}
        run_neighbors = {rn: build_neighbors_index(d, global_agent_ids) for rn, d in run_data.items()}

        # validte that run neighbors are consistent ACROSS runs
        for rn, nbrs in run_neighbors.items():
            for other_rn, other_nbrs in run_neighbors.items():
                if rn == other_rn:
                    continue
                assert nbrs == other_nbrs, f"Neighbors differ between runs {rn} and {other_rn}"

    except Exception as e:
        print(f'Error loading run data: {str(e)[:120]}')
        raise e
    
    # Fit adjacency-based and fixed graph models
    try:
        # Fit Fj adjacency model with bias
        BEST_FJ_ADJ, _ = select_friedkin_johnsen_adjacency_lambdas(run_traj, run_neighbors)
        FJ_ADJ_L1 = BEST_FJ_ADJ['lambda1']
        FJ_ADJ_L2 = BEST_FJ_ADJ['lambda2']
        print("Best fj adjacency lambdas: lambda1 = ", FJ_ADJ_L1, " lambda2 = ", FJ_ADJ_L2)
        FJ_ADJ_GAMMA = BEST_FJ_ADJ['gamma']
        FJ_ADJ_BIAS = BEST_FJ_ADJ['bias']
        FJ_ADJ_MSE = BEST_FJ_ADJ['mse_pool']
        TOTAL_POINTS_FJ_BIAS = int(BEST_FJ_ADJ.get('total_points', 0))
        print("Finished fitting fj with bias with fixed adjacency matrices; MSE = ", FJ_ADJ_MSE, " with total points = ", TOTAL_POINTS_FJ_BIAS)

        # Fit fixed graph friedkin johnsen model
        BEST_FJ_FG = fit_friedkin_johnsen_joint_traj0(run_traj, run_neighbors, turn_off_graph_constraints=False)
        FJ_FG_LAMBDA1 = BEST_FJ_FG['lambda1']
        FJ_FG_LAMBDA2 = BEST_FJ_FG['lambda2']
        print("Best fj fixed graph lambdas: lambda1 = ", FJ_FG_LAMBDA1, " lambda2 = ", FJ_FG_LAMBDA2)
        FJ_FG_MSE = BEST_FJ_FG['mse_pool']
        FJ_FG_BIAS = BEST_FJ_FG['b']
        FJ_FG_W = BEST_FJ_FG['W']
        TOTAL_POINTS_FJ_FG = int(BEST_FJ_FG.get('total_points', 0))
        print("Finished fitting fj with a free adjacency matrix; MSE = ", FJ_FG_MSE, " with total points = ", TOTAL_POINTS_FJ_FG)

        BEST_FJ_FG_NO_GRAPH_CONSTRAINT = fit_friedkin_johnsen_joint_traj0(run_traj, run_neighbors, turn_off_graph_constraints=True)
        FJ_FG_NO_GRAPH_CONSTRAINT_LAMBDA1 = BEST_FJ_FG_NO_GRAPH_CONSTRAINT['lambda1']
        FJ_FG_NO_GRAPH_CONSTRAINT_LAMBDA2 = BEST_FJ_FG_NO_GRAPH_CONSTRAINT['lambda2']
        print("Best fj fixed graph NO GRAPH CONSTRAINT lambdas: lambda1 = ", FJ_FG_NO_GRAPH_CONSTRAINT_LAMBDA1, " lambda2 = ", FJ_FG_NO_GRAPH_CONSTRAINT_LAMBDA2)
        FJ_FG_NO_GRAPH_CONSTRAINT_MSE = BEST_FJ_FG_NO_GRAPH_CONSTRAINT['mse_pool']
        FJ_FG_NO_GRAPH_CONSTRAINT_BIAS = BEST_FJ_FG_NO_GRAPH_CONSTRAINT['b']
        FJ_FG_NO_GRAPH_CONSTRAINT_W = BEST_FJ_FG_NO_GRAPH_CONSTRAINT['W']
        TOTAL_POINTS_FJ_FG_NO_GRAPH_CONSTRAINT = int(BEST_FJ_FG_NO_GRAPH_CONSTRAINT.get('total_points', 0))
        print("Finished fitting fj with a free adjacency matrix and no graph constraints; MSE = ", FJ_FG_NO_GRAPH_CONSTRAINT_MSE, " with total points = ", TOTAL_POINTS_FJ_FG_NO_GRAPH_CONSTRAINT)

    except Exception as e:
        print(f'Error fitting models: {str(e)[:120]}')
        raise e
    
    # Now construct rollout maps
    try:
        test_run_dirs = sorted([p for p in test_path.iterdir() if p.is_dir()])
        test_run_data = {r.name: load_run_data(r) for r in test_run_dirs}
        test_traj = {run_name: build_run_trajectory(data, global_agent_ids, target_agent_fraction=PARAMS['target_agent_fraction'], return_post_mask=False, constrain_messages=PARAMS['constrain_messages']) for run_name, data in test_run_data.items()}
        test_neighbors = {run_name: build_neighbors_index(data, global_agent_ids) for run_name, data in test_run_data.items()}
        # assert test neighbors are consistent across runs, and are the same as train neighbors
        for rn, nbrs in test_neighbors.items():
            for other_rn, other_nbrs in test_neighbors.items():
                if rn == other_rn:
                    continue
                assert nbrs == other_nbrs, f"Test neighbors differ between runs {rn} and {other_rn}"
            # also check against train neighbors
            for train_rn, train_nbrs in run_neighbors.items():
                assert nbrs == train_nbrs, f"Test neighbors in run {rn} differ from train neighbors in run {train_rn}"


        # build rollout maps. For adjacency model use nbs from first run (should be same across runs)
        def build_rollout_maps(traj_map, neighbors_map):
            return {
                'fj_adj': {
                    run_name: friedkin_johnsen_adjacency_rollout(
                        FJ_ADJ_GAMMA* build_row_normalized_adjacency(neighbors_map[run_name], n_agents) 
                        + (1.0 - FJ_ADJ_GAMMA) * np.eye(n_agents),
                        FJ_ADJ_BIAS,
                        np.asarray(traj_map[run_name], dtype=float)[0],
                        PARAMS['rollout_horizon_cap'],
                        FJ_ADJ_L1,
                        FJ_ADJ_L2,
                    )
                    for run_name in traj_map.keys()
                },
                'fj_fg': {
                    # note: friedkin_johnsen_adjacency_rollout can be used for the fixed graph case
                    run_name: friedkin_johnsen_adjacency_rollout(
                        FJ_FG_W,
                        FJ_FG_BIAS,
                        np.asarray(traj_map[run_name], dtype=float)[0],
                        PARAMS['rollout_horizon_cap'],
                        FJ_FG_LAMBDA1,
                        FJ_FG_LAMBDA2,
                    )
                    for run_name in traj_map.keys()
                },
                'fj_fg_ngc': {
                    run_name: friedkin_johnsen_adjacency_rollout(
                        FJ_FG_NO_GRAPH_CONSTRAINT_W,
                        FJ_FG_NO_GRAPH_CONSTRAINT_BIAS,
                        np.asarray(traj_map[run_name], dtype=float)[0],
                        PARAMS['rollout_horizon_cap'],
                        FJ_FG_NO_GRAPH_CONSTRAINT_LAMBDA1,
                        FJ_FG_NO_GRAPH_CONSTRAINT_LAMBDA2,
                    )
                    for run_name in traj_map.keys()
                },
            }


        rollout_maps_test = build_rollout_maps(test_traj, test_neighbors)

        # Evaluate models on both train and test
        per_run_frames = []
        summary_rows = []
        for raw_model_name in rollout_maps_test.keys():
            test_result = evaluate_model(raw_model_name, test_traj, rollout_maps_test[raw_model_name])

            result = test_result
            per_run_df = result['per_run'].copy()
            per_run_df['raw_model'] = raw_model_name
            per_run_df['model'] = MODEL_DISPLAY_NAMES.get(raw_model_name, raw_model_name)
            per_run_frames.append(per_run_df)

            # build summary row from the test set rollout metrics only; add train fit objective as mse_pool
            summary_row = dict(test_result['summary'])
            if raw_model_name == 'fj_adj':
                summary_row['train_mse_pool'] = float(FJ_ADJ_MSE)
            elif raw_model_name == 'fj_fg':
                summary_row['train_mse_pool'] = float(FJ_FG_MSE)
            summary_row.update({'raw_model': raw_model_name, 'model': MODEL_DISPLAY_NAMES.get(raw_model_name, raw_model_name)})
            summary_rows.append(summary_row)

        summary_df = pd.DataFrame(summary_rows)

        ranking_cols = [
            'model', 'n_runs',
            'run_max_wasserstein_mean', 'run_max_wasserstein_var',
            'run_integral_wasserstein_mean', 'run_integral_wasserstein_var',
            'run_max_mean_error_mean', 'run_max_mean_error_var',
            'run_integral_mean_error_mean', 'run_integral_mean_error_var',
            'run_max_variance_error_mean', 'run_max_variance_error_var',
            'run_integral_variance_error_mean', 'run_integral_variance_error_var',
            'train_mse_pool',
        ]
        for col in ranking_cols:
            if col not in summary_df.columns:
                summary_df[col] = np.nan
        combo_df = summary_df[ranking_cols].copy()
        combo_csv = combo_dir / f'{llm_name}__{topic_name}_model_rankings_fixed_graph_vs_adjacency.csv'
        combo_df.to_csv(combo_csv, index=False)

    except Exception as e:
        print(f'Error loading test run data: {str(e)[:120]}')
        raise e
    

