from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# ensure project imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import notebook-used helpers
from modeling.models.data_prep import load_run_data, build_run_trajectory, build_neighbors_index, _numeric_agent_key, build_row_normalized_adjacency  # type: ignore
from modeling.models.analysis_utils import (  # type: ignore
    plot_wasserstein_distance_per_timestep,  # not used, but keep for parity
)

# Import the same model-fitting functions as notebook
from modeling.models.adjacency_based.degroot import fit_degroot_adjacency_scalar, degroot_rollout_prediction  # type: ignore
from modeling.models.adjacency_based.friedkin_johnsen import (
    fit_base_friedkin_johnson_adjency,
    base_friedkin_johnsen_adjacency_rollout,
    select_base_friedkin_johnsen_adjacency_lambda,
    fit_friedkin_johnsen_adjacency,
    select_friedkin_johnsen_adjacency_lambdas,
    friedkin_johnsen_adjacency_rollout,
)  # type: ignore
from modeling.models.adjacency_based.homophily import (
    fit_homophily,
    rollout_with_homophily,
    fit_homophily_stubborness,
    rollout_with_homophily_stubborness,
    fit_homophily_friedkin_johnsen,
    rollout_with_homophily_friedkin_johnsen,
)  # type: ignore

# NOTE: this script mirrors the notebook's evaluate logic but is a standalone runner that writes per-combo CSVs.

MODEL_DISPLAY_NAMES = {
    'degroot_adjacency_scalar': 'degroot',
    'fj_adjacency_no_bias': 'friedkin_johnsen',
    'fj_adjacency': 'friedkin_johnsen_bias',
    'homophily': 'homophily',
    'homophily_friedkin_johnsen': 'homophily_friedkin_johnsen',
    'homophily_stubbornness': 'homophily_friedkin_johnsen_bias',
}

RANKING_METRIC_COLS = [
    'run_max_wasserstein',
    'run_integral_wasserstein',
    'run_max_mean_error',
    'run_integral_mean_error',
    'run_max_variance_error',
    'run_integral_variance_error',
]

PARAMS = {
    'target_agent_fraction': 0.4,
    'constrain_messages': 150,
    'rollout_horizon_cap': 20,
}


def stack_curves(curves):
    curves = [np.asarray(curve, dtype=float).ravel() for curve in curves if len(curve) > 0]
    if not curves:
        return np.empty((0, 0), dtype=float)
    common_t = min(curve.shape[0] for curve in curves)
    return np.stack([curve[:common_t] for curve in curves], axis=0)


def evaluate_model(model_name, run_traj_map, rollout_map):
    per_run_rows = []
    observed_curves = []
    predicted_curves = []
    mean_true_curves = []
    mean_pred_curves = []
    var_true_curves = []
    var_pred_curves = []
    wasserstein_curves = []

    for run_name in sorted(run_traj_map.keys()):
        observed = np.asarray(run_traj_map[run_name], dtype=float)
        predicted = np.asarray(rollout_map[run_name], dtype=float)
        t_common = min(observed.shape[0], predicted.shape[0])
        observed = observed[:t_common]
        predicted = predicted[:t_common]

        mean_true = np.mean(observed, axis=1)
        mean_pred = np.mean(predicted, axis=1)
        var_true = np.var(observed, axis=1)
        var_pred = np.var(predicted, axis=1)
        # fallback: if modeling/plot_utils exists, adjust import; otherwise compute stub
        try:
            from modeling.plot_utils import compute_wasserstein_distance_per_timestep
            wasserstein = compute_wasserstein_distance_per_timestep(observed, predicted)
        except Exception:
            wasserstein = np.abs(mean_pred - mean_true)  # fallback
        mean_error = np.abs(mean_pred - mean_true)
        var_error = np.abs(var_pred - var_true)

        per_run_rows.append({
            'model': model_name,
            'run_name': run_name,
            'run_length': int(t_common - 1),
            'run_max_wasserstein': float(np.max(wasserstein)) if wasserstein.size else np.nan,
            'run_integral_wasserstein': float(np.sum(wasserstein)) if wasserstein.size else np.nan,
            'run_max_mean_error': float(np.max(mean_error)) if mean_error.size else np.nan,
            'run_integral_mean_error': float(np.sum(mean_error)) if mean_error.size else np.nan,
            'run_max_variance_error': float(np.max(var_error)) if var_error.size else np.nan,
            'run_integral_variance_error': float(np.sum(var_error)) if var_error.size else np.nan,
        })

        observed_curves.append(observed)
        predicted_curves.append(predicted)
        mean_true_curves.append(mean_true)
        mean_pred_curves.append(mean_pred)
        var_true_curves.append(var_true)
        var_pred_curves.append(var_pred)
        wasserstein_curves.append(wasserstein)

    per_run = pd.DataFrame(per_run_rows)
    summary = {
        'model': model_name,
        'n_runs': int(per_run.shape[0]),
        'run_max_wasserstein_mean': float(per_run['run_max_wasserstein'].mean()),
        'run_max_wasserstein_var': float(per_run['run_max_wasserstein'].var(ddof=0)),
        'run_integral_wasserstein_mean': float(per_run['run_integral_wasserstein'].mean()),
        'run_integral_wasserstein_var': float(per_run['run_integral_wasserstein'].var(ddof=0)),
        'run_max_mean_error_mean': float(per_run['run_max_mean_error'].mean()),
        'run_max_mean_error_var': float(per_run['run_max_mean_error'].var(ddof=0)),
        'run_integral_mean_error_mean': float(per_run['run_integral_mean_error'].mean()),
        'run_integral_mean_error_var': float(per_run['run_integral_mean_error'].var(ddof=0)),
        'run_max_variance_error_mean': float(per_run['run_max_variance_error'].mean()),
        'run_max_variance_error_var': float(per_run['run_max_variance_error'].var(ddof=0)),
        'run_integral_variance_error_mean': float(per_run['run_integral_variance_error'].mean()),
        'run_integral_variance_error_var': float(per_run['run_integral_variance_error'].var(ddof=0)),
    }

    return {
        'per_run': per_run,
        'summary': summary,
        'observed_stack': stack_curves(observed_curves),
        'predicted_stack': stack_curves(predicted_curves),
        'mean_true_stack': stack_curves(mean_true_curves),
        'mean_pred_stack': stack_curves(mean_pred_curves),
        'var_true_stack': stack_curves(var_true_curves),
        'var_pred_stack': stack_curves(var_pred_curves),
        'wasserstein_stack': stack_curves(wasserstein_curves),
    }


if __name__ == '__main__':
    RUNS_DIR = ROOT / 'modeling' / 'runs'
    ALL_LLMS = sorted([d.name for d in RUNS_DIR.iterdir() if d.is_dir()])
    print(f'Discovered LLMs: {ALL_LLMS}.')
    combo_dir = ROOT / 'llm_topic_model_rankings'
    combo_dir.mkdir(parents=True, exist_ok=True)

    for llm_name in ALL_LLMS:
        llm_path = RUNS_DIR / llm_name
        topics = sorted([d.name for d in llm_path.iterdir() if d.is_dir()])
        for topic_name in topics:
            train_path = llm_path / topic_name / 'train'
            test_path = llm_path / topic_name / 'test'
            if not train_path.exists() or not test_path.exists():
                continue
            run_dirs = sorted([p for p in train_path.iterdir() if p.is_dir()])
            print(f'\n[{llm_name}/{topic_name}] {len(run_dirs)} runs')
            try:
                run_data = {r.name: load_run_data(r) for r in run_dirs}
                global_agents = sorted({a for d in run_data.values() for a in d['agent_ids']}, key=_numeric_agent_key)
                n_agents = len(global_agents)
                traj_mask = {rn: build_run_trajectory(d, global_agents, target_agent_fraction=PARAMS['target_agent_fraction'], return_post_mask=True, constrain_messages=PARAMS['constrain_messages']) for rn, d in run_data.items()}
                run_traj = {rn: tm[0] for rn, tm in traj_mask.items()}
                run_neighbors = {rn: build_neighbors_index(d, global_agents) for rn, d in run_data.items()}
            except Exception as e:
                print(f'  Fit failed: {str(e)[:120]}')
                # still continue to next topic

            # Build test rollouts using fitted params (best-effort)
            try:
                test_run_dirs = sorted([p for p in test_path.iterdir() if p.is_dir()])
                test_run_data = {r.name: load_run_data(r) for r in test_run_dirs}
                test_traj = {run_name: build_run_trajectory(data, global_agents, target_agent_fraction=PARAMS['target_agent_fraction'], return_post_mask=False, constrain_messages=PARAMS['constrain_messages']) for run_name, data in test_run_data.items()}
                test_neighbors = {run_name: build_neighbors_index(data, global_agents) for run_name, data in test_run_data.items()}

                # Fit all adjacency-based models on pooled training data (to use for test rollouts)
                #LAMBDA_GRID = np.linspace(0.0, 1.0, 50)
                DEGROOT_ADJ = fit_degroot_adjacency_scalar(run_traj, run_neighbors)
                DEGROOT_GAMMA = DEGROOT_ADJ.get('gamma', np.nan)

                BEST_BASE_FJ_ADJ, _ = select_base_friedkin_johnsen_adjacency_lambda(run_traj, run_neighbors)
                BASE_FJ_ADJ_L1 = BEST_BASE_FJ_ADJ['lambda1']
                BASE_FJ_ADJ_FIT = fit_base_friedkin_johnson_adjency(run_traj, run_neighbors, BASE_FJ_ADJ_L1)
                BASE_FJ_ADJ_GAMMA = BASE_FJ_ADJ_FIT.get('gamma', np.nan)

                BEST_FJ_ADJ, _ = select_friedkin_johnsen_adjacency_lambdas(run_traj, run_neighbors)
                FJ_ADJ_L1 = BEST_FJ_ADJ['lambda1']
                FJ_ADJ_L2 = BEST_FJ_ADJ['lambda2']
                FJ_ADJ_FIT = fit_friedkin_johnsen_adjacency(run_traj, run_neighbors, FJ_ADJ_L1, FJ_ADJ_L2)
                FJ_ADJ_GAMMA = FJ_ADJ_FIT.get('gamma', np.nan)
                FJ_ADJ_BIAS = FJ_ADJ_FIT.get('bias', np.nan)

                HOMOPHILY_FIT = fit_homophily(run_traj, run_neighbors, gamma0=1.0)
                HOMOPHILY_GAMMA = HOMOPHILY_FIT.get('gamma', np.nan)
                HOMOPHILY_LAMBDA = HOMOPHILY_FIT.get('lambda', np.nan)

                BEST_HOMO_FJ = fit_homophily_friedkin_johnsen(run_traj, run_neighbors, gamma0=HOMOPHILY_GAMMA)
                BEST_HOMO_STUB = fit_homophily_stubborness(run_traj, run_neighbors, gamma0=HOMOPHILY_GAMMA)
                HOMO_FJ_GAMMA = BEST_HOMO_FJ.get('gamma', np.nan)
                HOMO_FJ_L1 = BEST_HOMO_FJ.get('lambda1', np.nan)
                HOMO_FJ_LSELF = BEST_HOMO_FJ.get('lambda_self', np.nan)
                HOMO_STUB_GAMMA = BEST_HOMO_STUB.get('gamma', np.nan)
                HOMO_STUB_LSELF = BEST_HOMO_STUB.get('lambda_self', np.nan)
                HOMO_STUB_L1 = BEST_HOMO_STUB.get('lambda1', np.nan)
                HOMO_STUB_L2 = BEST_HOMO_STUB.get('lambda2', np.nan)

                def build_rollout_maps(traj_map, neighbors_map):
                    return {
                    'degroot_adjacency_scalar': {
                        run_name: degroot_rollout_prediction(
                            DEGROOT_GAMMA * build_row_normalized_adjacency(neighbors_map[run_name], n_agents)
                            + (1.0 - DEGROOT_GAMMA) * np.eye(n_agents, dtype=float),
                            np.asarray(traj_map[run_name], dtype=float)[0],
                            PARAMS['rollout_horizon_cap'],
                        )
                        for run_name in traj_map.keys()
                    },
                    'fj_adjacency_no_bias': {
                        run_name: base_friedkin_johnsen_adjacency_rollout(
                            BASE_FJ_ADJ_FIT.get('gamma', np.nan) * build_row_normalized_adjacency(neighbors_map[run_name], n_agents)
                            + (1.0 - BASE_FJ_ADJ_FIT.get('gamma', np.nan)) * np.eye(n_agents, dtype=float),
                            np.asarray(traj_map[run_name], dtype=float)[0],
                            PARAMS['rollout_horizon_cap'],
                            BASE_FJ_ADJ_L1,
                        )
                        for run_name in traj_map.keys()
                    },
                    'fj_adjacency': {
                        run_name: friedkin_johnsen_adjacency_rollout(
                            FJ_ADJ_FIT.get('gamma', np.nan) * build_row_normalized_adjacency(neighbors_map[run_name], n_agents)
                            + (1.0 - FJ_ADJ_FIT.get('gamma', np.nan)) * np.eye(n_agents, dtype=float),
                            FJ_ADJ_FIT.get('bias', np.nan),
                            np.asarray(traj_map[run_name], dtype=float)[0],
                            PARAMS['rollout_horizon_cap'],
                            FJ_ADJ_L1,
                            FJ_ADJ_L2,
                        )
                        for run_name in traj_map.keys()
                    },
                    'homophily': {
                        run_name: rollout_with_homophily(
                            build_row_normalized_adjacency(neighbors_map[run_name], n_agents),
                            HOMOPHILY_GAMMA,
                            np.asarray(traj_map[run_name], dtype=float)[0],
                            PARAMS['rollout_horizon_cap'],
                            lambda_self=HOMOPHILY_LAMBDA,
                        )
                        for run_name in traj_map.keys()
                    },
                    'homophily_friedkin_johnsen': {
                        run_name: rollout_with_homophily_friedkin_johnsen(
                            build_row_normalized_adjacency(neighbors_map[run_name], n_agents),
                            HOMO_FJ_GAMMA,
                            HOMO_FJ_L1,
                            np.asarray(traj_map[run_name], dtype=float)[0],
                            PARAMS['rollout_horizon_cap'],
                            lambda_self=HOMO_FJ_LSELF,
                        )
                        for run_name in traj_map.keys()
                    },
                    'homophily_stubbornness': {
                        run_name: rollout_with_homophily_stubborness(
                            build_row_normalized_adjacency(neighbors_map[run_name], n_agents),
                            HOMO_STUB_GAMMA,
                            BEST_HOMO_STUB.get('bias', np.nan),
                            HOMO_STUB_L1,
                            HOMO_STUB_L2,
                            np.asarray(traj_map[run_name], dtype=float)[0],
                            PARAMS['rollout_horizon_cap'],
                            lambda_self=HOMO_STUB_LSELF,
                        )
                        for run_name in traj_map.keys()
                    },
                }

                rollout_maps_test = build_rollout_maps(test_traj, test_neighbors)
                rollout_maps_train = build_rollout_maps(run_traj, run_neighbors)

                # Evaluate models on both train and test
                per_run_frames = []
                summary_rows = []
                for raw_model_name in rollout_maps_test.keys():
                    train_result = evaluate_model(raw_model_name, run_traj, rollout_maps_train[raw_model_name])
                    test_result = evaluate_model(raw_model_name, test_traj, rollout_maps_test[raw_model_name])

                    result = test_result
                    per_run_df = result['per_run'].copy()
                    per_run_df['llm'] = llm_name
                    per_run_df['topic'] = topic_name
                    per_run_df['raw_model'] = raw_model_name
                    per_run_df['model'] = MODEL_DISPLAY_NAMES.get(raw_model_name, raw_model_name)
                    per_run_frames.append(per_run_df)

                    # build summary row from the test set rollout metrics only; add train fit objective as mse_pool
                    summary_row = dict(test_result['summary'])
                    if raw_model_name == 'degroot_adjacency_scalar':
                        summary_row['train_mse_pool'] = float(DEGROOT_ADJ['mse_pool'])
                    elif raw_model_name == 'fj_adjacency_no_bias':
                        summary_row['train_mse_pool'] = float(BASE_FJ_ADJ_FIT['mse_pool'])
                    elif raw_model_name == 'fj_adjacency':
                        summary_row['train_mse_pool'] = float(FJ_ADJ_FIT['mse_pool'])
                    elif raw_model_name == 'homophily':
                        summary_row['train_mse_pool'] = float(HOMOPHILY_FIT['mse_pool'])
                    elif raw_model_name == 'homophily_friedkin_johnsen':
                        summary_row['train_mse_pool'] = float(BEST_HOMO_FJ['mse_pool'])
                    elif raw_model_name == 'homophily_stubbornness':
                        summary_row['train_mse_pool'] = float(BEST_HOMO_STUB['mse_pool'])
                    summary_row.update({'llm': llm_name, 'topic': topic_name, 'raw_model': raw_model_name, 'model': MODEL_DISPLAY_NAMES.get(raw_model_name, raw_model_name)})
                    summary_rows.append(summary_row)

                summary_df = pd.DataFrame(summary_rows)
                # ranking and concise export
                summary_df['rank_run_integral_wasserstein'] = summary_df.get('run_integral_wasserstein_mean', pd.Series(np.nan)).rank(method='dense', ascending=True)
                summary_df = summary_df.sort_values(['rank_run_integral_wasserstein', 'model']).reset_index(drop=True)

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
                combo_csv = combo_dir / f'{llm_name}__{topic_name}_model_rankings.csv'
                combo_df.to_csv(combo_csv, index=False)
                print(f'  Saved per-combo table to: {combo_csv}')

            except Exception as e:
                print(f'  Evaluation failed: {str(e)[:200]}')
                continue
