from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse

ROOT = Path(__file__).resolve().parents[1]
# ensure project imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.models.data_prep import _numeric_agent_key, build_dataset_from_run, build_expected_message_matrix, build_neighbors_index, _make_homophily_step
from modeling.models.data_prep_network_size import load_cleaned_run_data, build_run_trajectory_from_clean

PARAMS = {
    'target_agent_fraction': 0.4,
    'constrain_messages': 150,
    'rollout_horizon_cap': 20,
}

OD_MODEL_LIST = [
    'degroot',
    'friedkin_johnsen',
    'friedkin_johnsen_bias',
    'homophily',
    'homophily_friedkin_johnsen',
    'homophily_friedkin_johnsen_bias',
    'bias_only',
    'bias_init_only'
]

# Define which parameters to compute Hessians for each model
HESSIANS_TO_GET = {
    'degroot': ['lambda_soc'],
    'friedkin_johnsen': ['lambda_soc', 'lambda_init'],
    'friedkin_johnsen_bias': ['lambda_soc', 'lambda_init', 'lambda_bias', 'bias'],
    'homophily': ['lambda_soc'],
    'homophily_friedkin_johnsen': ['lambda_soc', 'lambda_init'],
    'homophily_friedkin_johnsen_bias': ['lambda_soc', 'lambda_init', 'lambda_bias', 'bias'],
    'bias_only': ['lambda_bias', 'bias'],
    'bias_init_only': ['lambda_init', 'lambda_bias', 'bias']
}

PARAMS_TO_PLOT = ['lambda_soc', 'lambda_init', 'lambda_bias', 'bias']

# Constants for checking
BOUNDARY_TOLERANCE = 2e-03 # Tolerance for checking if a parameter is at the boundary of its valid range

def evaluate_mse(run_traj_map, run_neighbors,
                 lambda_self = 0.0, 
                 lambda_soc = 0.0,
                 lambda_init = 0.0,
                 lambda_bias = 0.0,
                 bias = 0.0,
                 gamma = 0.0,
                 return_as_np_array = False):

    # utility to help evaluate the mse of a given set of parameters on a set of runs
    if lambda_self < 0 or lambda_soc < 0 or lambda_init < 0 or lambda_bias < 0:
        raise ValueError("Lambda parameters must be non-negative.")
    if not np.isclose(lambda_self + lambda_soc + lambda_init + lambda_bias, 1.0):
        raise ValueError("Lambda parameters must sum to one.")
    if bias < -1 or bias > 1:
        raise ValueError("Bias must be in the range [-1, 1].")
    
    run_names = sorted(run_traj_map.keys())
    if not run_names:
        raise ValueError("No runs provided for evaluation.")
    
    # build up all the big matrices for the runs
    x_blocks, y_blocks, x0_blocks = [], [], []
    for run_name in run_names:
        traj = np.asarray(run_traj_map[run_name], dtype=float)
        x,y = build_dataset_from_run(traj)
        x_blocks.append(x)
        y_blocks.append(y)
        x0_blocks.append(np.repeat(traj[0].reshape(1, -1), x.shape[0], axis=0))

    x_pool = np.vstack(x_blocks)
    y_pool = np.vstack(y_blocks)
    x0_pool = np.vstack(x0_blocks)

    n = x_pool.shape[1]
    abar_blocks = [build_expected_message_matrix(run_neighbors.get(rn, {}), n) for rn in run_names]
    homophily_steps = [_make_homophily_step(A) for A in abar_blocks]

    homop_blocks = [
        np.asarray([homophily_steps[i](x_blocks[i][t], gamma) for t in range(x_blocks[i].shape[0])], dtype=float)
        for i in range(len(x_blocks))
    ]
    homop_pool = np.vstack(homop_blocks)

    fitted_pool = (lambda_self * x_pool) + \
                    (lambda_soc * homop_pool) + \
                    (lambda_init * x0_pool) + \
                    (lambda_bias * bias)

    mse_pool = float(np.mean((y_pool - fitted_pool) ** 2))

    if return_as_np_array:
        return np.array([mse_pool])
    else:
        return mse_pool

def get_perturbation_function(run_traj, run_neighbors, param_name, base_params):
    # Utility function to get a 1D function of the MSE with respect to a single parameter,
    #  while adjusting the other parameters proportionally to maintain the sum-to-one constraint.

    # for all results we have, the self weights are positive, 
    #   so we assume that base_params['lambda_self'] is in (0,1). 
    # Check this assumption isn't violated
    if base_params['lambda_self'] <= BOUNDARY_TOLERANCE or base_params['lambda_self'] >= 1 - BOUNDARY_TOLERANCE:
        raise ValueError("Base parameter 'lambda_self' is at or near the boundary of its valid range. Sensitivity analysis may not be valid.")

    if param_name == 'lambda_self':
        raise NotImplementedError("Sensitivity analysis for 'lambda_self' is not implemented. \
                                  We instead examine sensitivity with respect to other params.")
    elif param_name == 'lambda_soc':
        mse_func_scalar = lambda r: evaluate_mse(run_traj, run_neighbors,
                                            lambda_self = base_params['lambda_self'] - r,
                                            lambda_soc = base_params['lambda_soc'] + r,
                                            lambda_init = base_params['lambda_init'] ,
                                            lambda_bias = base_params['lambda_bias'] ,
                                            bias = base_params['bias'],
                                            gamma = base_params['gamma'],
                                            return_as_np_array = False)
    elif param_name == 'lambda_init':
        mse_func_scalar = lambda r: evaluate_mse(run_traj, run_neighbors,
                                            lambda_self = base_params['lambda_self'] - r,
                                            lambda_soc = base_params['lambda_soc'],
                                            lambda_init = base_params['lambda_init'] + r,
                                            lambda_bias = base_params['lambda_bias'],
                                            bias = base_params['bias'],
                                            gamma = base_params['gamma'],
                                            return_as_np_array = False)
    elif param_name == 'lambda_bias': 
        mse_func_scalar = lambda r: evaluate_mse(run_traj, run_neighbors,
                                            lambda_self = base_params['lambda_self'] - r,
                                            lambda_soc = base_params['lambda_soc'],
                                            lambda_init = base_params['lambda_init'],
                                            lambda_bias = base_params['lambda_bias'] + r,
                                            bias = base_params['bias'],
                                            gamma = base_params['gamma'],
                                            return_as_np_array = False)
    elif param_name == 'bias':
        mse_func_scalar = lambda r: evaluate_mse(run_traj, run_neighbors,
                                            lambda_self = base_params['lambda_self'],
                                            lambda_soc = base_params['lambda_soc'],
                                            lambda_init = base_params['lambda_init'],
                                            lambda_bias = base_params['lambda_bias'],
                                            bias = base_params['bias'] + r,
                                            gamma = base_params['gamma'],
                                            return_as_np_array = False)
    elif param_name == 'gamma':
        mse_func_scalar = lambda r: evaluate_mse(run_traj, run_neighbors,
                                            lambda_self = base_params['lambda_self'],
                                            lambda_soc = base_params['lambda_soc'],
                                            lambda_init = base_params['lambda_init'],
                                            lambda_bias = base_params['lambda_bias'],
                                            bias = base_params['bias'],
                                            gamma = base_params['gamma'] + r,
                                            return_as_np_array = False)
    else:
        raise ValueError(f"Unknown parameter name: {param_name}")
    
    # return mse_func_scalar applied element wise to a numpy array, so that we can use it with np.vectorize or similar
    return np.vectorize(mse_func_scalar)
                                        

def validate_mses_against_stored_data(llm_name,
                                      topic_name,
                                      model_rankings_dir,
                                      fitted_param_dict):
    model_rankings_fn = f"{llm_name}__{topic_name}_model_rankings.csv"
    model_rankings_path = model_rankings_dir / model_rankings_fn

    model_rankings_data = []
    if not model_rankings_path.exists():
        raise FileNotFoundError(f"Model rankings file not found: {model_rankings_path}")
    else:
        with open(model_rankings_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_rankings_data.append(row)

    covered = np.zeros(len(OD_MODEL_LIST), dtype=bool)
    for model_row in model_rankings_data:
        model_name = model_row['model']
        if not model_name in OD_MODEL_LIST:
            raise ValueError(f"Model {model_name} not in expected model list.")
        else:
            covered[OD_MODEL_LIST.index(model_name)] = True
        stored_mse = float(model_row['train_mse_pool'])

        if not model_name in fitted_param_dict:
            raise ValueError(f"Fitted parameters for model {model_name} not found in fitted params data.")
        
        mse_pool_recomputed = evaluate_mse(run_traj, run_neighbors,
                                            lambda_self = fitted_param_dict[model_name]['lambda_self'],
                                            lambda_soc = fitted_param_dict[model_name]['lambda_soc'],
                                            lambda_init = fitted_param_dict[model_name]['lambda_init'],
                                            lambda_bias = fitted_param_dict[model_name]['lambda_bias'],
                                            bias = fitted_param_dict[model_name]['bias'],
                                            gamma = fitted_param_dict[model_name]['gamma'])

        if not np.isclose(stored_mse, mse_pool_recomputed, atol=1e-6):
            print(f"Mismatch in MSE for model {model_name} on {llm_name}/{topic_name}:")
            print(f"  Stored MSE: {stored_mse}")
            print(f"  Recomputed MSE: {mse_pool_recomputed}")
            raise ValueError(f"MSE mismatch for model {model_name} on {llm_name}/{topic_name}.")


    if not np.all(covered):
        missing_models = [OD_MODEL_LIST[i] for i, c in enumerate(covered) if not c]
        raise ValueError(f"Some models in OD_MODEL_LIST are not covered in the model rankings data: {missing_models}")


def get_lambda_from_csv_dict(csv_dict, lambda_key):
    if not lambda_key in csv_dict:
        raise KeyError(f"Key '{lambda_key}' not found in CSV dictionary.")
    lambda_str = csv_dict.get(lambda_key, None)
    if lambda_str is None:
        raise ValueError(f"Value not found for key '{lambda_key}' in CSV dictionary.")
    elif lambda_str == '':
        return 0.0
    else:
        return float(lambda_str)

def num_hessian(f,
                step_sizes = np.logspace(-3, -5, num=6), # define a set of step sizes to try for the derivative calculation
                pct_tol = 1.0, # percent tolerance for checking that the last 3 hessian estimates agree
                debug = False):
    # quick implementation of central differencing about 0.0
    #  (jax/scipy weren't playing nice)
    
    hess_values = []
    for s in step_sizes:
        num_hess = f(s) - 2 * f(0.0) + f(-s)
        denom_hess = s ** 2
        hess = num_hess / denom_hess
        hess_values.append(hess)
    
    # check that the last 3 values agree up to 1%
    if len(hess_values) >= 3:
        last_three = hess_values[-3:]
        if abs(last_three[0]) < 1e-5:
            if not np.all(np.abs(last_three) < 1e-5):
                raise ValueError(f"Hessian estimates are not all near zero for the last 3 step sizes. Values: {last_three}")
        else:
            pct_diff = abs(np.array(last_three) - last_three[0])/np.maximum(np.abs(last_three[0]), 1e-8) * 100
            if not np.all(pct_diff < pct_tol):
                raise ValueError(f"Hessian estimates  do not agree within {pct_tol}% for the last 3 step sizes. Values: {last_three}, Percent differences: {pct_diff}")

    if debug:
        print(f"Hessian estimates for step sizes {step_sizes}: {hess_values}")

    return hess_values[-1]  # return the last value as the estimate of the Hessian

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate sensitivity of mse to parameters')
    parser.add_argument('--reverse-graph', action='store_true', help='If set, interpret the graph edges in reverse direction when building neighbor indices.')
    parser.add_argument('--use-reembedded', action='store_true', help='If set, use re-embedded stance scores for evaluation (requires rescored runs).')
    parser.add_argument('--plot', action='store_true', help='If set, generate plots of the sensitivity analysis results.')
    parser.add_argument('--debug', action='store_true', help='If set, enable debug mode with more verbose output.')
    args = parser.parse_args()

    if args.use_reembedded:
        RUNS_DIR = ROOT / 'modeling' / 'runs_rescored'
    else:
        raise ValueError("This script should really be run with --use_reembedded set. Please run with --use_reembedded.")
    
    if not args.reverse_graph:
        raise ValueError("This script should really be run with --reverse-graph set. Please run with --reverse-graph.")

    REVERSE_GRAPH = args.reverse_graph
    print("Reverse graph mode:", REVERSE_GRAPH)

    ALL_LLMS = sorted([d.name for d in RUNS_DIR.iterdir() if d.is_dir()])

    MODEL_RANKINGS_DIR = ROOT / 'llm_topic_model_rankings'
    FITTED_PARAMS_DIR = ROOT / 'llm_topic_model_rankings' / 'fitted_params'

    OUTPUT_DIR = ROOT / 'verification' / 'param_sensitivity_results'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for llm_name in ALL_LLMS:
        llm_path = RUNS_DIR  / llm_name
        topics = sorted([d.name for d in llm_path.iterdir() if d.is_dir()])

        for topic_name in topics:

            # Load the training data for the runs of this LLM and topic
            train_path = llm_path / topic_name / 'train'
            if not train_path.exists():
                continue

            run_dirs = sorted([p for p in train_path.iterdir() if p.is_dir()])
            print(f'\n[{llm_name}/{topic_name}] {len(run_dirs)} runs')
            try:
                if args.use_reembedded:
                    run_data = {r.name: load_cleaned_run_data(r) for r in run_dirs}
                    global_agents = sorted({a for d in run_data.values() for a in d['agent_ids']}, key=_numeric_agent_key)
                    n_agents = len(global_agents)
                    traj_mask = {rn: build_run_trajectory_from_clean(d, 
                                                                     global_agents, 
                                                                     target_agent_fraction=PARAMS['target_agent_fraction'], 
                                                                     return_post_mask=True, 
                                                                     constrain_messages=PARAMS['constrain_messages'], 
                                                                     reference_agent_number = n_agents) 
                                    for rn, d in run_data.items()}
                else:
                    raise NotImplementedError("This script should really be run with --use_reembedded set. Please run with --use_reembedded.")
                run_traj = {rn: tm[0] for rn, tm in traj_mask.items()}
                run_neighbors = {rn: build_neighbors_index(d, global_agents, reverse=REVERSE_GRAPH) for rn, d in run_data.items()}
            
            except Exception as e:
                print(f' Error loading run data: {str(e)[:120]}')
                raise e
            
            # Load the fitted params
            fitted_params_fn = f"{llm_name}__{topic_name}_fitted_params.csv"
            fitted_params_path = FITTED_PARAMS_DIR / fitted_params_fn

            fitted_params_data = []
            if not fitted_params_path.exists():
                raise FileNotFoundError(f"Fitted params file not found: {fitted_params_path}")
            else:
                with open(fitted_params_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        fitted_params_data.append(row)

            fitted_param_dict = {}

            # Extract and parse all fitted params
            for model_row in fitted_params_data:
                model_name = model_row['model']
                if not model_name in OD_MODEL_LIST:
                    raise ValueError(f"Model {model_name} not in expected model list.")
                
                # extract fitted params
                lambda_self = get_lambda_from_csv_dict(model_row, 'self_weight')
                lambda_soc = get_lambda_from_csv_dict(model_row, 'social_weight')
                lambda_init = get_lambda_from_csv_dict(model_row, 'init_weight')
                lambda_bias = get_lambda_from_csv_dict(model_row, 'bias_weight')
                bias = get_lambda_from_csv_dict(model_row, 'bias')
                gamma = get_lambda_from_csv_dict(model_row, 'gamma')

                fitted_param_dict[model_name] = {
                    'lambda_self': lambda_self,
                    'lambda_soc': lambda_soc,
                    'lambda_init': lambda_init,
                    'lambda_bias': lambda_bias,
                    'bias': bias,
                    'gamma': gamma
                }

            # Validation: open up the model rankings, 
            #   and validate that the stored mse values match the mse values computed from the fitted parameters
            validate_mses_against_stored_data(llm_name,
                                              topic_name,
                                              MODEL_RANKINGS_DIR,
                                              fitted_param_dict)
            
            # Sensitivity analysis: for each model evaluate the (diagonal)Hessian of the MSE with respect to the parameters at the fitted point
            #  In particular, for each lambda, we'll perturb the weight, and change the self weight to compensate, and 
            #  evaluate the MSE of the induced 1d function

            # Additionally, in the case where we're plotting, generate plots of the MSE as a function
            #  of the parameter perturbation. 
            #  We should plot these on a |fitted_param_dict| * 5 grid, with each row corresponding to a model, 
            #   and each column corresponding to a parameter (lambda_soc, lambda_init, lambda_bias, bias, gamma).
            if args.plot:
                num_models = len(fitted_param_dict)
                num_params = len(PARAMS_TO_PLOT)
                fig, axes = plt.subplots(num_models, num_params, figsize=(4 * num_params, 3 * num_models))
                if num_models == 1:
                    axes = np.expand_dims(axes, axis=0)  # Ensure axes is 2D for consistency


            hessian_results_llm_topic_norm = {}

            for model_name, params in fitted_param_dict.items():
                print(f"Evaluating sensitivity for model {model_name} on {llm_name}/{topic_name}...")
                hessian_results = {}

                for param_name in PARAMS_TO_PLOT:
                    hessian_results[param_name] = '-'
                    hessian_results[param_name + '_unnormalized'] = '-'

                for param_name in HESSIANS_TO_GET[model_name]:
                    if not param_name in PARAMS_TO_PLOT:
                        raise ValueError(f"Parameter {param_name} for model {model_name} is not in the list of parameters to plot.")

                    # get a 1d function based on lambda
                    mse_func = get_perturbation_function(run_traj, run_neighbors, param_name, params)

                    # If plotting enabled, plot the MSE as a function of param. perturbation
                    if args.plot:
                        perturbation_values = np.linspace(-0.05, 0.05, 11) 
                        # truncate perturbation values to ensure they don't push parameters out of bounds
                        valid_perturbations = []
                        for p in perturbation_values:
                            if param_name.startswith('lambda_'):
                                new_value = params[param_name] + p
                                if new_value < 0 or new_value > 1:
                                    continue
                            elif param_name == 'bias':
                                new_value = params[param_name] + p
                                if new_value < -1 or new_value > 1:
                                    continue
                            elif param_name == 'gamma':
                                new_value = params[param_name] + p
                                if new_value < 0:
                                    continue
                            else:
                                raise ValueError(f"Unknown parameter name: {param_name}")
                            valid_perturbations.append(p)

                        # plot valid_perturbation 
                        row_index = OD_MODEL_LIST.index(model_name)
                        col_index = PARAMS_TO_PLOT.index(param_name)
                        ax = axes[row_index, col_index]
                        perturbation_mse_values = [mse_func(p) for p in valid_perturbations]
                        ax.plot(valid_perturbations, perturbation_mse_values, marker='o')
                        ax.set_title(f"{model_name} - {param_name}")
                        ax.set_xlabel("Perturbation")
                        ax.set_ylabel("MSE")

                    # Take the last value as the estimate of the Hessian
                    base_mse = evaluate_mse(run_traj, run_neighbors,
                                            lambda_self = fitted_param_dict[model_name]['lambda_self'],
                                            lambda_soc = fitted_param_dict[model_name]['lambda_soc'],
                                            lambda_init = fitted_param_dict[model_name]['lambda_init'],
                                            lambda_bias = fitted_param_dict[model_name]['lambda_bias'],
                                            bias = fitted_param_dict[model_name]['bias'],
                                            gamma = fitted_param_dict[model_name]['gamma'])
                    
                    hessian_results['base_mse'] = base_mse

                    # Compute the actual Hessian
                    param_value = params[param_name]
                    # check if parameter lies at the boundary; if so the Hessian does not provide meaningful information, so we set it to NaN
                    if param_name.startswith('lambda_'):
                        if param_value < BOUNDARY_TOLERANCE or param_value > 1 - BOUNDARY_TOLERANCE:
                           hessian_results[param_name] = np.nan
                           hessian_results[param_name + '_unnormalized'] = np.nan
                           continue
                    elif param_name == 'bias':
                        if param_value < -1 + BOUNDARY_TOLERANCE or param_value > 1 - BOUNDARY_TOLERANCE:
                            hessian_results[param_name] = np.nan
                            hessian_results[param_name + '_unnormalized'] = np.nan
                            continue
                    elif param_name == 'gamma':
                        if param_value < BOUNDARY_TOLERANCE:
                            hessian_results[param_name] = np.nan
                            hessian_results[param_name + '_unnormalized'] = np.nan
                            continue
                    else:
                        raise ValueError(f"Unknown parameter name: {param_name}")

                    hess = float(num_hessian(mse_func, debug=args.debug))
                    hessian_results[param_name + '_unnormalized'] = hess
                    hessian_results[param_name] = hess/base_mse
                    
                hessian_results_llm_topic_norm[model_name] = hessian_results
                if args.debug:
                    print(f"Hessian results for {llm_name}/{topic_name} - {model_name}: {hessian_results}")
                
            
            PARAMS_TO_PLOT_UNNORM = [param_name + '_unnormalized' for param_name in PARAMS_TO_PLOT]

            # Save the Hessian results to a CSV file
            hessian_output_fn = f"{llm_name}__{topic_name}_hessian_results.csv"
            hessian_output_path = OUTPUT_DIR / hessian_output_fn
            with open(hessian_output_path, 'w', newline='') as f:
                fieldnames = ['model'] + PARAMS_TO_PLOT + ['base_mse'] + PARAMS_TO_PLOT_UNNORM
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for model_name, hessians in hessian_results_llm_topic_norm.items():
                    row = {'model': model_name}
                    row.update(hessians)
                    writer.writerow(row)
            
            if args.plot:
                plt.show()



