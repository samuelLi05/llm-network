"""
    Extra log parsing functions for analysis with varied network size.
"""

import shutil
from pathlib import Path
import sys
import random
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm



ROOT = Path(__file__).resolve().parents[1]
# ensure project imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from modeling.models.data_prep import _load_jsonl, _load_json, _numeric_agent_key
from modeling.stance_analysis_for_modeling.embedding_analyzer_sync import EmbeddingAnalyzerSync

POISSON_MEAN = 15.0
TIME_UNIT_MS = 8000
RATE_PER_S = (POISSON_MEAN / (TIME_UNIT_MS / 1000.0))  # (incorrect but kept for consistency with previous code)

def redefine_init_opinions(original_run_dir,
                           embedding_analyzer = None,
                           error_tolerance = 1e-03,     # tolerance for differences in recomputed init opinions
                           ):

    # Get the list of agents
    manifest = _load_json(original_run_dir / 'run_manifest.json')
    agent_ids = manifest['agent_ids']
    agent_ids = sorted(agent_ids, key=_numeric_agent_key)

    for i in range(len(agent_ids)):
        # check that agent_{i} is in agent_ids
        if f'agent_{i+1}' not in agent_ids:
            raise ValueError(f'agent_{i+1} not found in agent_ids: {agent_ids}')
    

    static_init_file = _load_json(original_run_dir / 'static_init.json')

    initial_stance_map = {}

    for a_id in agent_ids:

        # load the initial opinion logged in simulation

        # check that per_agent/{a_id}.jsonl exists
        per_agent_file = original_run_dir / 'per_agent' / f'{a_id}.jsonl'
        if not per_agent_file.exists():
            raise ValueError(f'per_agent file not found: {per_agent_file}')
        
        # Pull the first line from the jsonl file 
        per_agent_rows = _load_jsonl(per_agent_file)
        if len(per_agent_rows) == 0:
            init_opinion_saved = None
        else:
            first_row = per_agent_rows[0]
            init_opinion_saved = first_row["topology_profile_for_agent"]["ss"]



        # re-embed + score to check the initial approach
        baseline_opinion_str = static_init_file["agent_configs"][a_id]["stable_perspective_sentence"]
        scored = embedding_analyzer.embed_and_score(baseline_opinion_str)
        
        init_opinion_recomputed = scored['stance_score']

        if init_opinion_saved is not None:
            if abs(init_opinion_saved - init_opinion_recomputed)/init_opinion_saved > error_tolerance:
                raise ValueError(f'Initial opinion mismatch for {a_id}: logged={init_opinion_saved}, recomputed={init_opinion_recomputed}')

        initial_stance_map[a_id] = {
            "sim_logged": init_opinion_saved,
            "recomputed": init_opinion_recomputed,
            "baseline_opinion_str": baseline_opinion_str
        }

    return initial_stance_map



def clean_log_times_and_init(experiment_dir, out_dir, poisson_lambda, rng = None,
                             embedding_analyzer = None,
                             view_init_opinions_for_debug = False):

    # Correct the interarrival times and initial opinions in the log files for a given
    #  experiment directory. This is necessary because the logs may contain artifacts due to asyncio, which can affect the analysis of inter-post times.

    # assumption: experiment_dir is a Path object pointing to train/test directions
    #  which in turn contain runs

    experiment_dir = Path(experiment_dir)
    out_dir = Path(out_dir)


    # for debugging, track lists of logged and recomputed initial opinions. 
    #  should be very close to equal
    ss_sim_logged_list = []
    ss_recomputed_list = []

    if rng is None:
        rng = random.Random(1234)

    for split_dir in sorted(experiment_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        for run_dir in tqdm(sorted(split_dir.iterdir()), desc=f"Processing {experiment_dir.name}/{split_dir.name}"):
            if not run_dir.is_dir():
                continue


            # ensure out_dir / split_dir.name / run_dir.name exists
            original_run_dir = experiment_dir / split_dir.name / run_dir.name
            out_run_dir = out_dir / split_dir.name / run_dir.name
            out_run_dir.mkdir(parents=True, exist_ok=True)


            # STAGE 1: 
            # copy the connection_graph.json, run_manifest.json and static_init.json
            #  across to the new directory as is

            to_copy = ['connection_graph.json', 'run_manifest.json', 'static_init.json']

            for fn in to_copy:
                
                path_old = original_run_dir / fn
                path_new = out_run_dir / fn

                shutil.copy(path_old, path_new)

            # STAGE 2: 
            # parse the messages_with_alignment.jsonl file, correct the interarrival times and write a new file to the out_dir
            
            old_index = 0
            old_time_ms_in_logs = 0

            simulated_time_s = 0 
            
            new_rows = []
            for row in _load_jsonl(original_run_dir / 'messages_with_alignment.jsonl'):

                # VALIDATION: ensure
                #   - index increases by one
                #   - times are monotically increasing
                if not row['index'] == old_index + 1:
                    raise ValueError('Indices are not contiguous. Are these logs correct?')
                else:
                    old_index = row['index']
                if not row['time']['t_ms'] >= old_time_ms_in_logs:
                    raise ValueError('Time is not increasing in the loaded logs. Are these logs correct?')
                else:
                    old_time_ms_in_logs = row['time']['t_ms']
    

                # Copy across the old row to the new row with
                #  a newly sampled time stamp
                new_row = {}

                new_row['sender_id'] = row['sender_id']
                new_row['message'] = row['message']
                new_row['index'] = row['index']
                new_row['published'] = {
                    'stance_score' : row['published']['stance_score']
                }
                new_row['used_indices'] = row['used_indices']
                new_row['recommendation_indices'] = row['recommendation_indices']
                new_row['time'] = {
                    't_s' : simulated_time_s,
                    't_ms' : simulated_time_s * 1000
                }

                # increment according to given exponential parameter
                simulated_time_s += rng.expovariate(poisson_lambda)

                new_rows.append(new_row)
            
            # write the new rows to the out_dir
            with open(out_run_dir / 'messages_with_alignment.jsonl', 'w', encoding='utf-8') as f:
                for row in new_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')

            # STAGE 3: build a directory of initial opinions 
            #   using the stance analyzer
            #   (as a sanity check, we can compare these to the logged initial opinions in the per_agent jsonl files)
            initial_stance_map = redefine_init_opinions(original_run_dir, embedding_analyzer=embedding_analyzer)
            if split_dir.name == 'train':
                for a_id, stance_info in initial_stance_map.items():
                    ss_sim_logged_list.append(stance_info["sim_logged"])
                    ss_recomputed_list.append(stance_info["recomputed"])

            # save initial_stance_map to out_dir / split_dir.name / run_dir.name / initial_stance_map.json
            with open(out_run_dir / 'initial_stance_map.json', 'w', encoding='utf-8') as f:
                json.dump(initial_stance_map, f, ensure_ascii=False, indent=4)

    # plot the logged vs recomputed initial opinions
    if view_init_opinions_for_debug:
        plt.figure(figsize=(8, 6))
        plt.scatter(ss_sim_logged_list, ss_recomputed_list, alpha=0.5)
        plt.xlabel('Logged Initial Opinions (sim)')
        plt.ylabel('Recomputed Initial Opinions (stance analyzer)')
        plt.title('Comparison of Logged vs Recomputed Initial Opinions')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":

    # TEMP



    topic = "Vaccines cause austim"


    embedding_analyzer = EmbeddingAnalyzerSync(
        topic=topic
    )

    rng = random.Random(1234)
    clean_log_times_and_init(experiment_dir='modeling/runs_varied_size/llama3.1/vaccines/n_30', 
                             out_dir = 'modeling/runs_varied_size_corrected/llama3.1/vaccines/n_30',
                             poisson_lambda=RATE_PER_S,
                             rng = rng,
                             embedding_analyzer = embedding_analyzer,
                             view_init_opinions_for_debug = True)
    
    clean_log_times_and_init(experiment_dir='modeling/runs_varied_size/llama3.1/vaccines/n_60', 
                             out_dir = 'modeling/runs_varied_size_corrected/llama3.1/vaccines/n_60',
                             poisson_lambda=RATE_PER_S,
                             rng=rng,
                             embedding_analyzer = embedding_analyzer,
                             view_init_opinions_for_debug=True)
                             
    clean_log_times_and_init(experiment_dir='modeling/runs_varied_size/llama3.1/vaccines/n_100', 
                             out_dir = 'modeling/runs_varied_size_corrected/llama3.1/vaccines/n_100',
                             poisson_lambda=RATE_PER_S,
                             rng=rng,
                             embedding_analyzer = embedding_analyzer,
                             view_init_opinions_for_debug=True)