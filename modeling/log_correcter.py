"""
    Extra log parsing functions for analysis with varied network size.
"""

import shutil
from pathlib import Path
import sys
import random
import numpy as np
import json


ROOT = Path(__file__).resolve().parents[1]
# ensure project imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from modeling.models.data_prep import _load_jsonl

POISSON_MEAN = 15.0
TIME_UNIT_MS = 8000
RATE_PER_S = (POISSON_MEAN / (TIME_UNIT_MS / 1000.0))  # (incorrect but kept for consistency with previous code)


def clean_log_times_and_init(experiment_dir, out_dir, poisson_lambda, interpost_seed = 1234):

    # Correct the interarrival times and initial opinions in the log files for a given
    #  experiment directory. This is necessary because the logs may contain artifacts due to asyncio, which can affect the analysis of inter-post times.

    # assumption: experiment_dir is a Path object pointing to train/test directions
    #  which in turn contain runs

    experiment_dir = Path(experiment_dir)
    out_dir = Path(out_dir)

    rng = random.Random(interpost_seed)

    for split_dir in sorted(experiment_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        for run_dir in sorted(split_dir.iterdir()):
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
                # breakpoint()

                # VALIDATION: ensure
                #   - index increases by one
                #   - 
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

if __name__ == "__main__":

    # TEMP

    clean_log_times_and_init(experiment_dir='modeling/runs_varied_size/llama3.1/vaccines/n_30', 
                             out_dir = 'modeling/runs_varied_size_corrected/llama3.1/vaccines/n_30',
                             poisson_lambda=RATE_PER_S)