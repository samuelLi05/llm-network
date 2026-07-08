"""
    A script to sample messages, along with their stance scores, for judging, 
    so that we can compare the assigned stance scores to human and LLM-as-judge rankings
"""

from pathlib import Path
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import json
import csv


ROOT = Path(__file__).resolve().parents[1]
# ensure project imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.models.data_prep import _load_jsonl

# Fix the LLM and topic to use
LLM = "gemma3"
TOPIC = "climate"
RUNS_DIR = ROOT / 'modeling' / 'runs'
MESSAGE_WRITE_DIR = ROOT / 'verification' / 'sampled_messages'

def check_are_valid_run_dirs(base_dir, run_dir_names):
    # function for validating that runs contain messages_with_alignment.jsonl files

    for dir_name in run_dir_names:

        msg_file = base_dir / dir_name / 'messages_with_alignment.jsonl'

        # assert that the file exists
        if not msg_file.exists():
            raise ValueError(f"The path at {msg_file} does not exist.")
    

def get_message_num_statistics(train_runs_dir):

    run_dirs = [d.name for d in train_runs_dir.iterdir() if d.is_dir()]
    num_messages = {}

    check_are_valid_run_dirs(train_runs_dir, run_dirs)

    for dir_name in run_dirs:
        msg_file = train_runs_dir / dir_name / 'messages_with_alignment.jsonl'
        msg_list = _load_jsonl(msg_file)
        num_messages[dir_name] = len(msg_list)

    vals = np.array(list(num_messages.values()))
    # Summarize statistics

    print(f"Mean num. messages : {np.mean(vals)}")
    print(f"Max num. messages : {np.min(vals)}")
    print(f"Max num. messages : {np.max(vals)}")

def sample_messages_from_fixed_proportion(train_runs_dir, rng, sample_proportions,
                                          num_messages = 100, debug = False):

    # train_runs_dir : directory of runs, assumed to have a messages_with_alignment.jsonl file
    # rng : np random number generator
    # sample_proportions : a two element tuple (p,q). For a trajectory with N total messages, we sample
    #       messages from between the pN^th and qN^th messages 
    # num_messages : the total number of messages to sample
    # plot : toggle to choose whether to visualize the sampled stances

    if not isinstance(sample_proportions, tuple):
        raise ValueError(f"{sample_proportions} is not a tuple!" )
    elif len(sample_proportions) != 2:
        raise ValueError(f"{sample_proportions} should have 2 elements. It has {len(sample_proportions)}")
    else:    
        if not (0 <= sample_proportions[0] <= sample_proportions[1] <= 1):
            raise ValueError(f"[{sample_proportions[0]},{sample_proportions[1]}] is not a valid proportion interval")

    run_dirs = [d.name for d in train_runs_dir.iterdir() if d.is_dir()]
    check_are_valid_run_dirs(train_runs_dir, run_dirs)

    sampled_messages_list = []
    sampled_stance_list = []    # for debug: make sure we're getting an even stance distribution; obviously won't log these values

    dir_sample_count = {}

    while (len(sampled_messages_list)) < num_messages:

        # sample a directory from run_dirs
        sampled_dir_name = rng.choice(run_dirs)

        # sample a message from the trajectory
        msg_file = train_runs_dir / sampled_dir_name / 'messages_with_alignment.jsonl'

        msg_list = _load_jsonl(msg_file)
        msg_list_len = len(msg_list)
        min_index = math.floor(sample_proportions[0] * msg_list_len)
        max_index = math.ceil(sample_proportions[1] * (msg_list_len - 1))

        allowed_msg_indices = list(range(min_index,max_index+1))

        sampled_index = rng.choice(allowed_msg_indices)

        text = msg_list[sampled_index]["message"]
        sample_obj = {'dir_name': str(sampled_dir_name), 'index': int(sampled_index), 'text': str(text)}
        if not (sample_obj in sampled_messages_list):
            sampled_messages_list.append(sample_obj)
            dir_sample_count[str(sampled_dir_name)] = dir_sample_count.get(str(sampled_dir_name), 0) + 1

        ss = msg_list[sampled_index]["published"]["stance_score"]

        sampled_stance_list.append(ss)

    if debug:
        print("Sampled messages : ", sampled_messages_list)

        fig, axes = plt.subplots(1,2)

        axes[0].hist(sampled_stance_list)
        axes[0].set_ylabel("Frequency")
        axes[0].set_xlabel("Logged stance score")

        sampled_indices = [item['index'] for item in sampled_messages_list]

        axes[1].hist(sampled_indices)
        axes[1].set_ylabel("Frequency")
        axes[1].set_xlabel("Index in run")

        print(dir_sample_count)

        plt.show()

    return sampled_messages_list




if __name__ == "__main__":

    train_runs_dir = RUNS_DIR / LLM / TOPIC / 'train'

    get_message_num_statistics(train_runs_dir)

    rng = np.random.default_rng(42)

    sampled_message_list = sample_messages_from_fixed_proportion(train_runs_dir, rng, (0.0, 0.1), num_messages = 100, debug = False)
    

    MESSAGE_WRITE_DIR.mkdir(parents=True, exist_ok=True)
    write_file = MESSAGE_WRITE_DIR / 'sampled_messages.jsonl'
    with write_file.open("w", encoding="utf-8") as handle:
        for row in sampled_message_list:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


    write_file_csv = MESSAGE_WRITE_DIR / 'sampled_messages.csv'
    # Using csv writing code from
    #  https://stackoverflow.com/questions/3086973/how-do-i-convert-this-list-of-dictionaries-to-a-csv-file
    keys = sampled_message_list[0].keys()
    with write_file_csv.open("w", newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(sampled_message_list)
