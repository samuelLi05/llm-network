
import unittest
import random
from pathlib import Path
import json
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse


ROOT = Path(__file__).resolve().parents[2]
# ensure project imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def vis_rq2():

    ORIGINAL_LOGS_DIR = Path(__file__).resolve().parents[2] / 'modeling' / 'runs'
    NEW_LOGS_DIR = Path(__file__).resolve().parents[2] / 'modeling' / 'runs_rescored'

    # iterate over all LLMs, topics, and runs,
    #  and visualize changes in the stance scores

    data_dict = {}

    for llm_dir in ORIGINAL_LOGS_DIR.iterdir():
        if not llm_dir.is_dir():
            raise ValueError(f"Expected LLM directory, but found a file: {llm_dir}")
        for topic_dir in llm_dir.iterdir():
            if not topic_dir.is_dir():
                raise ValueError(f"Expected topic directory, but found a file: {topic_dir}")
            
            data_dict[(llm_dir.name, topic_dir.name)] = {"original_scores": [],
                                                         "new_scores": []}

            for split_dir in topic_dir.iterdir():
                if not split_dir.is_dir():
                    raise ValueError(f"Expected split directory, but found a file: {split_dir}")
                if not split_dir.name in ["train", "test"]:
                    raise ValueError(f"Expected split directory to be 'train' or 'test', but found: {split_dir.name}")
                if split_dir.name == "test":
                    continue
                for run_dir in split_dir.iterdir():
                    if not run_dir.is_dir():
                        raise ValueError(f"Expected run directory, but found a file: {run_dir}")

                    # load the original and new logs
                    original_log_path = run_dir / "messages_with_alignment.jsonl"
                    new_log_path = NEW_LOGS_DIR / llm_dir.name / topic_dir.name / split_dir.name / run_dir.name / "messages_with_alignment.jsonl"

                    if not original_log_path.exists():
                        raise FileNotFoundError(f"Original log file not found: {original_log_path}")
                    if not new_log_path.exists():
                        raise FileNotFoundError(f"New log file not found: {new_log_path}")

                    with open(original_log_path, 'r') as f:
                        original_messages = [json.loads(line) for line in f]

                    with open(new_log_path, 'r') as f:
                        new_messages = [json.loads(line) for line in f]

                    # check that the number of messages is the same
                    if len(original_messages) != len(new_messages):
                        raise ValueError(f"Number of messages in original and new logs do not match for {llm_dir.name}/{topic_dir.name}/{split_dir.name}/{run_dir.name}: {len(original_messages)} vs {len(new_messages)}")

                    for i, (orig_msg, new_msg) in enumerate(zip(original_messages, new_messages)):
                        orig_stance = orig_msg['published']['stance_score']
                        new_stance = new_msg['published']['stance_score']

                        data_dict[(llm_dir.name, topic_dir.name)]["original_scores"].append(orig_stance)
                        data_dict[(llm_dir.name, topic_dir.name)]["new_scores"].append(new_stance)     


    # Next, build a <num_llms> x <num_topics> subplot of scatter plots, 
    #  where each subplot is a scatter plot of original vs. new stance scores for that LLM/topic pair    
    num_llms = len(set(llm for llm, topic in data_dict.keys()))
    num_topics = len(set(topic for llm, topic in data_dict.keys()))

    fig, axes = plt.subplots(num_llms, num_topics, figsize=(5*num_topics, 5*num_llms), squeeze=False)
    for i, llm in enumerate(sorted(set(llm for llm, topic in data_dict.keys()))):
        for j, topic in enumerate(sorted(set(topic for llm, topic in data_dict.keys()))):
            ax = axes[i, j]
            original_scores = data_dict[(llm, topic)]["original_scores"]
            new_scores = data_dict[(llm, topic)]["new_scores"]

            ax.scatter(original_scores, new_scores, alpha=0.5)
            ax.set_xlabel("Original Stance Score")
            ax.set_ylabel("New Stance Score")
            ax.set_title(f"{llm}/{topic}")

    plt.show()

def vis_rq3():

    ORIGINAL_LOGS_DIR = Path(__file__).resolve().parents[2] / 'modeling' / 'runs_fg_vs_adj_cr'
    NEW_LOGS_DIR = Path(__file__).resolve().parents[2] / 'modeling' / 'runs_fg_vs_adj_cr_rescored'

    # iterate over all LLMs, topics, and runs,
    #  and visualize changes in the stance scores
    # note that in this case we expect only two pairs, each of which has a single LLM and topic

    data_dict = {}

    for llm_dir in ORIGINAL_LOGS_DIR.iterdir():
        if not llm_dir.is_dir():
            raise ValueError(f"Expected LLM directory, but found a file: {llm_dir}")
        for topic_dir in llm_dir.iterdir():
            if not topic_dir.is_dir():
                raise ValueError(f"Expected topic directory, but found a file: {topic_dir}")
            
            data_dict[(llm_dir.name, topic_dir.name)] = {"original_scores": [],
                                                         "new_scores": []}

            for split_dir in topic_dir.iterdir():
                if not split_dir.is_dir():
                    raise ValueError(f"Expected split directory, but found a file: {split_dir}")
                if not split_dir.name in ["train", "test"]:
                    raise ValueError(f"Expected split directory to be 'train' or 'test', but found: {split_dir.name}")
                if split_dir.name == "test":
                    continue
                for run_dir in split_dir.iterdir():
                    if not run_dir.is_dir():
                        raise ValueError(f"Expected run directory, but found a file: {run_dir}")

                    # load the original and new logs
                    original_log_path = run_dir / "messages_with_alignment.jsonl"
                    new_log_path = NEW_LOGS_DIR / llm_dir.name / topic_dir.name / split_dir.name / run_dir.name / "messages_with_alignment.jsonl"

                    if not original_log_path.exists():
                        raise FileNotFoundError(f"Original log file not found: {original_log_path}")
                    if not new_log_path.exists():
                        raise FileNotFoundError(f"New log file not found: {new_log_path}")

                    with open(original_log_path, 'r') as f:
                        original_messages = [json.loads(line) for line in f]

                    with open(new_log_path, 'r') as f:
                        new_messages = [json.loads(line) for line in f]

                    # check that the number of messages is the same
                    if len(original_messages) != len(new_messages):
                        raise ValueError(f"Number of messages in original and new logs do not match for {llm_dir.name}/{topic_dir.name}/{split_dir.name}/{run_dir.name}: {len(original_messages)} vs {len(new_messages)}")

                    for i, (orig_msg, new_msg) in enumerate(zip(original_messages, new_messages)):
                        orig_stance = orig_msg['published']['stance_score']
                        new_stance = new_msg['published']['stance_score']
                        data_dict[(llm_dir.name, topic_dir.name)]["original_scores"].append(orig_stance)
                        data_dict[(llm_dir.name, topic_dir.name)]["new_scores"].append(new_stance)

    # Next build a #pairs subplot of scatter plots
    #  where each subplot is a scatter plot of original vs. new stance scores for that LLM/topic pair

    num_pairs = len(data_dict)

    fig, axes = plt.subplots(1, num_pairs, figsize=(5*num_pairs, 5), squeeze=False)
    for j, (llm, topic) in enumerate(sorted(data_dict.keys())):
        ax = axes[0, j]
        original_scores = data_dict[(llm, topic)]["original_scores"]
        new_scores = data_dict[(llm, topic)]["new_scores"]

        ax.scatter(original_scores, new_scores, alpha=0.5)
        ax.set_xlabel("Original Stance Score")
        ax.set_ylabel("New Stance Score")
        ax.set_title(f"{llm}/{topic}")

    plt.show()

def vis_rq4():
    # Similar for above, where rq4 refers to experiments 
    LLM = 'llama3.1'
    TOPIC = 'vaccines'

    ORIGINAL_LOGS_DIR = ROOT / 'modeling' / 'runs_varied_size' / LLM / TOPIC
    NEW_LOGS_DIR = ROOT / 'modeling' / 'runs_varied_size_rescored' / LLM / TOPIC

    data_dict = {}

    for experiment_dir in ORIGINAL_LOGS_DIR.iterdir():
        if not experiment_dir.is_dir():
            raise ValueError(f"Expected only directories in {ORIGINAL_LOGS_DIR}, but found {experiment_dir}.")

        data_dict[experiment_dir.name] = {"original_scores": [], "new_scores": []}
        for split_dir in experiment_dir.iterdir():
            if not split_dir.is_dir():
                raise ValueError(f"Expected only directories in {experiment_dir}, but found {split_dir}.")
            if not split_dir.name in ["train","test"]:
                raise ValueError(f"Expected split directory to be 'train' or 'test', but found: {split_dir.name}")

            if split_dir.name == 'test':
                continue

            for run_dir in split_dir.iterdir():
                if not run_dir.is_dir():
                    raise ValueError(f"Expected run directory, but found a file: {run_dir}")

                # load the original and new logs
                original_log_path = run_dir / "messages_with_alignment.jsonl"
                new_log_path = NEW_LOGS_DIR / experiment_dir.name / split_dir.name / run_dir.name / "messages_with_alignment.jsonl"

                if not original_log_path.exists():
                    raise FileNotFoundError(f"Original log file not found: {original_log_path}")
                if not new_log_path.exists():
                    raise FileNotFoundError(f"New log file not found: {new_log_path}")

                with open(original_log_path, 'r') as f:
                    original_messages = [json.loads(line) for line in f]

                with open(new_log_path, 'r') as f:
                    new_messages = [json.loads(line) for line in f]

                # check that the number of messages is the same
                if len(original_messages) != len(new_messages):
                    raise ValueError(f"Number of messages in original and new logs do not match")

                for i, (orig_msg, new_msg) in enumerate(zip(original_messages, new_messages)):
                    orig_stance = orig_msg['published']['stance_score']
                    new_stance = new_msg['published']['stance_score']

                    data_dict[experiment_dir.name]["original_scores"].append(orig_stance)
                    data_dict[experiment_dir.name]["new_scores"].append(new_stance)     

    # Now build a subplot of scatter plots, where each subplot is a scatter plot of original vs. new stance scores for that experiment
    num_experiments = len(data_dict)
    
    fig, axes = plt.subplots(1, num_experiments, figsize=(5*num_experiments, 5), squeeze=False)
    for j, experiment_name in enumerate(sorted(data_dict.keys())):
        ax = axes[0, j]
        original_scores = data_dict[experiment_name]["original_scores"]
        new_scores = data_dict[experiment_name]["new_scores"]

        ax.scatter(original_scores, new_scores, alpha=0.5)
        ax.set_xlabel("Original Stance Score")
        ax.set_ylabel("New Stance Score")
        ax.set_title(f"{experiment_name}")
    plt.show()
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize changes when rescoring runs for each rq")
    parser.add_argument("--rq", type=int, default=2, help="The research question number for which to rescore runs (default: 2)")
    args = parser.parse_args()

    if not args.rq in [2,3,4]:
        raise ValueError(f"Invalid research question number {args.rq}. Must be 2, 3, or 4.")
    else:
        print(f"Visualizing for research question {args.rq}...")
    if args.rq == 2:
        vis_rq2()
    if args.rq == 3:
        vis_rq3()
    if args.rq == 4:
        vis_rq4()