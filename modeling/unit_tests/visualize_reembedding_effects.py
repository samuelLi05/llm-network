
import unittest
import random
from pathlib import Path
import json
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
# ensure project imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ORIGINAL_LOGS_DIR = Path(__file__).resolve().parents[2] / 'modeling' / 'runs'
NEW_LOGS_DIR = Path(__file__).resolve().parents[2] / 'modeling' / 'runs_rescored'

if __name__ == "__main__":

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