"""
Utility script to carry out the following functions:
  1. Rescore all runs using common baseline statements
  2. Rescore runs for a fixed llm/topic using a range of baseline statements to assess sensitivity of stance scores
"""

import random
from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt
import argparse

ROOT = Path(__file__).resolve().parents[1]
# ensure project imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.log_correcter import clean_log_times_and_init
from modeling.stance_analysis_for_modeling.embedding_analyzer_sync import EmbeddingAnalyzerSync

POISSON_MEAN = 15.0
TIME_UNIT_MS = 8000
RATE_PER_S = (POISSON_MEAN / (TIME_UNIT_MS / 1000.0))  # (incorrect but kept for consistency with previous code)

# Reference baseline statements that are
#  1. consistent in punctuation
#  2. correct in spelling
# Otherwise same as baseline statements used for scoring

REFERENCE_BASELINE_STATEMENTS = {
    "climate": "Climate Change is caused by human activities",
    "vaccines": "Vaccines cause autism",
    "gun-control": "Gun ownership is a fundamental right"
}

def rescore_all_runs_rq2(args):
    ORIG_RUNS_DIR = ROOT / 'modeling' / 'runs'
    NEW_RUNS_DIR = ROOT / 'modeling' / 'runs_rescored'

    ALL_LLMS = sorted([d.name for d in ORIG_RUNS_DIR.iterdir() if d.is_dir()])

    rng = random.Random(1234)
    for llm_name in ALL_LLMS:
        llm_path = ORIG_RUNS_DIR / llm_name
        topics = sorted([d.name for d in llm_path.iterdir() if d.is_dir()])

        for topic_name in topics:
            if not topic_name in REFERENCE_BASELINE_STATEMENTS:
                raise ValueError(f"No reference baseline statement tracked for {llm_name}/{topic_name} in REFERENCE_BASELINE_STATEMENTS.")
            
            embedding_analyzer = EmbeddingAnalyzerSync(topic=REFERENCE_BASELINE_STATEMENTS[topic_name])

            experiment_dir = llm_path / topic_name
            out_dir = NEW_RUNS_DIR / llm_name / topic_name
            out_dir.mkdir(parents=True, exist_ok=True)

            message_list_out_dict = clean_log_times_and_init(experiment_dir=experiment_dir,
                                                             out_dir=out_dir,
                                                             poisson_lambda=RATE_PER_S,
                                                             rng=rng,
                                                             embedding_analyzer=embedding_analyzer,
                                                             re_embed_all_messages=True,
                                                             return_out_dict=True,
                                                             bypass_init_match_validation=True,
                                                             skip_time_reassignment=True)['message_list_out_dict']
            
            # above code should regenerate all data as required
            if args.plot:
                ss_list_orig = []
                ss_list_recomputed = []

                for split_name, runs_dict in message_list_out_dict.items():
                    if split_name == 'test':
                        continue
                    for run_name, messages in runs_dict.items():
                        for message in messages:
                            original_stance = message['published']['old_stance_score']
                            new_stance = message['published']['stance_score']

                            ss_list_orig.append(original_stance)
                            ss_list_recomputed.append(new_stance)

                plt.scatter(ss_list_orig, ss_list_recomputed)
                plt.xlabel("Original Stance Score")
                plt.ylabel("Recomputed Stance Score")
                plt.title(f"Stance Score Comparison for {llm_name}/{topic_name}")
                plt.show()

def rescore_all_runs_rq3(args):

    ORIG_RUNS_DIR = ROOT / 'modeling' / 'runs_fg_vs_adj_cr'
    NEW_RUNS_DIR = ROOT / 'modeling' / 'runs_fg_vs_adj_cr_rescored'

    for llm_name in sorted([d.name for d in ORIG_RUNS_DIR.iterdir() if d.is_dir()]):
        llm_path = ORIG_RUNS_DIR / llm_name
        topics = sorted([d.name for d in llm_path.iterdir() if d.is_dir()])

        for topic_name in topics:
            if not topic_name in REFERENCE_BASELINE_STATEMENTS:
                raise ValueError(f"No reference baseline statement tracked for {llm_name}/{topic_name} in REFERENCE_BASELINE_STATEMENTS.")
            
            embedding_analyzer = EmbeddingAnalyzerSync(topic=REFERENCE_BASELINE_STATEMENTS[topic_name])

            experiment_dir = llm_path / topic_name
            out_dir = NEW_RUNS_DIR / llm_name / topic_name
            out_dir.mkdir(parents=True, exist_ok=True)

            message_list_out_dict = clean_log_times_and_init(experiment_dir=experiment_dir,
                                                             out_dir=out_dir,
                                                             poisson_lambda=RATE_PER_S,
                                                             rng=random.Random(1234),   # doesn't matter as we skip time reassignment
                                                             embedding_analyzer=embedding_analyzer,
                                                             re_embed_all_messages=True,
                                                             return_out_dict=True,
                                                             bypass_init_match_validation=True,
                                                             skip_time_reassignment=True)['message_list_out_dict']

            if args.plot:
                # Plot new vs. old stance scores for all runs
                ss_list_orig = []
                ss_list_recomputed = []

                for split_name, runs_dict in message_list_out_dict.items():
                    if split_name == 'test':
                        continue
                    for run_name, messages in runs_dict.items():
                        for message in messages:
                            original_stance = message['published']['old_stance_score']
                            new_stance = message['published']['stance_score']

                            ss_list_orig.append(original_stance)
                            ss_list_recomputed.append(new_stance)
                
                plt.scatter(ss_list_orig, ss_list_recomputed)
                plt.xlabel("Original Stance Score")
                plt.ylabel("Recomputed Stance Score")
                plt.title(f"Stance Score Comparison for {llm_name}/{topic_name}")
                plt.show()


def rescore_all_runs_rq4(args):

    LLM = 'llama3.1'
    TOPIC = 'vaccines'

    ORIG_RUNS_DIR = ROOT / 'modeling' / 'runs_varied_size' / LLM / TOPIC
    NEW_RUNS_DIR = ROOT / 'modeling' / 'runs_varied_size_rescored' / LLM / TOPIC

    embedding_analyzer = EmbeddingAnalyzerSync(topic=REFERENCE_BASELINE_STATEMENTS[TOPIC])

    for ns_dir in ORIG_RUNS_DIR.iterdir():
        if not ns_dir.is_dir():
            raise ValueError(f"Expected only directories in {ORIG_RUNS_DIR}, but found {ns_dir}.")
        out_dir = NEW_RUNS_DIR / ns_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # ns_dir.name will be of the form n_{num_agents}
        # if num_agents is not 30, we should not skip assignment

        num_agents = int(ns_dir.name.split('_')[1])
        skip_time_reassignment = (num_agents == 30)

        message_list_out_dict = clean_log_times_and_init(experiment_dir=ns_dir,
                                 out_dir = out_dir,
                                 poisson_lambda=RATE_PER_S,
                                 rng=random.Random(1234),
                                 embedding_analyzer=embedding_analyzer,
                                 re_embed_all_messages=True,
                                 return_out_dict=True,
                                 bypass_init_match_validation=True,
                                    skip_time_reassignment=skip_time_reassignment)['message_list_out_dict']

        if args.plot:
            # Plot new vs. old stance scores for all runs
            ss_list_orig = []
            ss_list_recomputed = []

            for split_name, runs_dict in message_list_out_dict.items():
                if split_name == 'test':
                    continue
                for run_name, messages in runs_dict.items():
                    for message in messages:
                        original_stance = message['published']['old_stance_score']
                        new_stance = message['published']['stance_score']

                        ss_list_orig.append(original_stance)
                        ss_list_recomputed.append(new_stance)
            
            plt.scatter(ss_list_orig, ss_list_recomputed)
            plt.xlabel("Original Stance Score")
            plt.ylabel("Recomputed Stance Score")
            plt.title(f"Stance Score Comparison for {LLM}/{TOPIC}/{ns_dir.name}")
            plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Rescore runs using various baseline statements")
    parser.add_argument("--plot", action="store_true", help="Plot new vs. old stance scores for each run")
    parser.add_argument("--rq", type=int, default=2, help="The research question number for which to rescore runs (default: 2)")
    args = parser.parse_args()

    if not args.rq in [2,3,4]:
        raise ValueError(f"Invalid research question number {args.rq}. Must be 2, 3, or 4.")
    else:
        print(f"Rescoring runs for research question {args.rq}...")
    if args.rq == 2:
        rescore_all_runs_rq2(args)
    if args.rq == 3:
        rescore_all_runs_rq3(args)
    if args.rq == 4:
        rescore_all_runs_rq4(args)