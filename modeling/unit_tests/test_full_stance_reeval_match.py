"""
(Partial) unit tests for checking log_correcter.py functionality when we re-evaluate
the stance of all messages in all runs.

In particular, this file focusses on checking that, when we reembed with the prompts used for
embedding during experiments, we recover the same values
"""

import unittest
import random
from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
# ensure project imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.stance_analysis_for_modeling.embedding_analyzer_sync import EmbeddingAnalyzerSync
from modeling.log_correcter import clean_log_times_and_init

POISSON_MEAN = 15.0
TIME_UNIT_MS = 8000
RATE_PER_S = (POISSON_MEAN / (TIME_UNIT_MS / 1000.0))  # (incorrect but kept for consistency with previous code)

PLOT_FOR_DEBUG = False

# baseline statements which old and new stance analysis should match
baseline_statements = {
    ("gemma3", "climate"): "Climate Change is caused by human activities",
    ("gemma3", "vaccines"): "Vaccines cause austim",
    ("gemma3", "gun-control"): "Gun ownership is a fundamental right.",
    ("llama3.1", "climate"): "Climate Change is caused by human activities",
    ("llama3.1", "vaccines"): "Vaccines cause austim",
    ("llama3.1", "gun-control"): "Gun ownership is a fundamental right",
    ("qwen-3.1", "climate"): "Climate Change is caused by human activities",
    ("qwen-3.1", "vaccines"): "Vaccines cause autism",
    ("qwen-3.1", "gun-control"): "Gun ownership is a fundamental right.",

}

# for the case of gun control llama 3.1, use a slightly different baseline statement 
#  for runs after the time-stamp 20260508-134010 as the prompt was changed  mid-experiment to "Gun ownership is a fundemental right"
time_switch_file_gc_l_3_1 = "run_20260508-134010"
alt_l3_1_gun_control_baseline_statement = "Gun ownership is a fundemental right"

THRESHOLD = 1e-03 # threshold for flagging differences in recomputed initial opinions


class TestLogCorrectionFullReEval(unittest.TestCase):

    def test_compare_opinions_expect_match(self):

        RUNS_DIR = ROOT / 'modeling' / 'runs'

        ALL_LLMS = sorted([d.name for d in RUNS_DIR.iterdir() if d.is_dir()])

        # assert that discovered LLMs are a subset of the baseline_statements keys
        discovered_llm_topic_pairs = {(llm, topic) for llm in ALL_LLMS for topic in sorted([d.name for d in (RUNS_DIR / llm).iterdir() if d.is_dir()])}
        baseline_llm_topic_pairs = set(baseline_statements.keys())

        self.assertTrue(discovered_llm_topic_pairs.issubset(baseline_llm_topic_pairs), f"Discovered LLM/topic pairs {discovered_llm_topic_pairs} are not a subset of baseline_statements keys {baseline_llm_topic_pairs}.")
        self.assertTrue(discovered_llm_topic_pairs.issuperset(baseline_llm_topic_pairs), f"Discovered LLM/topic pairs {discovered_llm_topic_pairs} are not a superset of baseline_statements keys {baseline_llm_topic_pairs}.")

        error_count_dict = {}


        for llm_name in ALL_LLMS:

            llm_path = RUNS_DIR / llm_name
            topics = sorted([d.name for d in llm_path.iterdir() if d.is_dir()])

            for topic_name in topics:
                error_count_dict[(llm_name, topic_name)] = 0
                
                if not (llm_name, topic_name) in baseline_statements:
                    raise ValueError(f"No baseline statement tracked for {llm_name}/{topic_name} in baseline_statements.")

                embedding_analyzer = EmbeddingAnalyzerSync(topic=baseline_statements[(llm_name, topic_name)])

                llm_topic_path = llm_path / topic_name
                experiment_dir = llm_topic_path

                rng = random.Random(1234)
                message_list_out_dict = clean_log_times_and_init(experiment_dir=experiment_dir,
                                         out_dir = '.',
                                         poisson_lambda=RATE_PER_S,
                                         rng=rng,
                                         embedding_analyzer=embedding_analyzer,
                                         re_embed_all_messages=True,
                                         skip_file_write_test=True)["message_list_out_dict"]

                # build two lists of original and recomputed stances to plot later

                ss_list_orig = []
                ss_list_recomputed = []

                # iterate over all the messages in the out dict, 
                #   check that the recomputed stance matches the original stance for each message
                for split_name, runs_dict in message_list_out_dict.items():
                    for run_name, messages in runs_dict.items():
                        # if llm_name == "llama3.1" and topic_name == "gun-control" 
                        #  and run comes after the time-stamp in time_switch_file_gc_l_3_1, 
                        #  skip the check for that run as the prompt was changed mid-experiment
                        if (llm_name == "llama3.1" and topic_name == "gun-control" and run_name >= time_switch_file_gc_l_3_1):
                            continue

                        for message in messages:
                            original_stance = message['published']['old_stance_score']
                            new_stance = message['published']['stance_score']

                            if not (abs(original_stance - new_stance) < THRESHOLD):
                                print(f"Stance mismatch of magnitude {abs(original_stance - new_stance)} for {llm_name}/{topic_name}/{run_name}: original stance {original_stance}, new stance {new_stance}")
                                error_count_dict[(llm_name, topic_name)] = error_count_dict.get((llm_name, topic_name), 0) + 1
                
                            ss_list_orig.append(original_stance)
                            ss_list_recomputed.append(new_stance)

                if PLOT_FOR_DEBUG:
                    plt.figure(figsize=(10, 5))
                    plt.scatter(ss_list_orig, ss_list_recomputed, alpha=0.5)
                    plt.title(f"Stance Comparison for {llm_name}/{topic_name}")
                    plt.xlabel("Original Stance")
                    plt.ylabel("Recomputed Stance")
                    plt.plot([-1, 1], [-1, 1], color='red', linestyle='--', label='y=x')
                    plt.xlim(-1, 1)
                    plt.ylim(-1, 1)
                    plt.legend()
                    plt.grid(True)
                    plt.show()

                for (llm_name, topic_name), count in error_count_dict.items():
                    print(f"Total stance mismatches for {llm_name}/{topic_name}: {count}")
                for (llm_name, topic_name), count in error_count_dict.items():
                    self.assertEqual(count, 0, f"Failure due to positive stance mismatches {llm_name}/{topic_name}: {count}")
                    

if __name__ == "__main__":
    unittest.main()