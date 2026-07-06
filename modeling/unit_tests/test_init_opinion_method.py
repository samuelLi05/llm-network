"""
    There are two ways of computing the initial opinion for agents who don't post in the first time slice:
        1. Re-embed the stable baseline statement for the agent
        2. Go through the route used in data_prep.py where it's pulled from the topology snapshot

    In this script, we compare the two methods to check that they produce the same results        
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import unittest

import argparse

ROOT = Path(__file__).resolve().parents[2]
# ensure project imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.models.data_prep import _numeric_agent_key, load_run_data, build_run_trajectory
from modeling.stance_analysis_for_modeling.embedding_analyzer_sync import EmbeddingAnalyzerSync


from modeling.models.data_prep_network_size import (
    load_cleaned_run_data,
    build_run_trajectory_from_clean
)

PARAMS = {
    'target_agent_fraction': 0.4,
    'constrain_messages': 150,
    'rollout_horizon_cap': 20,
    'reference_agent_number': 30
}

# for each topic and LLM define the actual baseline statement used
#   these differ slightly due to spelling/punctuation differences in the prompts used for different LLMs when editing shell scripts
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


class TestInitOpinionMethods(unittest.TestCase):

    def test_compare_init_opinion_methods(self):
        # as in generate_model_rankings.py, load the run_dir

        RUNS_DIR = ROOT / 'modeling' / 'runs'

        ALL_LLMS = sorted([d.name for d in RUNS_DIR.iterdir() if d.is_dir()])

        # assert that discovered LLMs are a subset of the baseline_statements keys
        discovered_llm_topic_pairs = {(llm, topic) for llm in ALL_LLMS for topic in sorted([d.name for d in (RUNS_DIR / llm).iterdir() if d.is_dir()])}
        baseline_llm_topic_pairs = set(baseline_statements.keys())
        self.assertTrue(discovered_llm_topic_pairs.issubset(baseline_llm_topic_pairs), f"Discovered LLM/topic pairs {discovered_llm_topic_pairs} are not a subset of baseline_statements keys {baseline_llm_topic_pairs}.")

        pct_diff_list = []

        for llm_name in ALL_LLMS:

            llm_path = RUNS_DIR / llm_name
            topics = sorted([d.name for d in llm_path.iterdir() if d.is_dir()])


            # build an embedding analyzer for the LLM


            for topic_name in topics:
                if not (llm_name, topic_name) in baseline_statements:
                    raise ValueError(f"No baseline statement tracked for {llm_name}/{topic_name} in baseline_statements.")

                embedding_analyzer = EmbeddingAnalyzerSync(topic=baseline_statements[(llm_name, topic_name)])


                # method one: load up the opinions from the run data
                train_path = llm_path / topic_name / 'train'
                test_path = llm_path / topic_name / 'test'

                if not train_path.exists() or not test_path.exists():
                    raise ValueError(f"Train or test path does not exist for {llm_name}/{topic_name}: {train_path}, {test_path}")
                

                run_dirs = sorted([p for p in train_path.iterdir() if p.is_dir()])
                run_dirs += sorted([p for p in test_path.iterdir() if p.is_dir()])
                

                run_data = {r.name: load_run_data(r) for r in run_dirs}
                global_agents = sorted({a for d in run_data.values() for a in d['agent_ids']}, key=_numeric_agent_key)
                n_agents = len(global_agents)

                traj_mask = {rn: build_run_trajectory(d, global_agents, target_agent_fraction=PARAMS['target_agent_fraction'], return_post_mask=True, constrain_messages=PARAMS['constrain_messages']) for rn, d in run_data.items()}
                run_traj = {rn: tm[0] for rn, tm in traj_mask.items()}
                agent_index = {a: i for i, a in enumerate(global_agents)}


                # method two: recompute the initial opinions from the stable baseline statements
                
                swapped = False
                for r in tqdm(run_dirs, desc=f"[{llm_name}/{topic_name}] Recomputing initial opinions"):
                    
                    # do the embedding swap out for llama 3.1 gun control runs after the time-stamp 20260508-134010
                    if llm_name == "llama3.1" and topic_name == "gun-control" and r.name >= time_switch_file_gc_l_3_1:
                        if not swapped:
                            embedding_analyzer = EmbeddingAnalyzerSync(topic=alt_l3_1_gun_control_baseline_statement, use_local_embedding_model=True)
                            swapped = True

                    # load the static_init.json for the run
                    static_init_file = r / 'static_init.json'

                    if not static_init_file.exists():
                        raise ValueError(f"static_init.json not found for run {r.name} at {static_init_file}")
                    
                    # load the json and pull [agent_configs]
                    with open(static_init_file, 'r') as f:
                        static_init_data = json.load(f)
                    agent_configs = static_init_data.get('agent_configs', [])

                    for a_id, a_config in agent_configs.items():
                        # check if the agent posted in the first time slice using the traj_mask data
                        a_idx = agent_index[a_id]
                        posted_in_first_slice = traj_mask[r.name][1][0, a_idx]

                        if not posted_in_first_slice:
                            # recompute the initial opinion using the embedding analyzer
                            baseline_opinion_str = a_config.get('stable_perspective_sentence', None)
                            if baseline_opinion_str is None:
                                raise ValueError(f"No stable_perspective_sentence found for agent {a_id} in run {r.name}")

                            scored = embedding_analyzer.embed_and_score(baseline_opinion_str)
                            init_opinion_recomputed = scored['stance_score']

                            # pull the logged initial opinion from the run_traj data
                            init_opinion_logged = run_traj[r.name][0, a_idx]

                            # compute the percentage difference from logged to recomputed
                            if init_opinion_logged == 0:
                                if init_opinion_recomputed == 0:
                                    pct_diff = 0.0
                                else:
                                    pct_diff = float('inf')
                            else:
                                pct_diff = abs(init_opinion_logged - init_opinion_recomputed) / abs(init_opinion_logged)

                            self.assertLessEqual(pct_diff, THRESHOLD, f"[{llm_name}/{topic_name}] Run {r.name}, Agent {a_id}: Logged init opinion = {init_opinion_logged}, Recomputed init opinion = {init_opinion_recomputed}, Pct diff = {pct_diff:.4f}")
                            pct_diff_list.append({
                                'llm': llm_name,
                                'topic': topic_name,
                                'run': r.name,
                                'agent_id': a_id,
                                'logged_init_opinion': init_opinion_logged,
                                'recomputed_init_opinion': init_opinion_recomputed,
                                'pct_diff': pct_diff
                            })

    def test_compare_init_opinion_methods_ns(self):
        # Same as above but for the experiments on network size effects

        RUNS_DIR = ROOT / 'modeling' / 'runs_varied_size_corrected' / 'llama3.1' / 'vaccines'

        ALL_EXPERIMENTS = sorted([d.name for d in RUNS_DIR.iterdir() if d.is_dir()])


        for exp_dir in ALL_EXPERIMENTS:

            exp_path = RUNS_DIR / exp_dir
            embedding_analyzer = EmbeddingAnalyzerSync(topic=baseline_statements[("llama3.1", "vaccines")], use_local_embedding_model=True)

            train_path = exp_path / 'train'
            test_path = exp_path / 'test'

            if not train_path.exists() or not test_path.exists():
                raise ValueError(f"Train or test path does not exist for experiment {exp_dir}: {train_path}, {test_path}")
            
            run_dirs = sorted([p for p in train_path.iterdir() if p.is_dir()])
            run_dirs += sorted([p for p in test_path.iterdir() if p.is_dir()])

            run_data = {r.name: load_cleaned_run_data(r) for r in run_dirs}
            global_agents = sorted({a for d in run_data.values() for a in d['agent_ids']}, key=_numeric_agent_key)

            traj_mask = {rn: build_run_trajectory_from_clean(d, global_agents, target_agent_fraction=PARAMS['target_agent_fraction'], return_post_mask=True, constrain_messages=PARAMS['constrain_messages'], reference_agent_number=PARAMS['reference_agent_number']) for rn, d in run_data.items()}
            run_traj = {rn: tm[0] for rn, tm in traj_mask.items()}
            agent_index = {a: i for i, a in enumerate(global_agents)}

            for r in tqdm(run_dirs, desc=f"[{exp_dir}] Recomputing initial opinions"):

                static_init_file = r / 'static_init.json'

                if not static_init_file.exists():
                    raise ValueError(f"static_init.json not found for run {r.name} at {static_init_file}")
                
                # load the json and pull [agent_configs]
                with open(static_init_file, 'r') as f:
                    static_init_data = json.load(f)
                agent_configs = static_init_data.get('agent_configs', {})

                for a_id, a_config in agent_configs.items():
                    # check if the agent posted in the first time slice using the traj_mask data
                    a_idx = agent_index[a_id]
                    posted_in_first_slice = traj_mask[r.name][1][0, a_idx]

                    if not posted_in_first_slice:
                        # recompute the initial opinion using the embedding analyzer
                        baseline_opinion_str = a_config.get('stable_perspective_sentence', None)
                        if baseline_opinion_str is None:
                            raise ValueError(f"No stable_perspective_sentence found for agent {a_id} in run {r.name}")

                        scored = embedding_analyzer.embed_and_score(baseline_opinion_str)
                        init_opinion_recomputed = scored['stance_score']

                        # pull the logged initial opinion from the run_traj data
                        init_opinion_logged = run_traj[r.name][0, a_idx]

                        # compute the percentage difference from logged to recomputed
                        if init_opinion_logged == 0:
                            if init_opinion_recomputed == 0:
                                pct_diff = 0.0
                            else:
                                pct_diff = float('inf')
                        else:
                            pct_diff = abs(init_opinion_logged - init_opinion_recomputed) / abs(init_opinion_logged)

                        self.assertLessEqual(pct_diff, THRESHOLD, f"[{exp_dir}] Run {r.name}, Agent {a_id}: Logged init opinion = {init_opinion_logged}, Recomputed init opinion = {init_opinion_recomputed}, Pct diff = {pct_diff:.4f}")


if __name__ == "__main__":
    test_obj = TestInitOpinionMethods()
    test_obj.test_compare_init_opinion_methods_ns()
    breakpoint()

    unittest.main()