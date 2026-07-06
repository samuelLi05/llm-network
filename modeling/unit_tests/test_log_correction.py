"""
Docstring for modeling.unit_tests.test_log_correction

Unit tests for the log correction in log_correcter.py
"""

import unittest
import random
from pathlib import Path
import json

# Define directories for original and corrected logs
ORIGINAL_LOGS_DIR = Path(__file__).resolve().parents[1] / "runs_varied_size" / "llama3.1" / "vaccines"
CORRECTED_LOGS_DIR = Path(__file__).resolve().parents[1] / "runs_varied_size_corrected" / "llama3.1" / "vaccines"

# add root to sys.path for imports
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.log_correcter import clean_log_times_and_init
from modeling.stance_analysis_for_modeling.embedding_analyzer_sync import EmbeddingAnalyzerSync


POISSON_MEAN = 15.0
TIME_UNIT_MS = 8000
RATE_PER_S = (POISSON_MEAN / (TIME_UNIT_MS / 1000.0))  
POISSON_LAMBDA = (POISSON_MEAN / (TIME_UNIT_MS / 1000.0)) 


class TestLogCorrection(unittest.TestCase):

    def test_log_copying(self):

        # Check that the files in to_copy are the same for 
        #  each run in the original and corrected logs directories
        to_copy = ['connection_graph.json', 'run_manifest.json', 'static_init.json']

        # additionally, check that, for the messages_with_alignment.jsonl files, all fields in
        #   the new file are present in the original file, and that values match (up to dictionaries, where we nest the check)

        for experiment_dir in sorted(ORIGINAL_LOGS_DIR.iterdir()):
            if not experiment_dir.is_dir():
                continue
            
            print(f"Checking experiment_dir: {experiment_dir.name}")

            for split_dir in sorted(experiment_dir.iterdir()):
                if not split_dir.is_dir():
                    continue
                for run_dir in sorted(split_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue

                    original_run_dir = ORIGINAL_LOGS_DIR / experiment_dir.name / split_dir.name / run_dir.name
                    corrected_run_dir = CORRECTED_LOGS_DIR / experiment_dir.name / split_dir.name / run_dir.name

                    # check that the corrected run directory exists
                    self.assertTrue(corrected_run_dir.exists(), f"Corrected run directory {corrected_run_dir} does not exist")

                    for fn in to_copy:
                        original_file = original_run_dir / fn
                        corrected_file = corrected_run_dir / fn

                        # check that the file exists in both directories
                        self.assertTrue(original_file.exists(), f"Original file {original_file} does not exist")
                        self.assertTrue(corrected_file.exists(), f"Corrected file {corrected_file} does not exist")

                        # check that the contents of the files are the same
                        with open(original_file, 'r') as f1, open(corrected_file, 'r') as f2:
                            # load each file as json and compare
                            original_json = json.load(f1)
                            corrected_json = json.load(f2)
                            self.assertEqual(original_json, corrected_json, f"Contents of {original_file} and {corrected_file} differ")

                    # check the messages_with_alignment.jsonl files
                    original_jsonl_file = original_run_dir / 'messages_with_alignment.jsonl'
                    corrected_jsonl_file = corrected_run_dir / 'messages_with_alignment.jsonl'

                    # check that the files exist
                    self.assertTrue(original_jsonl_file.exists(), f"Original jsonl file {original_jsonl_file} does not exist")
                    self.assertTrue(corrected_jsonl_file.exists(), f"Corrected jsonl file {corrected_jsonl_file} does not exist")

                    # load the original jsonl file into a list of dicts
                    original_rows = []
                    with original_jsonl_file.open('r', encoding='utf-8') as f:
                        for line in f:
                            s = line.strip()
                            if s:
                                original_rows.append(json.loads(s))

                    corrected_rows = []
                    with corrected_jsonl_file.open('r', encoding='utf-8') as f:
                        for line in f:
                            s = line.strip()
                            if s:
                                corrected_rows.append(json.loads(s))

                    # assert that the number of rows is the same
                    self.assertEqual(len(original_rows), len(corrected_rows), f"Number of rows in {original_jsonl_file} and {corrected_jsonl_file} differ")

                    # define sub keys where the values are allowed to differ (because we're rewriting them)
                    diff_allowed_keys = {'t_s', 't_ms'}

                    # for each row in the corrected file, check that all fields are present in the original row
                    #  and that, if the key is a dictionary, the values match (recursively)
                    for i, (orig_row, corr_row) in enumerate(zip(original_rows, corrected_rows)):
                        for key in corr_row:
                            self.assertIn(key, orig_row, f"Key {key} in corrected row {i} not found in original row")
                            if isinstance(corr_row[key], dict):
                                # check that all keys in the corrected dict are present in the original dict
                                for subkey in corr_row[key]:
                                    self.assertIn(subkey, orig_row[key], f"Subkey {subkey} in corrected row {i} not found in original row")
                                    # check that the values match, unless the key is in the diff_allowed_keys set
                                    if subkey not in diff_allowed_keys:
                                        self.assertEqual(corr_row[key][subkey], orig_row[key][subkey], f"Value for subkey {subkey} in corrected row {i} does not match original row")
                            else:
                                # check that the values match
                                self.assertEqual(corr_row[key], orig_row[key], f"Value for key {key} in corrected row {i} does not match original row")


    def test_interarrival_distribution(self):
        # get the set of interarrival times for each run in the corrected logs directory,
        #  and check that the distribution is exponential

        # check that mean values are close to expected value of 1/POISSON_LAMBDA
        expected_mean_s = 1.0 / POISSON_LAMBDA

        cumulative_interarrival_times = []
        for experiment_dir in sorted(CORRECTED_LOGS_DIR.iterdir()):
            if not experiment_dir.is_dir():
                continue
            
            print(f"Checking interarrival distribution for experiment_dir: {experiment_dir.name}")

            for split_dir in sorted(experiment_dir.iterdir()):
                if not split_dir.is_dir():
                    continue
                for run_dir in sorted(split_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue

                    corrected_run_dir = CORRECTED_LOGS_DIR / experiment_dir.name / split_dir.name / run_dir.name

                    # check the messages_with_alignment.jsonl file
                    corrected_jsonl_file = corrected_run_dir / 'messages_with_alignment.jsonl'

                    # check that the file exists
                    self.assertTrue(corrected_jsonl_file.exists(), f"Corrected jsonl file {corrected_jsonl_file} does not exist")

                    # load the corrected jsonl file into a list of dicts
                    corrected_rows = []
                    with corrected_jsonl_file.open('r', encoding='utf-8') as f:
                        for line in f:
                            s = line.strip()
                            if s:
                                corrected_rows.append(json.loads(s))

                    # extract the t_s values and compute interarrival times
                    t_s_values = [row['time']['t_s'] for row in corrected_rows]
                    interarrival_times = [t2 - t1 for t1, t2 in zip(t_s_values[:-1], t_s_values[1:])]
                    cumulative_interarrival_times.extend(interarrival_times)

                    # compute mean interarrival time
                    mean_interarrival_time = sum(interarrival_times) / len(interarrival_times)

                    print(f"Run {run_dir.name}: mean interarrival time = {mean_interarrival_time:.4f}s, expected = {expected_mean_s:.4f}s")
                    # assert that the mean is close to expected value
                    self.assertAlmostEqual(mean_interarrival_time, expected_mean_s, delta=0.2 * expected_mean_s, msg=f"Mean interarrival time {mean_interarrival_time} for run {run_dir.name} differs from expected {expected_mean_s}")
        # compute overall mean interarrival time
        overall_mean_interarrival_time = sum(cumulative_interarrival_times) / len(cumulative_interarrival_times)
        print(f"Overall mean interarrival time = {overall_mean_interarrival_time:.4f}s, expected = {expected_mean_s:.4f}s")
        # assert that the overall mean is close to expected value
        self.assertAlmostEqual(overall_mean_interarrival_time, expected_mean_s, delta=0.01 * expected_mean_s, msg=f"Overall mean interarrival time {overall_mean_interarrival_time} differs from expected {expected_mean_s}")

        # check that the overall variance is close to expected value of 1/(POISSON_LAMBDA^2)
        expected_variance_s2 = 1.0 / (POISSON_LAMBDA ** 2)
        overall_variance = sum((x - overall_mean_interarrival_time) ** 2 for x in cumulative_interarrival_times) / len(cumulative_interarrival_times)
        print(f"Overall variance of interarrival times = {overall_variance:.4f}s^2, expected = {expected_variance_s2:.4f}s^2")
        self.assertAlmostEqual(overall_variance, expected_variance_s2, delta=0.1 * expected_variance_s2, msg=f"Overall variance of interarrival times {overall_variance} differs from expected {expected_variance_s2}")

    def test_initialization_computation(self):

        # Given some reference logs, make sure that the recomputed initial opinions match what one would expect
        reference_dir = Path(__file__).parent.resolve() / "test_data_fixed"
        reference_run_dir = reference_dir / "train" / "run_1"
        output_dir = Path(__file__).parent.resolve() / "test_data_fixed_out"
        output_run_dir = output_dir / "train" / "run_1"

        rng1 = random.Random(1234)
        rng2 = random.Random(1234)

        ref_statement = "The sky is blue."

        # define a sequence
        init_opinions = {
            "agent_1": "The sky is red.",
            "agent_2": "The sky is blue",
            "agent_3": "The sky is green.",
        }

        embedding_analyzer = EmbeddingAnalyzerSync(topic=ref_statement)


        # get the scores of the initial opinions against the reference statement
        expected_scores = {}
        for agent_id, opinion in init_opinions.items():
            scored = embedding_analyzer.embed_and_score(opinion)
            expected_scores[agent_id] = scored['stance_score']


        clean_log_times_and_init(experiment_dir=reference_dir, out_dir=output_dir, poisson_lambda=RATE_PER_S, rng=rng1,
                                 embedding_analyzer=embedding_analyzer, view_init_opinions_for_debug=False)
        # load connection_graph.json, static_init.json, and run_manifest.json from the output_run_dir
        #  these files should be identical to the input files in reference_run_dir
        for fn in ['connection_graph.json', 'static_init.json', 'run_manifest.json']:
            with open(reference_run_dir / fn, 'r') as f1, open(output_run_dir / fn, 'r') as f2:
                ref_json = json.load(f1)
                out_json = json.load(f2)
                self.assertEqual(ref_json, out_json, f"File {fn} in output_run_dir does not match reference_run_dir")

        # load up the initial_stance_map.json
        with open(output_run_dir / 'initial_stance_map.json', 'r') as f:
            init_stance_map = json.load(f)

        # check that the recomputed scores match the expected scores
        for agent_id, expected_score in expected_scores.items():
            self.assertIn(agent_id, init_stance_map, f"Agent {agent_id} not found in initial_stance_map")
            recomputed_score = init_stance_map[agent_id]['recomputed']
            self.assertAlmostEqual(recomputed_score, expected_score, delta=1e-6, msg=f"Recomputed score for agent {agent_id} does not match expected score")

        # also, check that the logged message times match recomputed values from rng2
        with open(output_run_dir / 'messages_with_alignment.jsonl', 'r') as f:
            messages_with_alignment = [json.loads(line) for line in f]

        # recompute the expected t_s values using rng2
        expected_t_s_values = []
        current_time = 0.0

        for _ in range(len(messages_with_alignment)):
            expected_t_s_values.append(current_time)
            interarrival_time = rng2.expovariate(RATE_PER_S)
            current_time += interarrival_time

        # check that the t_s values in messages_with_alignment match the expected values
        for i, row in enumerate(messages_with_alignment):
            self.assertAlmostEqual(row['time']['t_s'], expected_t_s_values[i], delta=1e-6, msg=f"t_s value for message {i} does not match expected value")
            # also, check that t_ms values are consistent with t_s values
            expected_t_ms = (row['time']['t_s'] * 1000)
            self.assertEqual(row['time']['t_ms'], expected_t_ms, f"t_ms value for message {i} does not match expected value")

if __name__ == '__main__':
    unittest.main()