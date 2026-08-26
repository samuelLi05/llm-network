"""
(Partial) unit tests for checking log_correcter.py functionality when we re-evaluate
the stance of all messages in all runs.

In particular, this file focusses on checking that, when we reembed, all the data is in the expected format
"""

import unittest
import random
from pathlib import Path
import json
import sys
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
# ensure project imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



from modeling.log_correcter import clean_log_times_and_init
from modeling.stance_analysis_for_modeling.embedding_analyzer_sync import EmbeddingAnalyzerSync

POISSON_MEAN = 15.0
TIME_UNIT_MS = 8000
RATE_PER_S = (POISSON_MEAN / (TIME_UNIT_MS / 1000.0))  

class TestLogCorrectionFullReEval(unittest.TestCase):

    def test_log_copying(self):
        ORIGINAL_LOGS_DIR = Path(__file__).resolve().parents[2] / 'modeling' / 'runs'
        NEW_LOGS_DIR = Path(__file__).resolve().parents[2] / 'modeling' / 'runs_rescored'

        # Check that the files in to_copy are the same for
        #  each run in the original and new logs directories
        to_copy = ['connection_graph.json', 'run_manifest.json', 'static_init.json']

        # additionally, check that, for the messages_with_alignment.jsonl files, all fields in
        #   the new file are present in the original file, and that values match (up to dictionaries, where we nest the check)

        # finally, check that there is an initial_stance_map file with the right shape

        total_run_count = 0
        for llm_dir in ORIGINAL_LOGS_DIR.iterdir():
            if not llm_dir.is_dir():
                raise ValueError(f"Expected {llm_dir} to be a directory")
            for topic_dir in llm_dir.iterdir():
                if not topic_dir.is_dir():
                    raise ValueError(f"Expected {topic_dir} to be a directory")
                
                print("Checking run set:", llm_dir.name, topic_dir.name)

                for split_dir in topic_dir.iterdir():
                    if not split_dir.name in ['train', 'test']:
                        raise ValueError(f"Expected {split_dir} to be a directory named 'train' or 'test'")
                    for run_dir in tqdm(split_dir.iterdir(), desc=f"Checking runs for {llm_dir.name}/{topic_dir.name}/{split_dir.name}"):
                        if not run_dir.is_dir():
                            raise ValueError(f"Expected {run_dir} to be a directory")

                        orig_run_dir = run_dir
                        new_run_dir = NEW_LOGS_DIR / llm_dir.name / topic_dir.name / split_dir.name / run_dir.name
                        if not new_run_dir.exists():
                            raise ValueError(f"Expected {new_run_dir} to exist")

                        # STAGE 1: check that the files in to_copy are the same
                        for fn in to_copy:
                            orig_file = orig_run_dir / fn
                            new_file = new_run_dir / fn

                            # check that the file exists in both directories
                            self.assertTrue(orig_file.exists(), f"Expected {orig_file} to exist")
                            self.assertTrue(new_file.exists(), f"Expected {new_file} to exist")

                            # check that the files are the same
                            with open(orig_file, 'r') as f1:
                                orig_data = f1.read()
                            
                            with open(new_file, 'r') as f2:
                                new_data = f2.read()

                            self.assertEqual(orig_data, new_data, f"Files {orig_file} and {new_file} do not match")

                        # STAGE 2: 
                        original_jsonl_file = orig_run_dir / 'messages_with_alignment.jsonl'
                        new_jsonl_file = new_run_dir / 'messages_with_alignment.jsonl'

                        original_rows = []
                        with original_jsonl_file.open('r') as f:
                            for line in f:
                                s = line.strip()
                                if s:
                                    original_rows.append(json.loads(s))
                                else:
                                    raise ValueError(f"Empty line in {original_jsonl_file}")
                        new_rows = []
                        with new_jsonl_file.open('r') as f:
                            for line in f:
                                s = line.strip()
                                if s:
                                    new_rows.append(json.loads(s))
                                else:
                                    raise ValueError(f"Empty line in {new_jsonl_file}")

                        # assert that the number of rows is the same
                        self.assertEqual(len(original_rows), len(new_rows), f"Number of rows in {original_jsonl_file} and {new_jsonl_file} do not match")

                        # for each row in the corrected file, check that all fields are present in the original row
                        #  and that, if the key is a dictionary, the values match (recursively)

                        # exceptions:
                        #  - 'published' field: the 'stance_score' may differ 
                        #  - 'published' field: the 'old_stance_score' will only be present in the new file
                        for i, (orig_row, new_row) in enumerate(zip(original_rows, new_rows)):
                            for key in new_row:
                                self.assertIn(key, orig_row, f"Key {key} in new row {i} not found in original row")
                                if isinstance(new_row[key], dict):
                                    if key == 'published':
                                        # check that all keys in new_row['published'] are in orig_row['published']
                                        for subkey in new_row[key]:
                                            if not(subkey == 'old_stance_score'):
                                                self.assertIn(subkey, orig_row[key], f"Subkey {subkey} in new row {i} not found in original row")
                                                if not (subkey == 'stance_score'):
                                                    # check that the values match
                                                    self.assertEqual(new_row[key][subkey], orig_row[key][subkey], f"Value for subkey {subkey} in new row {i} does not match original row")
                                    else:
                                        # check that all keys in new_row[key] are in orig_row[key]
                                        for subkey in new_row[key]:
                                            self.assertIn(subkey, orig_row[key], f"Subkey {subkey} in new row {i} not found in original row")
                                            # check that the values match
                                            self.assertEqual(new_row[key][subkey], orig_row[key][subkey], f"Value for subkey {subkey} in new row {i} does not match original row")
                                else:
                                    # check that the values match
                                    self.assertEqual(new_row[key], orig_row[key], f"Value for key {key} in new row {i} does not match original row")

                        # finally, check that an initial_stance_map.json file exists
                        # and that it
                        #  - has keys agent_1 ... agent_30
                        #  - each value is a dict with keys ['sim_logged', 'recomputed', 'baseline_opinion_str']
                        init_stance_map_file = new_run_dir / 'initial_stance_map.json'
                        self.assertTrue(init_stance_map_file.exists(), f"Expected {init_stance_map_file} to exist")
                        with init_stance_map_file.open('r') as f:
                            init_stance_map = json.load(f)
                        self.assertEqual(set(init_stance_map.keys()), {f'agent_{i}' for i in range(1, 31)}, f"Keys in {init_stance_map_file} do not match expected agent keys")

                        for agent_key, agent_value in init_stance_map.items():
                            self.assertIsInstance(agent_value, dict, f"Value for {agent_key} in {init_stance_map_file} is not a dict")
                            self.assertEqual(set(agent_value.keys()), {'sim_logged', 'recomputed', 'baseline_opinion_str'}, f"Keys in value for {agent_key} in {init_stance_map_file} do not match expected keys")
                        total_run_count += 1

        self.assertEqual(total_run_count, 288, f"Expected 288 runs in total, but found {total_run_count}")

    # Repeat for rq3 and rq4
    def test_log_copying_rq3(self):

        ORIGINAL_LOGS_DIR = Path(__file__).resolve().parents[2] / 'modeling' / 'runs_fg_vs_adj_cr'
        NEW_LOGS_DIR = Path(__file__).resolve().parents[2] / 'modeling' / 'runs_fg_vs_adj_cr_rescored'

        # Check that the files in to_copy are the same for
        #  each run in the original and new logs directories
        to_copy = ['connection_graph.json', 'run_manifest.json', 'static_init.json']

        # additionally, check that, for the messages_with_alignment.jsonl files, all fields in
        #   the new file are present in the original file, and that values match (up to dictionaries, where we nest the check)

        # finally, check that there is an initial_stance_map file with the right shape

        total_run_count = 0
        for llm_dir in ORIGINAL_LOGS_DIR.iterdir():
            if not llm_dir.is_dir():
                raise ValueError(f"Expected {llm_dir} to be a directory")
            for topic_dir in llm_dir.iterdir():
                if not topic_dir.is_dir():
                    raise ValueError(f"Expected {topic_dir} to be a directory")

                for split_dir in topic_dir.iterdir():
                    if not split_dir.name in ['train', 'test']:
                        raise ValueError(f"Expected {split_dir} to be a directory named 'train' or 'test'")
                    for run_dir in tqdm(split_dir.iterdir(), desc=f"Checking runs for {llm_dir.name}/{topic_dir.name}/{split_dir.name}"):
                        if not run_dir.is_dir():
                            raise ValueError(f"Expected {run_dir} to be a directory")

                        orig_run_dir = run_dir
                        new_run_dir = NEW_LOGS_DIR / llm_dir.name / topic_dir.name / split_dir.name / run_dir.name
                        if not new_run_dir.exists():
                            raise ValueError(f"Expected {new_run_dir} to exist")

                        # STAGE 1: check that the files in to_copy are the same
                        for fn in to_copy:
                            orig_file = orig_run_dir / fn
                            new_file = new_run_dir / fn

                            # check that the file exists in both directories
                            self.assertTrue(orig_file.exists(), f"Expected {orig_file} to exist")
                            self.assertTrue(new_file.exists(), f"Expected {new_file} to exist")

                            # check that the files are the same
                            with open(orig_file, 'r') as f1:
                                orig_data = f1.read()
                            
                            with open(new_file, 'r') as f2:
                                new_data = f2.read()

                            self.assertEqual(orig_data, new_data, f"Files {orig_file} and {new_file} do not match")


                        # STAGE 2: Check messages_with_alignment.jsonl
                        original_jsonl_file = orig_run_dir / 'messages_with_alignment.jsonl'
                        new_jsonl_file = new_run_dir / 'messages_with_alignment.jsonl'

                        original_rows = []
                        with original_jsonl_file.open('r') as f:
                            for line in f:
                                s = line.strip()
                                if s:
                                    original_rows.append(json.loads(s))
                                else:
                                    raise ValueError(f"Empty line in {original_jsonl_file}")
                        new_rows = []
                        with new_jsonl_file.open('r') as f:
                            for line in f:
                                s = line.strip()
                                if s:
                                    new_rows.append(json.loads(s))
                                else:
                                    raise ValueError(f"Empty line in {new_jsonl_file}")
                                
                        # assert that the number of rows is the same
                        self.assertEqual(len(original_rows), len(new_rows), f"Number of rows in {original_jsonl_file} and {new_jsonl_file} do not match")

                         # exceptions:
                        #  - 'published' field: the 'stance_score' may differ 
                        #  - 'published' field: the 'old_stance_score' will only be present in the new file
                        for i, (orig_row, new_row) in enumerate(zip(original_rows, new_rows)):
                            for key in new_row:
                                self.assertIn(key, orig_row, f"Key {key} in new row {i} not found in original row")
                                if isinstance(new_row[key], dict):
                                    if key == 'published':
                                        # check that all keys in new_row['published'] are in orig_row['published']
                                        for subkey in new_row[key]:
                                            if not(subkey == 'old_stance_score'):
                                                self.assertIn(subkey, orig_row[key], f"Subkey {subkey} in new row {i} not found in original row")
                                                if not (subkey == 'stance_score'):
                                                    # check that the values match
                                                    self.assertEqual(new_row[key][subkey], orig_row[key][subkey], f"Value for subkey {subkey} in new row {i} does not match original row")
                                    else:
                                        # check that all keys in new_row[key] are in orig_row[key]
                                        for subkey in new_row[key]:
                                            self.assertIn(subkey, orig_row[key], f"Subkey {subkey} in new row {i} not found in original row")
                                            # check that the values match
                                            self.assertEqual(new_row[key][subkey], orig_row[key][subkey], f"Value for subkey {subkey} in new row {i} does not match original row")
                                else:
                                    # check that the values match
                                    self.assertEqual(new_row[key], orig_row[key], f"Value for key {key} in new row {i} does not match original row")

                        # finally, check that an initial_stance_map.json file exists
                        # and that it
                        #  - has keys agent_1 ... agent_30
                        #  - each value is a dict with keys ['sim_logged', 'recomputed', 'baseline_opinion_str']
                        init_stance_map_file = new_run_dir / 'initial_stance_map.json'
                        self.assertTrue(init_stance_map_file.exists(), f"Expected {init_stance_map_file} to exist")
                        with init_stance_map_file.open('r') as f:
                            init_stance_map = json.load(f)
                        self.assertEqual(set(init_stance_map.keys()), {f'agent_{i}' for i in range(1, 31)}, f"Keys in {init_stance_map_file} do not match expected agent keys")

                        for agent_key, agent_value in init_stance_map.items():
                            self.assertIsInstance(agent_value, dict, f"Value for {agent_key} in {init_stance_map_file} is not a dict")
                            self.assertEqual(set(agent_value.keys()), {'sim_logged', 'recomputed', 'baseline_opinion_str'}, f"Keys in value for {agent_key} in {init_stance_map_file} do not match expected keys")
                        total_run_count += 1

        self.assertEqual(total_run_count, 82, f"Expected 82 runs in total, but found {total_run_count}")
    def test_log_copying_rq4(self):
        # varied size runs;

        ORIGINAL_LOGS_DIR = Path(__file__).resolve().parents[2] / 'modeling' / 'runs_varied_size' / 'llama3.1' / 'vaccines'
        NEW_LOGS_DIR = Path(__file__).resolve().parents[2] / 'modeling' / 'runs_varied_size_rescored' / 'llama3.1' / 'vaccines'

        for ns_dir in ORIGINAL_LOGS_DIR.iterdir():
            if not ns_dir.is_dir():
                raise ValueError(f"Expected only directories in {ORIGINAL_LOGS_DIR}, but found {ns_dir}.")
            out_dir = NEW_LOGS_DIR / ns_dir.name
            if not out_dir.exists():
                raise ValueError(f"Expected output directory {out_dir} to exist")

            # ns_dir.name will be of the form n_{num_agents}
            num_agents = int(ns_dir.name.split('_')[1])

            for split_dir in ns_dir.iterdir():
                if not split_dir.name in ['train', 'test']:
                    raise ValueError(f"Expected {split_dir} to be a directory named 'train' or 'test'")
                for run_dir in tqdm(split_dir.iterdir(), desc=f"Checking runs for {ns_dir.name}/{split_dir.name}"):
                    if not run_dir.is_dir():
                        raise ValueError(f"Expected {run_dir} to be a directory")

                    orig_run_dir = run_dir
                    new_run_dir = out_dir / split_dir.name / run_dir.name
                    if not new_run_dir.exists():
                        raise ValueError(f"Expected {new_run_dir} to exist")

                    # STAGE 1: check that the files in to_copy are the same
                    for fn in ['connection_graph.json', 'run_manifest.json', 'static_init.json']:
                        orig_file = orig_run_dir / fn
                        new_file = new_run_dir / fn

                        # check that the file exists in both directories
                        self.assertTrue(orig_file.exists(), f"Expected {orig_file} to exist")
                        self.assertTrue(new_file.exists(), f"Expected {new_file} to exist")

                        # check that the files are the same
                        with open(orig_file, 'r') as f1:
                            orig_data = f1.read()
                        
                        with open(new_file, 'r') as f2:
                            new_data = f2.read()

                        self.assertEqual(orig_data, new_data, f"Files {orig_file} and {new_file} do not match")

                    # STAGE 2: check that the messages_with_alignment.jsonl files have the same number of rows
                    #  + mostly similar checks as above
                    original_jsonl_file = orig_run_dir / 'messages_with_alignment.jsonl'
                    new_jsonl_file = new_run_dir / 'messages_with_alignment.jsonl'

                    original_rows = []
                    with original_jsonl_file.open('r') as f:
                        for line in f:
                            s = line.strip()
                            if s:
                                original_rows.append(json.loads(s))
                            else:
                                raise ValueError(f"Empty line in {original_jsonl_file}")
                    new_rows = []
                    with new_jsonl_file.open('r') as f:
                        for line in f:
                            s = line.strip()
                            if s:
                                new_rows.append(json.loads(s))
                            else:
                                raise ValueError(f"Empty line in {new_jsonl_file}")

                    # assert that the number of rows is the same
                    self.assertEqual(len(original_rows), len(new_rows), f"Number of rows in {original_jsonl_file} and {new_jsonl_file} do not match")

                    # for each row in the corrected file, check that all fields are present in the original row
                    #  and that, if the key is a dictionary, the values match (recursively)

                    # exceptions:
                    #  - 'published' field: the 'stance_score' may differ 
                    #  - 'published' field: the 'old_stance_score' will only be present in the new file
                    for i, (orig_row, new_row) in enumerate(zip(original_rows, new_rows)):
                        for key in new_row:
                            self.assertIn(key, orig_row, f"Key {key} in new row {i} not found in original row")
                            if isinstance(new_row[key], dict):
                                if key == 'published':
                                    # check that all keys in new_row['published'] are in orig_row['published']
                                    for subkey in new_row[key]:
                                        if not(subkey == 'old_stance_score'):
                                            self.assertIn(subkey, orig_row[key], f"Subkey {subkey} in new row {i} not found in original row")
                                            if not (subkey == 'stance_score'):
                                                # check that the values match
                                                self.assertEqual(new_row[key][subkey], orig_row[key][subkey], f"Value for subkey {subkey} in new row {i} does not match original row")
                                            
                                else:
                                    # check that all keys in new_row[key] are in orig_row[key]
                                    for subkey in new_row[key]:
                                        self.assertIn(subkey, orig_row[key], f"Subkey {subkey} in new row {i} not found in original row")
                                        # check that the values match
                                        # (exception if we're looking at t_s or t_ms keys in the 60 agent run )
                                        if not (subkey in ['t_s', 't_ms'] and num_agents == 60):
                                            self.assertEqual(new_row[key][subkey], orig_row[key][subkey], f"Value for subkey {subkey} in new row {i} does not match original row")
                            else:
                                # check that the values match
                                self.assertEqual(new_row[key], orig_row[key], f"Value for key {key} in new row {i} does not match original row")

                    # finally, check that an initial_stance_map.json file exists
                    # and that it
                    #  - has keys agent_1 ... agent_{num_agents}
                    #  - each value is a dict with keys ['sim_logged', 'recomputed', 'baseline_opinion_str']
                    init_stance_map_file = new_run_dir / 'initial_stance_map.json'
                    self.assertTrue(init_stance_map_file.exists(), f"Expected {init_stance_map_file} to exist")
                    with init_stance_map_file.open('r') as f:
                        init_stance_map = json.load(f)
                    self.assertEqual(set(init_stance_map.keys()), {f'agent_{i}' for i in range(1, num_agents + 1)}, f"Keys in {init_stance_map_file} do not match expected agent keys")

                    for agent_key, agent_value in init_stance_map.items():
                        self.assertIsInstance(agent_value, dict, f"Value for {agent_key} in {init_stance_map_file} is not a dict")
                        self.assertEqual(set(agent_value.keys()), {'sim_logged', 'recomputed', 'baseline_opinion_str'}, f"Keys in value for {agent_key} in {init_stance_map_file} do not match expected keys")

                    

    def test_rq4_rq2_match(self):
        # check that the rescored files at
        #  - modeling / runs_varied_size_rescored / llama3.1 / vaccines / n_30
        #  - modeling / runs_rescored / llama3.1 / vaccines
        # are the same after rescoring

        # in particular, the following files should match exactly
        exact_match = ['connection_graph.json', 'run_manifest.json', 'static_init.json']

        # meanwhile, the messages_with_alignment.jsonl files + initial_stance_map.json
        #  should satisfy almost equality for the stance scores, but should otherwise match exactly

        RQ2_LOGS_DIR = Path(__file__).resolve().parents[2] / 'modeling' / 'runs_rescored' / 'llama3.1' / 'vaccines'
        RQ4_LOGS_DIR = Path(__file__).resolve().parents[2] / 'modeling' / 'runs_varied_size_rescored' / 'llama3.1' / 'vaccines' / 'n_30'

        num_checked_runs = 0
        for split_dir in RQ2_LOGS_DIR.iterdir():
            if not split_dir.name in ['train', 'test']:
                raise ValueError(f"Expected {split_dir} to be a directory named 'train' or 'test'")
            for run_dir in tqdm(split_dir.iterdir(), desc=f"Checking runs for {split_dir.name}"):
                if not run_dir.is_dir():
                    raise ValueError(f"Expected {run_dir} to be a directory")

                rq2_run_dir = run_dir
                rq4_run_dir = RQ4_LOGS_DIR / split_dir.name / run_dir.name
                if not rq4_run_dir.exists():
                    raise ValueError(f"Expected {rq4_run_dir} to exist")

                # check exact match files
                for fn in exact_match:
                    rq2_file = rq2_run_dir / fn
                    rq4_file = rq4_run_dir / fn

                    self.assertTrue(rq2_file.exists(), f"Expected {rq2_file} to exist")
                    self.assertTrue(rq4_file.exists(), f"Expected {rq4_file} to exist")

                    with open(rq2_file, 'r') as f1:
                        rq2_data = f1.read()
                    
                    with open(rq4_file, 'r') as f2:
                        rq4_data = f2.read()

                    self.assertEqual(rq2_data, rq4_data, f"Files {rq2_file} and {rq4_file} do not match")

                # chcek messages_with_alignment.jsonl files
                rq2_jsonl_file = rq2_run_dir / 'messages_with_alignment.jsonl'
                rq4_jsonl_file = rq4_run_dir / 'messages_with_alignment.jsonl'

                rq2_rows = []
                with rq2_jsonl_file.open('r') as f:
                    for line in f:
                        s = line.strip()
                        if s:
                            rq2_rows.append(json.loads(s))
                        else:
                            raise ValueError(f"Empty line in {rq2_jsonl_file}")
                rq4_rows = []
                with rq4_jsonl_file.open('r') as f:
                    for line in f:
                        s = line.strip()
                        if s:
                            rq4_rows.append(json.loads(s))
                        else:
                            raise ValueError(f"Empty line in {rq4_jsonl_file}")
                        
                # assert that the number of rows is the same
                self.assertEqual(len(rq2_rows), len(rq4_rows), f"Number of rows in {rq2_jsonl_file} and {rq4_jsonl_file} do not match")

                # check that the rows match, with almost equality for the stance scores
                for i, (rq2_row, rq4_row) in enumerate(zip(rq2_rows, rq4_rows)):
                    # check that the keys match
                    self.assertEqual(set(rq2_row.keys()), set(rq4_row.keys()))

                    for key in rq4_row:
                        if isinstance(rq4_row[key], dict):
                            if key == 'published':
                                # check that all keys in rq4_row['published'] are in rq2_row['published']
                                for subkey in rq4_row[key]:
                                    if subkey == 'stance_score' or subkey == 'old_stance_score':
                                        # check that the values are almost equal
                                        self.assertAlmostEqual(rq4_row[key][subkey], rq2_row[key][subkey], places=5, msg=f"Value for subkey {subkey} in rq4 row {i} does not match rq2 row")
                                    else:
                                        self.assertEqual(rq4_row[key][subkey], rq2_row[key][subkey], f"Value for subkey {subkey} in rq4 row {i} does not match rq2 row")
                            else:
                                # check that all keys in rq4_row[key] are in rq2_row[key]
                                self.assertEqual(set(rq4_row[key].keys()), set(rq2_row[key].keys()), f"Keys in rq4 row {i} for key {key} do not match rq2 row")
                                for subkey in rq4_row[key]:
                                    # check that the values match
                                    self.assertEqual(rq4_row[key][subkey], rq2_row[key][subkey], f"Value for subkey {subkey} in rq4 row {i} does not match rq2 row")
                        else:
                            self.assertEqual(rq4_row[key], rq2_row[key], f"Value for key {key} in rq4 row {i} does not match rq2 row")

                init_stance_map_rq2_file = rq2_run_dir / 'initial_stance_map.json'
                init_stance_map_rq4_file = rq4_run_dir / 'initial_stance_map.json'

                self.assertTrue(init_stance_map_rq2_file.exists(), f"Expected {init_stance_map_rq2_file} to exist")
                self.assertTrue(init_stance_map_rq4_file.exists(), f"Expected {init_stance_map_rq4_file} to exist")

                with init_stance_map_rq2_file.open('r') as f:
                    init_stance_map_rq2 = json.load(f)
                with init_stance_map_rq4_file.open('r') as f:
                    init_stance_map_rq4 = json.load(f)

                self.assertEqual(set(init_stance_map_rq2.keys()), set(init_stance_map_rq4.keys()), f"Keys in {init_stance_map_rq2_file} and {init_stance_map_rq4_file} do not match")

                for agent_key, agent_value in init_stance_map_rq2.items():
                    self.assertEqual(set(agent_value.keys()), set(init_stance_map_rq4[agent_key].keys()), f"Keys in value for {agent_key} in {init_stance_map_rq2_file} and {init_stance_map_rq4_file} do not match")
                    for subkey in agent_value:
                        if subkey == 'recomputed' or subkey == 'sim_logged':
                            self.assertAlmostEqual(agent_value[subkey], init_stance_map_rq4[agent_key][subkey], places=5, msg=f"Value for subkey {subkey} in agent {agent_key} does not match between rq2 and rq4")
                        else:
                            self.assertEqual(agent_value[subkey], init_stance_map_rq4[agent_key][subkey], f"Value for subkey {subkey} in agent {agent_key} does not match between rq2 and rq4")
                num_checked_runs += 1

        self.assertEqual(num_checked_runs, 32, f"Expected 32 checked runs, but found {num_checked_runs}")

    def test_init_and_message_computation(self):

        # given some reference logs, make sure that the recomputed initial opinions
        #  and messages match expected

        reference_dir = Path(__file__).parent.resolve() / "test_data_fixed"
        reference_run_dir = reference_dir / "train" / "run_1"
        output_dir = Path(__file__).parent.resolve() / "test_data_fixed_full_reeval_out"
        output_run_dir = output_dir / "train" / "run_1"

        rng1 = random.Random(1234)

        ref_statement = "The sky is blue."

        # define a sequence
        init_opinions = {
            "agent_1": "The sky is red.",
            "agent_2": "The sky is blue",
            "agent_3": "The sky is green.",
        }

        messages = [
            "The sky is red.",
            "The sky is blue.",
            "The sky is green.",
            "I think the sky is red.",
            "I think the sky is blue",
            "I think the sky is green.",
        ]

        message_times_expected_ms = [
            655, 1115, 11126, 11760, 111282, 1111251
        ]

        embedding_analyzer = EmbeddingAnalyzerSync(topic=ref_statement)

        expected_initial_opinion_scores = {}
        for agent_id, opinion in init_opinions.items():
            scored = embedding_analyzer.embed_and_score(opinion)
            expected_initial_opinion_scores[agent_id] = scored['stance_score']

        expected_message_scores = []
        for m in messages:
            scored = embedding_analyzer.embed_and_score(m)
            expected_message_scores.append(scored['stance_score'])

        message_list_out_dict = clean_log_times_and_init(experiment_dir=reference_dir,
                                 out_dir = output_dir,
                                 poisson_lambda=RATE_PER_S,
                                 rng=rng1,
                                 embedding_analyzer=embedding_analyzer,
                                 re_embed_all_messages=True,
                                 skip_time_reassignment=True,
                                 return_out_dict=True)["message_list_out_dict"]
        
        for fn in ['connection_graph.json', 'run_manifest.json', 'static_init.json']:
            with open(reference_run_dir / fn, 'r') as f1:
                ref_json = json.load(f1)
            with open(output_run_dir / fn, 'r') as f2:
                out_json = json.load(f2)
            self.assertEqual(ref_json, out_json, f"File {fn} does not match between reference and output")

        with open(output_run_dir / 'initial_stance_map.json', 'r') as f:
            out_init_stance_map = json.load(f)

        for agent_id, expected_score in expected_initial_opinion_scores.items():
            self.assertIn(agent_id, out_init_stance_map, f"Agent {agent_id} not found in output initial_stance_map")
            self.assertAlmostEqual(out_init_stance_map[agent_id]['recomputed'], expected_score, places=5, msg=f"Recomputed score for {agent_id} does not match expected")

        # also, check that
        #  - logged message times match the original message times
        #  - recomputed message stances match expected value

        with open(output_run_dir / 'messages_with_alignment.jsonl', 'r') as f:
            messages_with_alignment = [json.loads(line) for line in f]

        for i, row in enumerate(messages_with_alignment):
            self.assertEqual(row['time']['t_ms'], message_times_expected_ms[i], f"Message time for message {i} does not match expected")
            self.assertAlmostEqual(row['published']['stance_score'], expected_message_scores[i], places=5, msg=f"Recomputed stance score for message {i} does not match expected")

            print(row['published']['stance_score'], expected_message_scores[i])

        # repeat this check for message_list_out_dict
        #  In particular, at ["train"]["run_1"], should have a list of dicts, where the 
        #   ['time']['t_ms'] and ['published']['stance_score'] match the expected values
        run_messages = message_list_out_dict['train']['run_1']
        for i, row in enumerate(run_messages):
            self.assertEqual(row['time']['t_ms'], message_times_expected_ms[i], f"Message time for message {i} in message_list_out_dict does not match expected")
            self.assertAlmostEqual(row['published']['stance_score'], expected_message_scores[i], places=5, msg=f"Recomputed stance score for message {i} in message_list_out_dict does not match expected")

if __name__ == "__main__":
    unittest.main()