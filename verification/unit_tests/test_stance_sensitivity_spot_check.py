"""
Spot-check test script for stance sensitivity. 
For a random sample of the embedded messages, check we can recompute the same stance scores to high precision
"""

import unittest
from pathlib import Path
import sys
import csv
from tqdm import tqdm
import json
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# ensure project imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from modeling.stance_analysis_for_modeling.embedding_analyzer_sync import EmbeddingAnalyzerSync

CHECK_PROBABILITY = 0.05

PROMPT_SENS_PATH = ROOT / 'verification' / 'prompt_sensitivity_results'
RUN_DIR = ROOT / 'modeling' / 'runs'

class TestStanceSensitivitySpotCheck(unittest.TestCase):

    def test_stance_sensitivity_spot_check(self):

        rng = random.Random(1234)

        # iterate over all prompt sensitivity data
        for prompt_sens_path in PROMPT_SENS_PATH.iterdir():

            psp_name = prompt_sens_path.name

            llm_name = psp_name.split('_')[0]
            topic_name = psp_name.split('_')[1]

            with open(prompt_sens_path, "r") as f:
                prompt_sens_dict = json.load(f)

            # assert that all stance lists are the same length
            min_length = 100000
            max_length = 0

            for _, stance_list in prompt_sens_dict.items():
                min_length = min(min_length, len(stance_list))
                max_length = max(max_length, len(stance_list))

            self.assertEqual(min_length, max_length)

            experiment_dir = RUN_DIR / llm_name / topic_name

            # now, iterate over all messages used to generate the stance data, 
            #  and for randomly sampled messages, check that we can recompute the same values
            for topic_sentence, stance_list in prompt_sens_dict.items():

                embedding_anlayzer = EmbeddingAnalyzerSync(topic = topic_sentence)


                counter = 0 # counter to keep an index to track the message

                for split_dir in sorted(experiment_dir.iterdir()):
                    if not split_dir.is_dir():
                        raise ValueError(f'Expected directory, got file: {split_dir}')
                    if not split_dir.name in ['train', 'test']:
                        raise ValueError(f'Unexpected directory name: {split_dir.name}. Expected "train" or "test".')
                    if split_dir.name == 'test':
                        continue
                    for run_dir in tqdm(sorted(split_dir.iterdir()), f"Iterating:  {llm_name}/{topic_name}/{topic_sentence}"):
                        if not run_dir.is_dir():
                            raise ValueError(f'Expected directory, got file: {run_dir}')

                        message_rows = []
                        message_file = run_dir / 'messages_with_alignment.jsonl'
                        with open(message_file, "r") as f:
                            for line in f:
                                message_rows.append(json.loads(line.strip()))

                        for row in message_rows:
                        
                            if rng.random() <= CHECK_PROBABILITY:
                                recomputed_stance_score = embedding_anlayzer.embed_and_score(row["message"])["stance_score"]

                                self.assertAlmostEqual(recomputed_stance_score, stance_list[counter], places = 5)

                            counter += 1
                # at end of all iterations, counter should match stance_list length
                self.assertEqual(counter, len(stance_list))

if __name__ == "__main__":
    unittest.main()