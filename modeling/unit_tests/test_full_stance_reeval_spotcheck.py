"""
Spot checks for stance evaluation in saved runs.
Basically, want to check that, if we randomly sample some messages,
and re-embed them, we get the same stance scores back
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
from modeling.stance_analysis_for_modeling.embedding_analyzer_sync import EmbeddingAnalyzerSync

REFERENCE_BASELINE_STATEMENTS = {
    "climate": "Climate Change is caused by human activities",
    "vaccines": "Vaccines cause autism",
    "gun-control": "Gun ownership is a fundamental right"
}

# number of messages to sample per run
NUMBER_SAMPLES = 20
# number of initial opinions to sample per run
NUMBER_INIT_OP_SAMPLES = 5


class TestStanceEvalSpotCheck(unittest.TestCase):

    def validate_llm_topic_pair(self, llm_topic_path, llm_name, topic_name, embedding_analyzer, rng, 
                                tqdm_info_message = ""):
        

        for split_dir in llm_topic_path.iterdir():

            split_name = split_dir.name
            for run_path in tqdm(list(split_dir.iterdir()), desc = f"Spot checking messages for [{llm_name}/{topic_name}/{split_name}] ({tqdm_info_message})"):
                
                # Load up the messages with alignment file
                msg_file = run_path / 'messages_with_alignment.jsonl'
                msg_rows = []
                with open(msg_file, 'r') as f:
                    for line in f:
                        msg_rows.append(json.loads(line.strip()))
                
                msg_rows_to_check = rng.sample(msg_rows, NUMBER_SAMPLES)

                for row in msg_rows_to_check:

                    recomputed_stance_score = embedding_analyzer.embed_and_score(row["message"])["stance_score"]
                    logged_stance_score = row["published"]["stance_score"]
                    
                    self.assertAlmostEqual(recomputed_stance_score, logged_stance_score, places = 5, msg = "Recomputed stance score does not match logged value")


                # Load up the intiial opinion file and do a similar check 
                static_init_fn = run_path / 'static_init.json'
                with open(static_init_fn, "r") as f:
                    static_init_json = json.load(f)

                agent_to_text_map = static_init_json["agent_configs"]

                init_stance_map_fn = run_path / 'initial_stance_map.json'
                with open(init_stance_map_fn, "r") as f:
                    init_stance_map = json.load(f)

                text_map_keys = list(agent_to_text_map.keys())
                init_stance_map_keys = list(init_stance_map.keys())

                num_agents = len(text_map_keys)
                ref_keys = set([f"agent_{i}" for i in range(1, num_agents + 1)])
                
                self.assertEqual(set(text_map_keys), ref_keys)
                self.assertEqual(set(init_stance_map_keys), ref_keys)

                keys_to_check = rng.sample(sorted(ref_keys), NUMBER_INIT_OP_SAMPLES)

                for k in keys_to_check:

                    text_io = agent_to_text_map[k]["stable_perspective_sentence"]
                    recomputed_stance_score_io = embedding_analyzer.embed_and_score(text_io)["stance_score"]

                    logged_stance_score_io = init_stance_map[k]["recomputed"]
                    logged_text_io = init_stance_map[k]["baseline_opinion_str"]

                    self.assertEqual(text_io, logged_text_io)
                    self.assertAlmostEqual(recomputed_stance_score_io, logged_stance_score_io, places = 5, msg = "Recomputed stance score does not match logged value")


    def test_stance_eval_spotcheck_rq2(self):

        NEW_RUNS_DIR = ROOT / 'modeling' / 'runs_rescored'

        ALL_LLMS = sorted([d.name for d in NEW_RUNS_DIR.iterdir() if d.is_dir()])

        rng = random.Random(1234)

        for llm_name in ALL_LLMS:
            llm_path = NEW_RUNS_DIR / llm_name
            topics = sorted([d.name for d in llm_path.iterdir() if d.is_dir()])

            for topic_name in topics:
                topic_path = llm_path / topic_name
                
                embedding_analyzer = EmbeddingAnalyzerSync(topic = REFERENCE_BASELINE_STATEMENTS[topic_name])
                self.validate_llm_topic_pair(topic_path, llm_name, topic_name, embedding_analyzer, rng,
                                            tqdm_info_message = "RQ2")

    def test_stance_eval_spotcheck_rq3(self):

        NEW_RUNS_DIR = ROOT / 'modeling' / 'runs_fg_vs_adj_cr_rescored'

        ALL_LLMS = sorted([d.name for d in NEW_RUNS_DIR.iterdir() if d.is_dir()])

        rng = random.Random(1235)

        for llm_name in ALL_LLMS:
            llm_path = NEW_RUNS_DIR / llm_name
            topics = sorted([d.name for d in llm_path.iterdir() if d.is_dir()])

            for topic_name in topics:
                topic_path = llm_path / topic_name
                
            embedding_analyzer = EmbeddingAnalyzerSync(topic = REFERENCE_BASELINE_STATEMENTS[topic_name])
            self.validate_llm_topic_pair(topic_path, llm_name, topic_name, embedding_analyzer, rng,
                                         tqdm_info_message = "RQ3")
            
    def test_stance_eval_spotcheck_size_rq(self):

        NEW_RUNS_DIR = ROOT / 'modeling' / 'runs_varied_size_rescored'
        LLM = 'llama3.1'
        TOPIC = 'vaccines'

        llm_topic_dir= NEW_RUNS_DIR / LLM / TOPIC

        embedding_analyzer = EmbeddingAnalyzerSync(topic = REFERENCE_BASELINE_STATEMENTS[TOPIC])

        rng = random.Random(1236)
        for experiment_dir in llm_topic_dir.iterdir():
            experiment_name = experiment_dir.name

            self.validate_llm_topic_pair(experiment_dir, LLM, TOPIC, embedding_analyzer, rng, 
                                         tqdm_info_message = f"RQ4 ; {experiment_name}")
            

if __name__ == "__main__":
    unittest.main()
