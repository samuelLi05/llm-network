import unittest
from pathlib import Path
import sys
import csv

ROOT = Path(__file__).resolve().parents[1]
# ensure project imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from modeling.models.data_prep import _load_jsonl
from verification.evaluate_rating_stance_fit import get_named_csvs, load_ratings_and_stance_scores

RATING_DIR = ROOT / 'verification' / 'sampled_messages_with_ratings'
TEMPLATE = 'sampled_message_survey_'


SAMPLED_MESSAGE_LOCATION = ROOT / 'verification' / 'sampled_messages' / 'sampled_messages.jsonl'
SAMPLED_MESSAGE_LOCATION_CSV = ROOT / 'verification' / 'sampled_messages' / 'sampled_messages.csv'
LLM = "gemma3"
TOPIC = "climate"

RUN_LOCATION = ROOT / 'modeling' / 'runs' / LLM / TOPIC / 'train'

class TestPostSampling(unittest.TestCase):

    def test_messages_match(self):

        loaded_messages = _load_jsonl(SAMPLED_MESSAGE_LOCATION)

        for msg_dict in loaded_messages:

            msg_ldd = msg_dict['text']

            index = msg_dict['index']
            dir_name = msg_dict['dir_name']

            msg_with_align_fn = RUN_LOCATION / dir_name / 'messages_with_alignment.jsonl'

            self.assertTrue(msg_with_align_fn.exists())

            msg_with_align_file = _load_jsonl(msg_with_align_fn)

            text_ref = msg_with_align_file[index]['message']

            self.assertEqual(text_ref, msg_ldd)

    def test_csv_jsonl_match(self):

        loaded_messages_jsonl = _load_jsonl(SAMPLED_MESSAGE_LOCATION)

        loaded_messages_csv = []

        with open(SAMPLED_MESSAGE_LOCATION_CSV, "r") as f:
            data = csv.DictReader(f)

            for row in data:
                row['index'] = int(row['index'])

                loaded_messages_csv.append(row)

        self.assertEqual(loaded_messages_csv, loaded_messages_jsonl)

    def test_message_match_post_ratings(self):

        # validate that when we load csvs through ratings in the evaluate_rating_stance_fit experiment,
        #  that the messages and stance scores match

        f_names, annotators = get_named_csvs(TEMPLATE, RATING_DIR)

        print(f"Found annotators : {annotators}")

        for f_name in f_names:
            ldd_data = load_ratings_and_stance_scores(RATING_DIR / f_name, embedding_analyzer=None)

            print(f"In file {f_name}, found {len(ldd_data)} rated messages.")

            for msg in ldd_data:

                ldd_text = msg["text"]              # text in list we'll use for processing
                ldd_stance = msg['stance_score']    # stance score we'll use for processing

                # check that these values match the logged data from simulations
                msg_with_align_fn = RUN_LOCATION / msg["dir_name"] / 'messages_with_alignment.jsonl'
                self.assertTrue(msg_with_align_fn.exists())
                msg_with_align_file = _load_jsonl(msg_with_align_fn)

                lookup_text = msg_with_align_file[int(msg['index'])]['message']
                lookup_stance = msg_with_align_file[int(msg['index'])]['published']['stance_score']

                self.assertEqual(ldd_text, lookup_text)
                self.assertEqual(ldd_stance, lookup_stance)


if __name__ == "__main__":
    unittest.main()