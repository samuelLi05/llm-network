import unittest
from pathlib import Path
import sys
import csv

ROOT = Path(__file__).resolve().parents[1]
# ensure project imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from modeling.models.data_prep import _load_jsonl

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

if __name__ == "__main__":
    unittest.main()