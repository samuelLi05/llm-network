"""
    Code to evaluate the fit between the stance scores used for opinion modeling and human ratings
"""

from pathlib import Path
import csv
import sys
import argparse
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
# ensure project imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.models.data_prep import _load_jsonl


# These three parameters are fixed for the rating experiment
RATING_DIR = ROOT / 'verification' / 'sampled_messages_with_test_ratings'
TEMPLATE = 'sampled_message_survey_'

# LLM and topic details for rating experiment
LLM = "gemma3"
TOPIC = "climate"
TOPIC_STATEMENT = "Climate Change is caused by human activities"
RUN_LOCATION = ROOT / 'modeling' / 'runs' / LLM / TOPIC / 'train'

# Threshold (in %) for stance recomputation errors
STANCE_DELTA_THRESHOLD = 0.0


def get_named_csvs(template: str, dir: Path):

    f_names = [f.name for f in dir.iterdir() if f.name.startswith(template)]

    annotators = [s.removeprefix(template) for s in f_names]
    annotators = [s.removesuffix('.csv') for s in annotators]

    return f_names, annotators

def load_ratings_and_stance_scores(rating_file_name,
                                   embedding_analyzer = None):

    # Loads the given file as a csv, and looks up the stance scores from the logs. Additionally, does some validation
    #   - Make sure ratings match between numeric and star-based scores
    #   - Make sure text-look up via directory name matches csv-text
    #   - Make sure that stance look up via directory name matches stance score recomputation (optional; run by setting embedding_analyzer)

    data_list = []

    with open(rating_file_name, "r") as f:
        data = csv.DictReader(f)
        
        for row in data:
            data_list.append(row)

    data_list_new = []
    for row in data_list:
        # validate that the rating fields are correct
        rating_star = row['rating']
        rating_star_count = rating_star.count('★')

        rating_num = int(row['rating_num'])

        if not rating_num == rating_star_count:
            raise ValueError("Mismatch in rating sources")
        if not rating_num in list(range(1,6)):
            raise ValueError("Invalid rating value")
        
        # pull the stance score from the message logs
        run_name = row['dir_name']
        run_fn = RUN_LOCATION / run_name / 'messages_with_alignment.jsonl'
        msg_list = _load_jsonl(run_fn)

        # Step 1. validate that the message in messages_with_alignment matches the saved message in the csv        
        index = int(row['index'])
        msg_dict_ldd = msg_list[index]
        if not row['text'] == msg_dict_ldd['message']:
            raise ValueError("Csv message does not match loaded jsonl message")

        # Step 2. re-embed the score to check
        if not embedding_analyzer is None:

            # recompute the score
            rescored = embedding_analyzer.embed_and_score(row['text'])
            msg_opinion_recomputed = rescored['stance_score']
            
            msg_opinion_logged = msg_dict_ldd["published"]["stance_score"]

            pct_diff = None
            if msg_opinion_logged == 0:
                if msg_opinion_recomputed == 0:
                    pct_diff = 0
                else:
                    pct_diff = np.inf
            else:
                pct_diff = 100 * abs(msg_opinion_recomputed - msg_opinion_logged) / abs(msg_opinion_logged) 

            if pct_diff > STANCE_DELTA_THRESHOLD:
                raise ValueError(f"Logged stance score was {msg_opinion_logged} " 
                                 f"but recomputed stance score is {msg_opinion_recomputed}. " 
                                 f"Delta is too high ({pct_diff} %)")
            
        # Step 3. save the logged stance score
        new_row = row.copy()
        new_row['stance_score'] = msg_dict_ldd["published"]["stance_score"]

        data_list_new.append(new_row)

    return data_list_new

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate the fit between human ratings and embedding-based ratings")
    parser.add_argument('--recompute_stances', action='store_true', help='If set, recompute the stances for the loaded messages, and check that they match what we have in the logs.')
    args = parser.parse_args()

    if args.recompute_stances:
        from modeling.stance_analysis_for_modeling.embedding_analyzer_sync import EmbeddingAnalyzerSync
        embedding_analyzer = EmbeddingAnalyzerSync(topic = TOPIC_STATEMENT)

    f_names, annotators = get_named_csvs(TEMPLATE, RATING_DIR)

    for f_name in f_names:
        load_ratings_and_stance_scores(RATING_DIR / f_name, embedding_analyzer=embedding_analyzer)

    print(f"Found annotator names : {annotators}")
    
