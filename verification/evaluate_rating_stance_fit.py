"""
    Code to evaluate the fit between the stance scores used for opinion modeling and human ratings
"""

from pathlib import Path
import csv
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
# ensure project imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.models.data_prep import _load_jsonl

import rpy2

import rpy2.robjects as ro
from rpy2.robjects.packages import importr, isinstalled
from rpy2.robjects.vectors import IntVector
from rpy2.robjects import pandas2ri


# These three parameters are fixed for the rating experiment
RATING_DIR = ROOT / 'verification' / 'sampled_messages_with_ratings'
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
        rating_num = int(row['rating_num'])
        if 'rating' in row.keys():

            rating_star = row['rating']
            rating_star_count = rating_star.count('★')

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
        new_row['rating_num'] = int(row['rating_num'])
        new_row['index'] = int(row['index'])

        data_list_new.append(new_row)

    return data_list_new

def get_r_objects_for_processing(data):
    # data: list of dicts with the fields 'rating_num' and 'stance_score'

    df = pd.DataFrame(data)

    with (ro.default_converter + pandas2ri.converter).context():

        r_df_from_pd = ro.conversion.get_conversion().py2rpy(df)

    r_rating = ro.r['ordered'](r_df_from_pd.rx2('rating_num'), levels = IntVector([1,2,3,4,5]))
    r_stance_score = r_df_from_pd.rx2('stance_score')

    # validation: check that nothing broke when converting to R-vectors
    for i, item in enumerate(data):
        fail = not ((item['rating_num'] == r_rating[i]) and (item['stance_score'] == r_stance_score[i]))
        if fail:
            raise RuntimeError("Error in parsing for r")

    return r_rating, r_stance_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate the fit between human ratings and embedding-based ratings")
    parser.add_argument('--recompute_stances', action='store_true', help='If set, recompute the stances for the loaded messages, and check that they match what we have in the logs.')
    parser.add_argument('--visualize', action='store_true', help='If set, Plot a scatter of the stance score and ratings.')
    args = parser.parse_args()

    embedding_analyzer = None
    if args.recompute_stances:
        from modeling.stance_analysis_for_modeling.embedding_analyzer_sync import EmbeddingAnalyzerSync
        embedding_analyzer = EmbeddingAnalyzerSync(topic = TOPIC_STATEMENT)

    f_names, annotators = get_named_csvs(TEMPLATE, RATING_DIR)

    if args.visualize:
        fig, axes = plt.subplots(2, len(f_names),
                                 figsize=(5 * len(f_names), 8),
                                 sharey='row')
        if len(f_names) == 1:
            axes = axes.reshape(-1, 1)
        axes[0, 0].set_ylabel('Frequency')
        axes[1, 0].set_ylabel('Human rating')
    for i, f_name in enumerate(f_names):
        data_with_stances = load_ratings_and_stance_scores(RATING_DIR / f_name, embedding_analyzer=embedding_analyzer)

        r_rating, r_stance_score = get_r_objects_for_processing(data_with_stances)

        polycor = importr('polycor')
        r_corr = polycor.polyserial(r_stance_score, r_rating, threshold=True, ML = True)

        print(f"For annotator \'{annotators[i]}\' with file \'{f_name}\', polyserial correlation is {r_corr}")
        if args.visualize:
            axes[0,i].hist(r_rating)
            axes[0,i].set_title(f"Rating distribution for {annotators[i]}")
            axes[0,i].set_xlabel("Human rating")

        if args.visualize:
            corr_val = round(float(r_corr[1][0]), 3)
            axes[1, i].scatter(r_stance_score, r_rating, alpha=0.6, s=20)
            axes[1, i].set_title(f"Polyserial correlation ρ = {corr_val}", fontsize=10)
            axes[1, i].set_xlabel("Stance score")



    if args.visualize:
        fig.tight_layout()
        plt.show()

    print(f"Found annotator names : {annotators}")
    
