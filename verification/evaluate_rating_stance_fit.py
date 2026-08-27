"""
    Code to evaluate the fit between the stance scores used for opinion modeling and human ratings
"""

from pathlib import Path
import csv
import sys
import argparse
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = False        # use matplotlib's own text renderer, not LaTeX
matplotlib.rcParams['font.family'] = 'serif'      # base text font -> serif (e.g. DejaVu Serif)
matplotlib.rcParams['mathtext.fontset'] = 'cm'     # math text (e.g. in $...$ labels) -> Computer Modern

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
RUN_LOCATION_ALT = ROOT / 'modeling' / 'runs_rescored' / LLM / TOPIC / 'train'

# Threshold (in %) for stance recomputation errors
STANCE_DELTA_THRESHOLD = 0.01

# location for saving joint data with all raters' messages
JOINT_RATING_LOCATION = ROOT / 'verification' / 'joint_rated_messages' / 'joint_rating_data.csv'

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
            raise ValueError("CSV message does not match loaded jsonl message")

        # Step 1.1: check that the different load locations give the same results
        msg_list_alt = _load_jsonl(RUN_LOCATION_ALT / run_name / 'messages_with_alignment.jsonl')
        delta_msg_srcs = abs(msg_list[index]['published']['stance_score'] - msg_list_alt[index]['published']['stance_score'])
        if not delta_msg_srcs < 1e-03:
            raise ValueError(f"Different sources of messages do not match; delta = {delta_msg_srcs}")

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

def merge_rating_data(rating_lists, annotator_names, save_location):

    # validation
    if not len(rating_lists) > 1: 
        raise ValueError(f"rating_lists only contains {len(rating_lists)} sets of rating data")
    for i in range(1, len(rating_lists)):
        if not (len(rating_lists[0]) == len(rating_lists[i])):
            raise ValueError(f"component lists 0 and {i} do not have the same lengths")
        
    joint_list = []
    # jointly iterate over the lists, while validating that requisite fields match
    match_fields = ['stance_score', 'index', 'dir_name', 'text']
    for i in range(len(rating_lists[0])):

        new_row = {}

        # validate that all requisite fields match
        for j in range(1,len(rating_lists)):
            for field_name in match_fields:
                match0 = rating_lists[0][i][field_name] == rating_lists[j][i][field_name]
                if not match0:
                    raise ValueError(f"Rating list 0 and list {j} do not match on field {field_name} at message {i}")

        # build new row
        for field_name in match_fields:
            new_row[field_name] = rating_lists[0][i][field_name]

        # add all annotators' data in fields indicating their names
        for data, rater in zip(rating_lists, annotator_names):
            field_name = f"rating_{rater}"
            new_row[field_name] = data[i]['rating_num']

        joint_list.append(new_row)

    keys = joint_list[0].keys()
    with save_location.open("w", newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(joint_list)

def build_scaled_ratings_vector(r_ratings, thresholds):

    new_rating_list = []

    if not len(thresholds) == 4:
        raise ValueError("Invalid threshold list; should contain four values")
    sorted = True
    for i in range(len(thresholds) - 1):
        if thresholds[i+1] < thresholds[i]:
            sorted = False
    if not sorted:
        raise ValueError("Invalid threshold list; should be sorted")

    rating_to_latent_dict = {}  # Build a mapping from each human-given rating to a latent value
    widths = []

    # map ratings 2,3,4 to the mid-points of the thresholds defined by indices (0,1),(1,2) and (2,3) respectively.
    for i in range(len(thresholds) - 1):
        rating_to_latent_dict[i+2] = (thresholds[i+1] + thresholds[i])/2
        widths.append(thresholds[i+1] - thresholds[i])

    widths = np.array(widths)
    avg_width = np.mean(widths)

    # for ratings 1 and 5 place them at half the average width away from the lowest and highest thresholds respectively.
    rating_to_latent_dict[1] = float(thresholds[0] - avg_width/2)
    rating_to_latent_dict[5] = float(thresholds[3] + avg_width/2)

    for i in range(len(r_ratings)):
        if not r_ratings[i] in [1,2,3,4,5]:
            raise ValueError("Invalid rating value")
        
        new_rating_list.append(rating_to_latent_dict[r_ratings[i]])

    return new_rating_list
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate the fit between human ratings and embedding-based ratings")
    parser.add_argument('--recompute_stances', action='store_true', help='If set, recompute the stances for the loaded messages, and check that they match what we have in the logs.')
    parser.add_argument('--visualize', action='store_true', help='If set, Plot a scatter of the stance score and ratings.')
    parser.add_argument('--scale_rating_axis', action='store_true', help='If set, for y axis of human ratings vs. stance scores, map the human ratings approximately on to the threshold values')
    parser.add_argument('--remove_rater_name', action='store_true', help='If set display raters as rater_i, rather than with their real names')
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
        if args.scale_rating_axis:
            axes[1,0].set_ylabel("Human rating (latent estimate)")
        else:
            axes[1,0].set_ylabel("Human rating")
            axes[1, 0].set_yticks([1, 2, 3, 4, 5])

    data_with_stance_list = []
    for i, f_name in enumerate(f_names):
        data_with_stances = load_ratings_and_stance_scores(RATING_DIR / f_name, embedding_analyzer=embedding_analyzer)

        data_with_stance_list.append(data_with_stances)

        r_rating, r_stance_score = get_r_objects_for_processing(data_with_stances)

        polycor = importr('polycor')
        r_corr = polycor.polyserial(r_stance_score, r_rating, threshold=True, ML = True, std_err = True)

        print(f"For annotator \'{annotators[i]}\' with file \'{f_name}\', polyserial correlation is {r_corr}")
        if args.visualize:
            axes[0,i].hist(r_rating, bins=np.arange(0.5, 6.5, 1), rwidth=0.5)
            if args.remove_rater_name:
                axes[0,i].set_title(f"Rating distribution for Rater {i + 1}")
            else:
                axes[0,i].set_title(f"Rating distribution for {annotators[i]}")
            axes[0,i].set_xlabel("Human rating")
            axes[0,i].set_xticks([1, 2, 3, 4, 5])

        if args.visualize:
            corr_val = round(float(r_corr.rx2('rho')[0]), 3)
            corr_std = round(float(np.sqrt(np.array(r_corr.rx2('var'))[0,0])), 3)
            
            if args.scale_rating_axis:
                rating_var = build_scaled_ratings_vector(r_ratings=r_rating, thresholds=r_corr[2])
            else:
                rating_var = r_rating

            axes[1, i].scatter(r_stance_score, rating_var, alpha=0.6, s=20)
            axes[1, i].set_title(f"Polyserial correlation ρ = {corr_val} ({corr_std})", fontsize=10)
            axes[1, i].set_xlabel("Stance score")

    merge_rating_data(rating_lists=data_with_stance_list, annotator_names=annotators, save_location=JOINT_RATING_LOCATION)

    if args.visualize:
        fig.tight_layout()
        fig.savefig(ROOT / 'verification' / 'rating_stance_fit.png', dpi=300)
        fig.savefig(ROOT / 'verification' / 'rating_stance_fit.svg')
        plt.show()

    print(f"Found annotator names : {annotators}")
    
