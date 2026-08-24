
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

from verification.evaluate_rating_stance_fit import get_named_csvs, load_ratings_and_stance_scores, get_r_objects_for_processing

import rpy2

import rpy2.robjects as ro
from rpy2.robjects.packages import importr, isinstalled
from rpy2.robjects.vectors import IntVector
from rpy2.robjects import pandas2ri


RATING_DIR = ROOT / 'verification' / 'sampled_messages_with_ratings'
TEMPLATE = 'sampled_message_survey_'

# LLM and topic details for rating experiment
LLM = "gemma3"
TOPIC = "climate"
RUN_LOCATION = ROOT / 'modeling' / 'runs' / LLM / TOPIC / 'train'


if __name__ == "__main__":

    f_names, annotators = get_named_csvs(TEMPLATE, RATING_DIR)

    num_raters = len(f_names)
    num_combs = num_raters * (num_raters - 1)//2

    fig, axes = plt.subplots(1,num_combs, sharey=True)


    rating_vectors = []

    for i, f_name in enumerate(f_names):
        data_with_stances = load_ratings_and_stance_scores(RATING_DIR / f_name, embedding_analyzer=None)

        r_rating, r_stance_score = get_r_objects_for_processing(data_with_stances)
        rating_vectors.append(r_rating)

    polycor = importr('polycor')
    levels = [1, 2, 3, 4, 5]
    ax_count = 0
    for i in range(num_raters):
        for j in range(i):

            r_corr = polycor.polychor(rating_vectors[i], rating_vectors[j], ML=True, std_err=True,thresholds=True)

            print(f"Correlation between ratings from {annotators[i]} and {annotators[j]} = {r_corr}")

            correlation = round(float(r_corr.rx2('rho')[0]),3)
            
            # visualize inter-rater confusion matrix
            x_ratings = np.array(list(rating_vectors[i]), dtype=int)
            y_ratings = np.array(list(rating_vectors[j]), dtype=int)

            # build 5x5 count matrix (rows = y rater, cols = x rater)
            counts = np.zeros((5, 5), dtype=int)
            for xi, yi in zip(x_ratings, y_ratings):
                counts[yi - 1, xi - 1] += 1

            ax = axes[ax_count] if num_combs > 1 else axes
            im = ax.imshow(counts, cmap='Blues', aspect='equal', origin='lower',
                           vmin=0, vmax=counts.max())

            # annotate each cell with the count
            for r in range(5):
                for c in range(5):
                    ax.text(c, r, str(counts[r, c]),
                            ha='center', va='center', fontsize=9,
                            color='black' if counts[r, c] < counts.max() * 0.6 else 'white')

            ax.set_xticks(range(5))
            ax.set_xticklabels(levels)
            ax.set_yticks(range(5))
            ax.set_yticklabels(levels)
            ax.set_xlabel(annotators[i])
            ax.set_ylabel(annotators[j])
            ax.set_title(f"ρ = {correlation}")
            plt.colorbar(im, ax=ax, shrink=0.8)

            ax_count += 1

    fig.suptitle("Inter-rater agreement (counts)", fontsize=12)
    fig.tight_layout()
    plt.show()
