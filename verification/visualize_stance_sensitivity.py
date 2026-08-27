"""
Visualize how stance scores change between different prompts
"""


from pathlib import Path
import sys
import json
import argparse
import random

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = False        # use matplotlib's own text renderer, not LaTeX
matplotlib.rcParams['font.family'] = 'serif'      # base text font -> serif (e.g. DejaVu Serif)
matplotlib.rcParams['mathtext.fontset'] = 'cm'     # math text (e.g. in $...$ labels) -> Computer Modern
matplotlib.rcParams['font.size'] = 15             # base font size (default 10 * 1.5)
matplotlib.rcParams['axes.titlesize'] = 15
matplotlib.rcParams['axes.labelsize'] = 15
matplotlib.rcParams['xtick.labelsize'] = 15
matplotlib.rcParams['ytick.labelsize'] = 15
matplotlib.rcParams['legend.fontsize'] = 15

ROOT = Path(__file__).resolve().parents[1]
# ensure project imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


STANCE_RESULT_LOCATION = ROOT / 'verification' / 'prompt_sensitivity_results'
FIGURE_SAVE_LOCATION = ROOT / 'verification' / 'prompt_sensitivity_figures'

VALID_LLMS = ['gemma3', 'llama3.1', 'qwen-3.1']
VALID_TOPICS = ['gun-control', 'climate', 'vaccines']


REFERENCE_BASELINE_STATEMENTS = {
    "climate": "Climate Change is caused by human activities",
    "vaccines": "Vaccines cause autism",
    "gun-control": "Gun ownership is a fundamental right"
}

REFERENCE_BASELINE_STATEMENTS_DISPLAY = {
    "climate": "Climate Change is caused \n by human activities",   # display slightly differently to get better spacing
    "vaccines": "Vaccines cause autism",
    "gun-control": "Gun ownership is a fundamental right"
}

def _parse_llm_topic_name_from_prompt_sensitivity(filename):

    llm_name = filename.split('_')[0]
    if not llm_name in VALID_LLMS:
        raise ValueError(f"Invalid LLM name : {llm_name}")
    
    topic_name = filename.split('_')[1]
    if not topic_name in VALID_TOPICS:
        raise ValueError(f"Invalid topic name: {topic_name}")
    
    return llm_name, topic_name

def _generate_plots_for_stance_sens_file(filepath):
    stance_data = {}
        
    llm_name, topic_name = _parse_llm_topic_name_from_prompt_sensitivity(filepath.name)

    with open(filepath, "r") as f:
        stance_data = json.load(f)

    # keys are alternate prompts, while values are lists of stance values
    #  1. Check that we have a (k,v) pair corresponding to the baseline statement
    #  2. Iterate over the dict and plot

    if not REFERENCE_BASELINE_STATEMENTS[topic_name] in stance_data:
        raise ValueError(f"Missing stance data for reference statement {REFERENCE_BASELINE_STATEMENTS[topic_name]}")

    num_alt_prompts = len(stance_data) - 1

    fig, axes = plt.subplots(1, num_alt_prompts,
                                 figsize=(5 * num_alt_prompts, 6.85),
                                 sharey='row',
                                 layout='constrained')

    reference_prompt = REFERENCE_BASELINE_STATEMENTS[topic_name]
    reference_data = stance_data[reference_prompt]

    counter = 0
    for stance_prompt, score_data in stance_data.items():
        
        if stance_prompt == reference_prompt:
            continue
        else:
            reference_prompt_display = REFERENCE_BASELINE_STATEMENTS_DISPLAY[topic_name]

            axes[counter].scatter(reference_data, score_data, alpha=0.2, s=5)
            axes[counter].set_ylabel(f"Stance score for\n \"{stance_prompt}\"")
            axes[counter].set_xlabel(f"Stance score for\n \"{reference_prompt_display}\"")

            counter += 1

    fig.savefig(FIGURE_SAVE_LOCATION / f'{llm_name}_{topic_name}_score_sensitivity.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURE_SAVE_LOCATION / f'{llm_name}_{topic_name}_score_sensitivity.svg', bbox_inches='tight')


def main():

    FIGURE_SAVE_LOCATION.mkdir(parents=True, exist_ok=True)
    for filepath in STANCE_RESULT_LOCATION.iterdir():
        _generate_plots_for_stance_sens_file(filepath)


if __name__ == "__main__":
    main()