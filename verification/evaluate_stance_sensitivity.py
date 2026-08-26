"""
Script to evaluate the correlation between stance scoring under different baseline statements
"""
from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt
import argparse
import random

ROOT = Path(__file__).resolve().parents[1]
# ensure project imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.log_correcter import clean_log_times_and_init
from modeling.stance_analysis_for_modeling.embedding_analyzer_sync import EmbeddingAnalyzerSync

POISSON_MEAN = 15.0
TIME_UNIT_MS = 8000
RATE_PER_S = (POISSON_MEAN / (TIME_UNIT_MS / 1000.0))  # (incorrect but kept for consistency with previous code)

REFERENCE_BASELINE_STATEMENTS = {
    "climate": "Climate Change is caused by human activities",
    "vaccines": "Vaccines cause autism",
    "gun-control": "Gun ownership is a fundamental right"
}


ALTERNATE_BASELINE_STATEMENTS = {
    "climate": ["climate change is human-caused",
                "The phenomenon of climate change is human caused",
                "Humans cause climate change"],
    "vaccines": ["autism can be caused by vaccines",
                 "Vaccines are a potential cause of autism",
                 "Vaccines are linked to autism"],
    "gun-control": ["being able to own a gun is a fundamental right",
                    "It's a fundamental right to be able to own a gun",
                    "The ownership of guns is a basic right"]
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate stance scoring sensitivity to baseline statements")
    # default llm-topic pair is llama3.1-climate
    parser.add_argument("--llm", type=str, default="llama3.1", help="LLM name")
    parser.add_argument("--topic", type=str, default="climate", help="Topic name")
    parser.add_argument("--plot", action="store_true", help="Plot the stance score comparisons")
    args = parser.parse_args()

    rng = random.Random(1234)   # random generator for timestamping - does not affect scoring

    experiment_dir = ROOT / 'modeling' / 'runs' / args.llm / args.topic
    if not experiment_dir.exists():
        raise ValueError(f"Experiment directory {experiment_dir} does not exist.")
    
    scoring_strings = ALTERNATE_BASELINE_STATEMENTS[args.topic]
    # prepend the reference baseline statement to the list of scoring strings
    scoring_strings = [REFERENCE_BASELINE_STATEMENTS[args.topic]] + scoring_strings

    message_list_out_dicts = {}
    message_stance_score_lists = {}

    # for each scoring string, score all of the messages against
    for scoring_string in scoring_strings:
        print(f"Scoring with baseline statement: {scoring_string}")
        embedding_analyzer = EmbeddingAnalyzerSync(topic=scoring_string)
        

        message_list_out_dict = clean_log_times_and_init(experiment_dir=experiment_dir,
                                                         out_dir='',
                                                         poisson_lambda=RATE_PER_S,
                                                         rng=rng,
                                                         embedding_analyzer=embedding_analyzer,
                                                         re_embed_all_messages=True,
                                                         skip_file_write_test=True,
                                                         bypass_init_match_validation=True,)['message_list_out_dict']

        message_list_out_dicts[scoring_string] = message_list_out_dict

        ss_list = []
        # also, build a list of stance scores for the messages in the train set
        for run_name, messages in message_list_out_dict['train'].items():

            
            if scoring_string == REFERENCE_BASELINE_STATEMENTS[args.topic]:
                # load up the stance scores for the expeimrnets
                main_save_messages = []
                MAIN_SAVE_LOCATION = ROOT / 'modeling' / 'runs_rescored' / args.llm / args.topic / 'train' / run_name / 'messages_with_alignment.jsonl'
                with open (MAIN_SAVE_LOCATION, 'r') as f:
                    for line in f:
                        main_save_messages.append(json.loads(line))


            for i, message in enumerate(messages):
                ss_list.append(message['published']['stance_score'])

                if scoring_string == REFERENCE_BASELINE_STATEMENTS[args.topic]:
                    # check that the stance scores match the main save messages
                    if not abs(message['published']['stance_score'] - main_save_messages[i]['published']['stance_score']) < 1e-3:
                        raise ValueError(f"Stance score mismatch for message {i} in run {run_name} for scoring string {scoring_string}.")

        message_stance_score_lists[scoring_string] = ss_list
    
    # save to
    SAVE_LOCATION = ROOT / 'verification' / 'prompt_sensitivity_results' / f"{args.llm}_{args.topic}_stance_score_sensitivity.json"
    SAVE_LOCATION.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVE_LOCATION, 'w') as f:
        json.dump(message_stance_score_lists, f, indent=2)

    if args.plot:
        # plotting
        #  for each scoring string in ALTERNATE_BASELINE_STATEMENTS, 
        #  plot the stance scores against the stance scores from the reference baseline statement
        ref_ss_list = message_stance_score_lists[REFERENCE_BASELINE_STATEMENTS[args.topic]]
        for scoring_string in scoring_strings[1:]:
            alt_ss_list = message_stance_score_lists[scoring_string]
            plt.figure(figsize=(8, 6))
            plt.scatter(ref_ss_list, alt_ss_list, alpha=0.5)
            plt.xlabel(f'Stance scores with reference baseline statement: {REFERENCE_BASELINE_STATEMENTS[args.topic]}')
            plt.ylabel(f'Stance scores with alternate baseline statement: {scoring_string}')
            plt.title(f'Stance score comparison for {args.llm}/{args.topic}')
            plt.grid(True)
            plt.show()
