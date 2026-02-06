""" Model fitting and plotting the results for single-shot prediction tests. """

import matplotlib.pyplot as plt
import numpy as np
import sklearn

BASELINE_STATEMENT = "Vaccines cause autism"
BASELINE_TOPIC = "vaccine safety and autism"

RAW_DATA_DIRECTORY = "tests/single_shot_tests/data/raw_text_jsonl"
CLEANED_DATA_DIRECTORY = "tests/single_shot_tests/data/cstance_converted"
