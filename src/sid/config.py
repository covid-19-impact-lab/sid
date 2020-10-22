from pathlib import Path

import numpy as np


BOOLEAN_STATE_COLUMNS = [
    "ever_infected",
    "immune",
    "infectious",
    "symptomatic",
    "needs_icu",
    "dead",
    "pending_test",
    "received_test_result",
    "knows_immune",
    "knows_infectious",
    "demands_test",
    "allocated_test",
    "to_be_processed_test",
]

DTYPE_COUNTDOWNS = np.int16
"""Dtype for the countdowns.

The dtype has to be signed integer because `-1` is assigned to counters which have not
been started.

"""
DTYPE_DRAW_COURSE_OF_DISEASE = np.int16
DTYPE_GROUP_CODE = np.int32
DTYPE_INDEX = np.uint32
DTYPE_INFECTED = np.bool_
DTYPE_INFECTION_COUNTER = np.uint16
DTYPE_N_CONTACTS = np.uint32
DTYPE_PERIOD = np.uint16

INDEX_NAMES = ["category", "subcategory", "name"]

ROOT_DIR = Path(__file__).parent

USELESS_COLUMNS = [
    # General countdowns.
    "cd_infectious_true",
    "cd_immune_false",
    "cd_symptoms_true",
    "cd_symptoms_false",
    "cd_needs_icu_true",
    "cd_dead_true",
    "cd_needs_icu_false",
    "cd_immune_false_draws",
    "cd_symptoms_true_draws",
    "cd_needs_icu_true_draws",
    "cd_dead_true_draws",
    "cd_symptoms_false_draws",
    "cd_needs_icu_false_draws",
    "cd_infectious_true_draws",
    "cd_infectious_false_draws",
    # Countdowns related to testing.
    "cd_received_test_result_true_draws",
    "cd_knows_immune_false",
    "cd_knows_infectious_false",
    # Others.
    "demands_test",
    "allocated_test",
    "to_be_processed_test",
    "pending_test_date",
    "pending_test_period",
]
