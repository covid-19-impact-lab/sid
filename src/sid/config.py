from pathlib import Path

import numpy as np
import pandas as pd


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

INDEX_NAMES = ["category", "subcategory", "name"]

ROOT_DIR = Path(__file__).parent

RELATIVE_POPULATION_PARAMETER = 1 / 100_000

SAVED_COLUMNS = {
    "initial_states": True,
    "disease_states": True,
    "testing_states": False,
    "countdowns": ["cd_infectious_false"],
    "contacts": False,
    "countdown_draws": False,
    "group_codes": False,
    "other": ["n_has_infected", "newly_infected"],
}


OPTIONAL_STATE_COLUMNS = {
    "contacts": False,
    "reason_for_infection": False,
}


SID_TIME_START = pd.Timestamp("2019-01-01")
