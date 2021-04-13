"""This module contains configuration values for sid."""
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
    "newly_infected",
    "newly_deceased",
    "new_known_case",
    "newly_vaccinated",
    "ever_vaccinated",
    "is_tested_positive_by_rapid_test",
]

DTYPE_COUNTDOWNS = np.int16
"""Dtype for the countdowns.

The dtype has to be signed integer because `-1` is assigned to counters which have not
been started.

"""
DTYPE_DRAW_COURSE_OF_DISEASE = np.int16
DTYPE_GROUP_CODE = np.int32
DTYPE_INDEX = np.uint32
DTYPE_VIRUS_STRAIN = np.int8
DTYPE_INFECTION_COUNTER = np.uint16
DTYPE_N_CONTACTS = np.uint16
DTYPE_SID_PERIOD = np.int16
DTYPE_VIRUS_STRAIN_CODES = np.int8
DTYPE_CONTACT_MODEL_CODES = np.int8
DTYPE_GROUP_TRANSITION_PROBABILITIES = np.float32

INDEX_NAMES = ["category", "subcategory", "name"]

ROOT_DIR = Path(__file__).parent

RELATIVE_POPULATION_PARAMETER = 1 / 100_000

SAVED_COLUMNS = {
    "time": ["date"],
    "initial_states": True,
    "disease_states": True,
    "testing_states": False,
    "countdowns": ["cd_infectious_false"],
    "contacts": False,
    "countdown_draws": False,
    "group_codes": False,
    "other": ["n_has_infected", "newly_infected", "new_known_case", "virus_strain"],
    "channels": False,
    "rapid_tests": True,
}


INITIAL_CONDITIONS = {
    "assort_by": None,
    "burn_in_periods": 1,
    "growth_rate": 1,
    "known_cases_multiplier": 1,
    "initial_infections": 0.01,
    "initial_immunity": None,
    "virus_shares": None,
}


DEFAULT_VIRUS_STRAINS = {"names": ["base_strain"], "factors": np.array([1])}


SID_TIME_START = pd.Timestamp("2019-01-01")
