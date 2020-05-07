import numpy as np


BOOLEAN_STATE_COLUMNS = [
    "ever_infected",
    "immune",
    "infectious",
    "knows",
    "symptomatic",
    "needs_icu",
    "dead",
]


# cd_infectious_true and cd_immune_false are triggered by an infection
# cd_knows is triggered by tests
# all other countdowns are triggered by chain reactions.
COUNTDOWNS = {
    "cd_infectious_true": {
        "changes": {"infectious": True, "n_has_infected": 0},
        "starts": ["cd_infectious_false", "cd_symptoms_true"],
    },
    # will be overriden if a person develops symptoms. In that case
    # infectiousness lasts as long as symptoms.
    "cd_infectious_false": {"changes": {"infectious": False, "knows": False}},
    "cd_immune_false": {"changes": {"immune": False}},
    "cd_symptoms_true": {
        "changes": {"symptomatic": True, "cd_infectious_false": -1},
        "starts": ["cd_symptoms_false", "cd_needs_icu_true"],
    },
    # will be overriden if a person needs icu. In that case symptoms
    # end with need for icu.
    "cd_symptoms_false": {
        "changes": {"symptomatic": False, "infectious": False, "knows": False}
    },
    "cd_needs_icu_true": {
        "changes": {"needs_icu": True, "cd_symptoms_false": -1},
        "starts": ["cd_dead", "cd_needs_icu_false"],
    },
    "cd_dead": {
        "changes": {
            "dead": True,
            "symptomatic": False,
            "needs_icu": False,
            "knows": False,
            "cd_immune_false": -1,
            # cd_infectious_false is set to 0 instead of -1 because this is needed
            # for the calculation of r_zero
            "cd_infectious_false": 0,
            "cd_symptoms_false": -1,
            "cd_needs_icu_false": -1,
        }
    },
    "cd_needs_icu_false": {
        "changes": {
            "needs_icu": False,
            "symptomatic": False,
            "infectious": False,
            "knows": False,
            # cd_infectious_false is set to 0 instead of -1 because this is needed
            # for the calculation of r_zero
            "cd_infectious_false": 0,
        }
    },
    "cd_knows_true": {"changes": {"knows": True}},
}


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

USELESS_COLUMNS = [
    "cd_infectious_true",
    "cd_immune_false",
    "cd_symptoms_true",
    "cd_symptoms_false",
    "cd_needs_icu_true",
    "cd_dead",
    "cd_needs_icu_false",
    "cd_knows_true",
    "cd_immune_false_draws",
    "cd_symptoms_true_draws",
    "cd_needs_icu_true_draws",
    "cd_dead_draws",
    "cd_symptoms_false_draws",
    "cd_needs_icu_false_draws",
    "cd_knows_true_draws",
    "cd_infectious_true_draws",
    "cd_infectious_false_draws",
]
