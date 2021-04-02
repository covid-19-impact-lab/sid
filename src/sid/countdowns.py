"""Specification of the countdowns that govern the disease progression and testing."""

COUNTDOWNS = {
    "cd_infectious_true": {
        "changes": {"infectious": True, "n_has_infected": 0},
        "starts": ["cd_infectious_false", "cd_symptoms_true"],
    },
    "cd_infectious_false": {"changes": {"infectious": False}},
    "cd_immune_false": {"changes": {"immune": False}},
    "cd_symptoms_true": {
        "changes": {"symptomatic": True},
        "starts": ["cd_symptoms_false", "cd_needs_icu_true"],
    },
    "cd_symptoms_false": {"changes": {"symptomatic": False, "infectious": False}},
    # If a person requires ICU, symptoms will end when the need for ICU ends.
    "cd_needs_icu_true": {
        "changes": {"needs_icu": True, "cd_symptoms_false": -1},
        "starts": ["cd_dead_true", "cd_needs_icu_false"],
    },
    "cd_dead_true": {
        "changes": {
            "dead": True,
            "symptomatic": False,
            "needs_icu": False,
            "knows_immune": False,
            "knows_infectious": False,
            "cd_immune_false": -1,
            "cd_symptoms_false": -1,
            "cd_needs_icu_false": -1,
        }
    },
    "cd_needs_icu_false": {
        "changes": {
            "needs_icu": False,
            "symptomatic": False,
            "infectious": False,
            "knows_immune": False,
            "knows_infectious": False,
        }
    },
    "cd_received_test_result_true": {"changes": {"received_test_result": True}},
    "cd_knows_immune_false": {"changes": {"knows_immune": False}},
    "cd_knows_infectious_false": {"changes": {"knows_infectious": False}},
    "cd_ever_infected": {},
    "cd_is_immune_by_vaccine": {"changes": {"immune": True}},
    "cd_received_rapid_test": {},
}
"""(dict): The dictionary with the information on countdowns.

- cd_infectious_true and cd_immune_false are triggered by an infection
- cd_received_test_result is triggered by tests
- all other countdowns are triggered by chain reactions.

"""

COUNTDOWNS_WITHOUT_DRAWS = ("cd_received_rapid_test", "cd_ever_infected")
"""Tuple[str]: Countdowns which do not have draws."""

COUNTDOWNS_WITH_DRAWS = tuple(set(COUNTDOWNS) - set(COUNTDOWNS_WITHOUT_DRAWS))
"""Tuple[str]: Countdowns whose length is a random variable."""
