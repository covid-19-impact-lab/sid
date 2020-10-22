COUNTDOWNS = {
    "cd_infectious_true": {
        "changes": {"infectious": True, "n_has_infected": 0},
        "starts": ["cd_infectious_false", "cd_symptoms_true"],
    },
    # will be overridden if a person develops symptoms. In that case infectiousness
    # lasts as long as symptoms.
    "cd_infectious_false": {"changes": {"infectious": False}},
    "cd_immune_false": {"changes": {"immune": False}},
    "cd_symptoms_true": {
        "changes": {"symptomatic": True, "cd_infectious_false": -1},
        "starts": ["cd_symptoms_false", "cd_needs_icu_true"],
    },
    # will be overridden if a person needs ICU. In that case symptoms end with need for
    # ICU.
    "cd_symptoms_false": {"changes": {"symptomatic": False, "infectious": False}},
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
            # cd_infectious_false is set to 0 instead of -1 because this is needed for
            # the calculation of r_zero
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
            "knows_immune": False,
            "knows_infectious": False,
            # cd_infectious_false is set to 0 instead of -1 because this is needed for
            # the calculation of r_zero
            "cd_infectious_false": 0,
        }
    },
    "cd_received_test_result_true": {"changes": {"received_test_result": True}},
    "cd_knows_immune_false": {"changes": {"knows_immune": False}},
    "cd_knows_infectious_false": {"changes": {"knows_infectious": False}},
    "cd_ever_infected": {},
}
"""(dict): The dictionary with the information on countdowns.

- cd_infectious_true and cd_immune_false are triggered by an infection
- cd_received_test_result is triggered by tests
- all other countdowns are triggered by chain reactions.

"""