"""Draw whether people get symptoms, need icu etc. and if so after what time.

This will have to be changed completely, once we allow for randomness in the
countdown lengths. Currently, most of it is deterministic.

"""
import numpy as np
import pandas as pd


def draw_course_of_disease(states, params):
    states = states.copy()

    # time of immunity
    cd_length = params.loc[("countdown_length", "cd_immune_false"), "value"]
    states["cd_immune_false_draws"] = cd_length

    # time until symptoms. -1 if no symptoms
    cd_length = params.loc[("countdown_length", "cd_symptoms_true"), "value"]
    prob = params.loc[("prob_symptoms_given_infection", "all"), "value"]
    states["cd_symptoms_true_draws"] = _two_stage_sampling(prob, cd_length, len(states))

    # time until icu
    cd_length = params.loc[("countdown_length", "cd_needs_icu_true"), "value"]
    probs = params.loc["prob_icu_given_symptoms", "value"]
    states["cd_needs_icu_true_draws"] = _age_varying_two_stage_sampling(
        states, probs, cd_length
    )
    states["cd_needs_icu_true_draws"] = states["cd_needs_icu_true_draws"].where(
        states["cd_symptoms_true_draws"] >= 0, -1
    )

    # time until death
    cd_length = params.loc[("countdown_length", "cd_dead"), "value"]
    probs = params.loc["prob_dead_given_icu", "value"]
    states["cd_dead_draws"] = _age_varying_two_stage_sampling(states, probs, cd_length)
    states["cd_dead_draws"] = states["cd_dead_draws"].where(
        states["cd_needs_icu_true_draws"] >= 0, -1
    )

    # length of symptoms; can be drawn for all because it will only be triggered if
    # needed anyways
    cd_length = params.loc[("countdown_length", "cd_symptoms_false"), "value"]
    states["cd_symptoms_false_draws"] = cd_length

    # length of icu treatment; can be drawn for all because it will only be triggered
    # for people who needed icu
    cd_length = params.loc[("countdown_length", "cd_needs_icu_false"), "value"]
    states["cd_needs_icu_false_draws"] = cd_length

    # length of testing
    cd_length = params.loc[("countdown_length", "cd_knows_true"), "value"]
    states["cd_knows_true_draws"] = cd_length

    # time until infectiousness
    cd_length = params.loc[("countdown_length", "cd_infectious_true"), "value"]
    states["cd_infectious_true_draws"] = cd_length

    # length of infectiousness, can be drawn for all because if will only be triggered
    # for people who became infectious
    cd_length = params.loc[("countdown_length", "cd_infectious_false"), "value"]
    states["cd_infectious_false_draws"] = cd_length

    return states


def _two_stage_sampling(prob, val, size):
    return np.random.choice(a=[val, -1], p=[prob, 1 - prob], size=size)


def _age_varying_two_stage_sampling(states, probs, val):
    """

    Args:
        probs (pd.Series): Index contains age groups in same codes as "age_group"
            column in states.
        val (int)

    """
    sr = pd.Series(-1, index=states.index, dtype=np.int32)
    for age_group, prob in probs.items():
        locs = states.query(f"age_group == '{age_group}'").index
        sr.loc[locs] = _two_stage_sampling(prob, val, len(locs))
    return sr
