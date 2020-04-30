"""Draw whether people get symptoms, need icu etc. and if so after what time.

This will have to be changed completely, once we allow for randomness in the
countdown lengths. Currently, most of it is deterministic.

"""
import numpy as np
import pandas as pd


def draw_course_of_disease(states, params, seed):
    """Draw course of the disease.

    The course of disease is drawn before the actual simulation and samples for each
    individual the length of the countdown.

    Countdowns govern the transition from being infectious to having symptoms,
    potentially needing icu and maybe even dying.

    Args:
        states (pandas.DataFrame): The initial states.
        params (pandas.DataFrame): DataFrame with parameters that influence the number
            of contacts, contagiousness and dangerousness of the disease, ... .
        seed (itertools.count): Seed counter to control randomness.

    Returns:
        states (pandas.DataFrame): The initial states extended with countdown lengths.

    """
    np.random.seed(next(seed))

    states = states.copy()

    # time of immunity
    cd_length = params.loc[("countdown_length", "cd_immune_false", None), "value"]
    states["cd_immune_false_draws"] = cd_length

    # time until symptoms. -1 if no symptoms
    cd_length = params.loc[("countdown_length", "cd_symptoms_true", None), "value"]
    prob = params.loc[("prob_symptoms_given_infection", "all", None), "value"]
    states["cd_symptoms_true_draws"] = _two_stage_sampling(prob, cd_length, len(states))

    # time until icu
    cd_length = params.loc[("countdown_length", "cd_needs_icu_true", None), "value"]
    probs = params.loc["prob_icu_given_symptoms", "value"]
    probs = probs.reset_index(level="name", drop=True)
    states["cd_needs_icu_true_draws"] = _age_varying_two_stage_sampling(
        states, probs, cd_length
    )
    states["cd_needs_icu_true_draws"] = states["cd_needs_icu_true_draws"].where(
        states["cd_symptoms_true_draws"] >= 0, -1
    )

    # time until death
    cd_length = params.loc[("countdown_length", "cd_dead", None), "value"]
    probs = params.loc["prob_dead_given_icu", "value"]
    probs = probs.reset_index(level="name", drop=True)
    states["cd_dead_draws"] = _age_varying_two_stage_sampling(states, probs, cd_length)
    states["cd_dead_draws"] = states["cd_dead_draws"].where(
        states["cd_needs_icu_true_draws"] >= 0, -1
    )

    # length of symptoms; can be drawn for all because it will only be triggered if
    # needed anyways
    cd_length = params.loc[("countdown_length", "cd_symptoms_false", None), "value"]
    states["cd_symptoms_false_draws"] = cd_length

    # length of icu treatment; can be drawn for all because it will only be triggered
    # for people who needed icu
    cd_length = params.loc[("countdown_length", "cd_needs_icu_false", None), "value"]
    states["cd_needs_icu_false_draws"] = cd_length

    # length of testing
    cd_length = params.loc[("countdown_length", "cd_knows_true", None), "value"]
    states["cd_knows_true_draws"] = cd_length

    # time until infectiousness
    cd_length = params.loc[("countdown_length", "cd_infectious_true", None), "value"]
    states["cd_infectious_true_draws"] = cd_length

    # length of infectiousness, can be drawn for all because if will only be triggered
    # for people who became infectious
    cd_length = params.loc[("countdown_length", "cd_infectious_false", None), "value"]
    states["cd_infectious_false_draws"] = cd_length

    return states


def _two_stage_sampling(prob, val, size):
    return np.random.choice(a=[val, -1], p=[prob, 1 - prob], size=size)


def _age_varying_two_stage_sampling(states, probs, val):
    """Sample probability by age groups.

    Args:
        probs (pandas.Series): Index contains age groups in same codes as "age_group"
            column in states.
        val (int): The sampled value is either this value or -1.

    Returns: s (pandas.Series): Series containing the sampled values with age-varying
        distributions.

    """
    s = pd.Series(-1, index=states.index, dtype=np.int32)
    # extract age groups from states instead of from probs and then look up the probs,
    # so we get a key error for missing parameters due to typos in the params index.
    # otherwise it would fail silently.
    age_groups = states["age_group"].unique().tolist()
    for age_group in age_groups:
        prob = probs[age_group]
        locs = states.query(f"age_group == '{age_group}'").index
        s.loc[locs] = _two_stage_sampling(prob, val, len(locs))

    return s
