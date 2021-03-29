"""Draw whether people get symptoms, need icu etc. and if so after what time.

This will have to be changed completely, once we allow for randomness in the
countdown lengths. Currently, most of it is deterministic.

"""
import numpy as np
import pandas as pd
from sid.config import DTYPE_DRAW_COURSE_OF_DISEASE
from sid.countdowns import COUNTDOWNS_WITH_DRAWS


def draw_course_of_disease(states, params, seed):
    """Draw course of the disease.

    The course of disease is drawn before the actual simulation and samples for each
    individual the length of the countdown.

    Countdowns govern the transition from becoming infectious, to potentially
    having symptoms, needing icu and maybe even dying.

    Args:
        states (pandas.DataFrame): The initial states.
        params (pandas.DataFrame): DataFrame with parameters that influence the number
            of contacts, contagiousness and dangerousness of the disease, ... .
        seed (int): Seed counter to control randomness.

    Returns:
        states (pandas.DataFrame): The initial states extended with countdown lengths.

    """
    np.random.seed(seed)

    for cd in COUNTDOWNS_WITH_DRAWS:
        states[f"{cd}_draws"] = _draw_countdowns(states, params.loc[cd])

    # Make sure no one who is scheduled to die recovers before their death.
    would_die_if_infected = states.cd_dead_true_draws > 0
    states.loc[would_die_if_infected, "cd_needs_icu_false_draws"] = -1

    return states


def _draw_countdowns(states, params_slice):
    """Draw the countdowns.

    Args:
        states (pandas.DataFrame): The initial states, includes the age_group by which
            probabilities may differ between individuals.
        params_slice (pandas.DataFrame):
            DataFrame slice with the parameters of the current countdown to be drawn.
            the "name" index level contains the possible realizations, the "value"
            column contains the probabilities. If either differ between age groups
            the "subcategory" index level contains the group values. If they do not
            differ the "subcategory" is "all".

    Returns:
        draws (pandas.Series): Series with the countdowns. Has the same index as states.

    """
    if len(params_slice) == 1:
        value = params_slice.index[0][1]
        draws = pd.Series(value, index=states.index)
    elif set(params_slice.index.get_level_values("subcategory")) == {"all"}:
        realizations = params_slice.loc["all"].index
        probs = params_slice["value"]
        draws = np.random.choice(a=realizations, p=probs, size=len(states))
        draws = pd.Series(draws, index=states.index)
    else:
        draws = pd.Series(np.nan, index=states.index)
        # extract age groups from states instead of probs and then look up the probs,
        # so we get a key error for missing parameters due to typos in the params index.
        # otherwise it would fail silently.
        age_groups = states["age_group"].unique()
        for age_group in age_groups:
            age_entry = params_slice.loc[age_group]
            realizations = age_entry.index.values
            probs = age_entry["value"]

            locs = states.query(f"age_group == '{age_group}'").index
            draws.loc[locs] = np.random.choice(a=realizations, p=probs, size=len(locs))

    return draws.astype(DTYPE_DRAW_COURSE_OF_DISEASE)
