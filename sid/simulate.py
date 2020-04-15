import numpy as np
import pandas as pd

from sid.config import BOOLEAN_STATE_COLUMNS
from sid.config import COUNTDOWNS
from sid.contacts import calculate_contacts
from sid.contacts import calculate_infections
from sid.contacts import create_group_indexer
from sid.contacts import create_group_transition_probs
from sid.contacts import get_group_to_code
from sid.pathogenesis import draw_course_of_disease
from sid.update_states import update_states


def simulate(
    params,
    initial_states,
    initial_infections,
    contact_models,
    policies,
    n_periods,
    assort_by=None,
):
    """Simulate the spread of covid-19.

    Args:
        params (pd.DataFrame): DataFrame with parameters that influence the number of
            contacts, contagiousness and dangerousness of the disease, ...
        initial_states (pd.DataFrame): See :ref:`states`
        initial_infectios (pd.Series): Series with the same index as states with
            initial infections.
        contact_models (list): List of dictionaries where each dictionary describes a
            channel by which contacts can be formed. See :ref:`contact_models`.
        policies (list): List of dictionaries with contact and testing policies. See
            :ref:`policies`
        n_periods (int): Number of periods to simulate.
        assort_by (list, optional): List of variable names. Contacts are assortative by
            these variables.

    """
    assort_by = _process_assort_by(assort_by, initial_states)
    states = _process_states(initial_states, assort_by)
    states = draw_course_of_disease(states, params)
    states = update_states(states, initial_infections, params)
    indexer = create_group_indexer(states, assort_by)
    first_probs = create_group_transition_probs(states, assort_by, params)

    statistics = []
    for period in range(n_periods):
        contacts = calculate_contacts(contact_models, states, params, period)
        infections, states = calculate_infections(
            states, contacts, params, indexer, first_probs
        )
        states = update_states(states, infections, params)
        statistics.append(calculate_statistics(states, infections))

    return statistics


def _process_assort_by(assort_by, states):
    assort_by = ["region", "age_group"] if assort_by is None else assort_by
    assort_by = [var for var in assort_by if var in states.columns]
    return assort_by


def _process_states(states, assort_by):
    states = states.copy()

    for col in BOOLEAN_STATE_COLUMNS:
        if col not in states.columns:
            states[col] = False

    for col in COUNTDOWNS:
        if col not in states.columns:
            states[col] = -1
        states[col] = states[col].astype(np.int32)

    states["infection_counter"] = 0
    group_to_code = get_group_to_code(states, assort_by)
    states["group_codes"] = list(map(tuple, states[assort_by].to_numpy().tolist()))
    states["group_codes"] = states["group_codes"].astype(str)
    states["group_codes"] = (
        states["group_codes"].replace(group_to_code).astype(np.uint16)
    )

    return states


def calculate_statistics(states, infections):
    """extract the following information from states"""
    infections = infections.copy()
    infections.name = "infections"
    df = pd.concat([states, infections], axis=1)
    gb = ["age_group", "region"]

    df["died"] = df["cd_dead"] == 0
    df["recovered"] = df["cd_infectious_false"] == 0

    stats = {}
    for var in ["infections", "recovered", "died"]:
        stats[f"abs_{var}"] = df.groupby(gb)[var].sum()
        stats[f"rel_{var}"] = df.groupby(gb)[var].mean()

    return stats
