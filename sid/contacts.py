import itertools
from inspect import getmembers

import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List as numba_list

from sid import contact_models as contact_models_module


def calculate_contacts(contact_models, states, params, period):
    """Calculate number of contacts of different types.

    # this is mainly a placeholder and has no support for policies yet.

    Args:
        contact_models (list): See :ref:`contact_models`
        states (pd.DataFrame): See :ref:`states`
        params (pd.DataFrame): See :ref:`params`
        period (int): Period of the model

    Returns:
        contacts (pd.DataFrame): DataFrame with one column per contact type.


    """
    contact_types = sorted(set([mod["contact_type"] for mod in contact_models]))
    contacts = pd.DataFrame(data=0, index=states.index, columns=contact_types)
    for model in contact_models:
        if isinstance(model["model"], str):
            func = getattr(contact_models_module, model["model"])
        else:
            func = model["model"]
        cont = func(states, params, period)
        # apply policies
        contacts[model["contact_type"]] = cont

    return contacts


def calculate_infections(states, contacts, params, indexer, group_probs):
    """Calculate infections from contacts.

    This function mainly converts the relevant parts from states and contacts into
    numpy arrays or other objects that are supported in numba nopython mode and
    then calls ``calculate_infections_numba``.

    """
    states = states.copy()
    infectious = states["infectious"].to_numpy()
    immune = states["immune"].to_numpy()
    group_codes = states["group_codes"].to_numpy()

    infected_sr = pd.Series(index=states.index, data=0)

    for contact_type in contacts.columns:
        cont = contacts[contact_type].to_numpy(copy=True)
        infect_prob = params.loc[("infection_prob", contact_type), "value"]
        infected, infection_counter, immune = _calculate_infections_numba(
            cont,
            infectious,
            immune,
            group_codes,
            infect_prob,
            group_probs,
            indexer,
            infect_prob,
        )
        infected_sr += infected
        states["infection_counter"] += infection_counter

    states["immune"] = immune
    infected_sr = infected_sr.astype(bool)

    return infected_sr, states


@njit
def _calculate_infections_numba(
    contacts,
    infectious,
    immune,
    group_codes,
    infect_prob,
    group_probs,
    indexer,
    infection_prob,
):
    """Match people, draw if they get infected and record who infected whom.

    Args:
        contacts (np.ndarray): 1-D integer array with number of contacts per individual.
            This is only for one contact type.
        infectious (np.ndarray): 1-D boolean array that indicates if a person is
            infectious. This is not changed after an infection.
        immune (np.ndarray): 1-D boolean array that indicates if a person is immune.
        group_codes (np.ndarray): 1-D integer array with the index of the group used
            in the first stage of matching.
        infect_prob (float): Probability of infection for the contact type.
        group_probs (np.ndarray): Array of shape n_group, n_groups. probs[i, j] is the
            probability that an individual from group i meets someone from group j.
        indexer (numba.typed.List): The i_th entry are the indices of the i_th group.

    Returns:
        infected (np.ndarray): 1-D boolean array that is True for individuals who got
            newly infected.
        infection_counter (np.ndarray): 1-D integer array
        immune (np.ndarray):

    """
    immune = immune.copy()
    infected = np.zeros(len(contacts))
    infection_counter = np.zeros_like(contacts)
    n_obs = len(contacts)
    groups = np.arange(len(group_probs))

    infection_events = np.array([True, False])
    infection_prob = np.array([infection_prob, 1 - infection_prob])

    # it is important not to loop over contact directly, because contacts is changed
    # in place during the loop
    for i in range(n_obs):
        n_contacts = contacts[i]
        group_i = group_codes[i]
        gp = group_probs[group_i]
        for _ in range(n_contacts):
            contact_takes_place = True
            group_j = _choose_one_element(groups, weights=gp)
            choice_indices = indexer[group_j]
            contacts_j = contacts[choice_indices]
            contacts_sum_j = contacts_j.sum()
            if contacts_sum_j == 0:
                contact_takes_place = False
            else:
                p = contacts_j / contacts_sum_j
                j = _choose_one_element(choice_indices, weights=p)
                if i == j:
                    contact_takes_place = False

            if contact_takes_place:
                contacts[i] -= 1
                contacts[j] -= 1
                if infectious[i] and not immune[j]:
                    is_infection = _choose_one_element(
                        infection_events, weights=infection_prob
                    )
                    if is_infection:
                        infection_counter[i] += 1
                        infected[j] = 1
                        immune[j] = True
                elif infectious[j] and not immune[i]:
                    is_infection = _choose_one_element(
                        infection_events, weights=infection_prob
                    )
                    if is_infection:
                        infection_counter[j] += 1
                        infected[i] = 1
                        immune[i] = True

    return infected, infection_counter, immune


@njit
def _choose_one_element(a, weights):
    """Return an element of choices.

    Args:
        a (np.ndarray): 1d array of choices
        weights (np.ndarrray): 1d array of weights.

    Returns:
        choice: An element of a.

    """

    cdf = weights.cumsum()
    u = np.random.uniform(0, 1)
    # Note that :func:`np.argmax` returns the first index for multiple maximum values.
    index = (u < cdf).argmax()
    return a[index]


def create_group_indexer(states, assort_by):
    """Map group number to indices of group members in the states DataFrame.

    Groups are defined by the assort_by variables. People who have the same value in
    all assort_by variables belong to the same group.

    For efficiency reasons, we assign each group a number instead of identifying by
    the values of the assort_by variables directly.

    Args:
        states (pd.DataFrame): See :ref:`states`
        assort_by (list): List of variables that influence matching probabilities.

    Returns:
        indexer (numba.typed.List): The i_th entry are the indices of the i_th group.

    """
    states = states.copy()
    states["indices"] = np.arange(len(states))

    indexer = numba_list()
    for group in _get_group_list(states, assort_by):
        df = states
        for var, val in zip(assort_by, group):
            df = df.query(f"{var} == '{val}'")
        indexer.append(df["indices"].to_numpy(dtype=np.uint32))
    return indexer


def create_group_transition_probs(states, assort_by, params):
    """Create a transition matrix for groups.

    Args:
        states (pd.DataFrame): see :ref:`states`
        assort_by (list): List of variables that influence matching probabilities.
        params (pd.DataFrame): See :ref:`params`

    Returns
        probs (np.ndarray): Array of shape n_group, n_groups. probs[i, j] is the
            probability that an individual from group i meets someone from group j.

    """
    groups = _get_group_list(states, assort_by)
    same_probs = []
    other_probs = []
    for var in assort_by:
        p = params.loc[("assortative_matching", var), "value"]
        n_vals = len(states[var].unique())
        same_probs.append(p)
        other_probs.append((1 - p) / (n_vals - 1))

    probs = np.ones((len(groups), len(groups)))

    for i, g_from in enumerate(groups):
        for j, g_to in enumerate(groups):
            for v, (val1, val2) in enumerate(zip(g_from, g_to)):
                if val1 == val2:
                    probs[i, j] *= same_probs[v]
                else:
                    probs[i, j] *= other_probs[v]
    return probs


def _get_group_list(states, assort_by):
    assort_values = []
    for var in assort_by:
        assort_values.append(sorted(states[var].unique().tolist()))

    return list(itertools.product(*assort_values))


def get_group_to_code(states, assort_by):
    group_list = _get_group_list(states, assort_by)
    return {str(group_tup): index for index, group_tup in enumerate(group_list)}
