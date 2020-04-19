import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List as NumbaList

from sid.config import DTYPE_INDEX
from sid.config import DTYPE_N_CONTACTS
from sid.shared import factorize_assortative_variables


def calculate_contacts(contact_models, contact_policies, states, params, period):
    """Calculate number of contacts of different types.

    Args:
        contact_models (dict): See :ref:`contact_models`.
        contact_policies (dict): See :ref:`policies`.
        states (pandas.DataFrame): See :ref:`states`.
        params (pandas.DataFrame): See :ref:`params`.
        period (int): Period of the model.

    Returns:
        contacts (pandas.DataFrame): DataFrame with one column per contact type.

    """
    contact_types = sorted({mod["contact_type"] for mod in contact_models.values()})
    contacts = pd.DataFrame(data=0, index=states.index, columns=contact_types)
    for model_name, model in contact_models.items():
        loc = model.get("loc", params.index)
        func = model["model"]
        cont = func(states, params.loc[loc], period)

        if model_name in contact_policies:
            cp = contact_policies[model_name]
            if cp["start"] <= period <= cp["end"] and cp["is_active"](states):
                cont *= cp["multiplier"]

        contacts[model["contact_type"]] += cont

    for contact_type in contact_types:
        contacts[contact_type] = _sum_preserving_round(
            contacts[contact_type].to_numpy()
        ).astype(DTYPE_N_CONTACTS)

    return contacts


def calculate_infections(states, contacts, params, indexer, group_probs):
    """Calculate infections from contacts.

    This function mainly converts the relevant parts from states and contacts into
    numpy arrays or other objects that are supported in numba nopython mode and
    then calls ``calculate_infections_numba``.

    Args:
        states (pandas.DataFrame): see :ref:`states`.
        contacts (pandas.DataFrame): One column per contact_type. Same index as states.
        params (pandas.DataFrame): See :ref:`params`.
        indexer (numba.typed.List): The i_th entry are the indices of the i_th group.
        group_probs (numpy.ndarray): group_probs (numpy.ndarray): Array of shape
            n_group, n_groups. probs[i, j] is the probability that an individual from
            group i meets someone from group j.

    Returns:
        infected_sr (pd.Series): Boolean Series that is True for newly infected people.
        states (pandas.DataFrame): Copy of states with updated immune column.

    """
    states = states.copy()
    infectious = states["infectious"].to_numpy()
    immune = states["immune"].to_numpy()
    group_codes = states["group_codes"].to_numpy()

    infected_sr = pd.Series(index=states.index, data=0)

    for contact_type in contacts.columns:
        cont = contacts[contact_type].to_numpy()
        infect_prob = params.loc[("infection_prob", contact_type), "value"]
        infected, infection_counter, immune, missed = _calculate_infections_numba(
            cont, infectious, immune, group_codes, group_probs, indexer, infect_prob,
        )
        infected_sr += infected
        states["infection_counter"] += infection_counter
        states[f"missed_{contact_type}"] = missed

    states["immune"] = immune
    infected_sr = infected_sr.astype(bool)

    return infected_sr, states


@njit
def _calculate_infections_numba(
    contacts, infectious, immune, group_codes, group_probs, indexer, infection_prob,
):
    """Match people, draw if they get infected and record who infected whom.

    Args:
        contacts (numpy.ndarray): 1-D integer array with number of contacts per
            individual. This is only for one contact type.
        infectious (numpy.ndarray): 1-D boolean array that indicates if a person is
            infectious. This is not changed after an infection.
        immune (numpy.ndarray): 1-D boolean array that indicates if a person is immune.
        group_codes (numpy.ndarray): 1-D integer array with the index of the group used
            in the first stage of matching.
        group_probs (numpy.ndarray): Array of shape n_group, n_groups. probs[i, j] is
            the probability that an individual from group i meets someone from group j.
        indexer (numba.typed.List): The i_th entry are the indices of the i_th group.
        infection_prob (float): Probability of infection for the contact type.

    Returns:
        infected (numpy.ndarray): 1d boolean array that is True for individuals who got
            newly infected.
        infection_counter (numpy.ndarray): 1d integer array
        immune (numpy.ndarray)
        missed (numpy.ndarray): 1d integer array with missed contacts. Same length as
            contacts.

    """
    contacts = contacts.copy()
    immune = immune.copy()
    infected = np.zeros(len(contacts))
    infection_counter = np.zeros_like(contacts)
    n_obs = len(contacts)
    groups = np.arange(len(group_probs))

    infection_events = np.array([True, False])
    infection_prob = np.array([infection_prob, 1 - infection_prob])

    # it is important not to loop over contact directly, because contacts is changed in
    # place during the loop
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

    missed = contacts

    return infected, infection_counter, immune, missed


@njit
def _choose_one_element(a, weights):
    """Return an element of choices.

    This function does the same as :func:`numpy.random.choice`, but is way faster.

    :func:`numpy.argmax` returns the first index for multiple maximum values.

    Args:
        a (numpy.ndarray): 1d array of choices
        weights (numpy.ndarray): 1d array of weights.

    Returns:
        choice (int): An element of a.

    Example:
        >>> chosen = _choose_one_element(np.arange(3), np.array([0.2, 0.3, 0.5]))
        >>> assert isinstance(chosen, int)

    """
    cdf = weights.cumsum()
    u = np.random.uniform(0, 1)
    index = (u < cdf).argmax()

    return a[index]


def create_group_indexer(states, assort_by):
    """Create the group indexer.

    The indexer is a list where the positions correspond to the group number defined by
    assortative variables. The values inside the list are one-dimensional integer arrays
    containing the indices of states belonging to the group.

    If there are no assortative variables, all individuals are assigned to a single
    group with code 0 and the indexer is a list where the first position contains all
    indices of states.

    For efficiency reasons, we assign each group a number instead of identifying by
    the values of the assort_by variables directly.

    Args:
        states (pandas.DataFrame): See :ref:`states`
        assort_by (list): List of variables that influence matching probabilities.

    Returns:
        indexer (numba.typed.List): The i_th entry are the indices of the i_th group.

    """
    if assort_by:
        groups = states.groupby(assort_by).groups
        _, group_codes_values = factorize_assortative_variables(states, assort_by)

        indexer = NumbaList()
        for group in group_codes_values:
            indexer.append(groups[group].to_numpy(dtype=DTYPE_INDEX))

    else:
        indexer = NumbaList()
        indexer.append(states.index.to_numpy(DTYPE_INDEX))

    return indexer


def create_group_transition_probs(states, assort_by, params):
    """Create a transition matrix for groups.

    Args:
        states (pandas.DataFrame): see :ref:`states`
        assort_by (list): List of variables that influence matching probabilities.
        params (pandas.DataFrame): See :ref:`params`

    Returns
        probs (numpy.ndarray): Array of shape n_group, n_groups. probs[i, j] is the
            probability that an individual from group i meets someone from group j.

    """
    _, group_codes_values = factorize_assortative_variables(states, assort_by)
    probs = np.ones((len(group_codes_values), len(group_codes_values)))

    if assort_by:
        same_probs = []
        other_probs = []
        for var in assort_by:
            p = params.loc[("assortative_matching", var), "value"]
            n_vals = len(states[var].unique())
            same_probs.append(p)
            other_probs.append((1 - p) / (n_vals - 1))

        for i, g_from in enumerate(group_codes_values):
            for j, g_to in enumerate(group_codes_values):
                for v, (val1, val2) in enumerate(zip(g_from, g_to)):
                    if val1 == val2:
                        probs[i, j] *= same_probs[v]
                    else:
                        probs[i, j] *= other_probs[v]

    return probs


@njit
def _sum_preserving_round(arr):
    """Round values in an array, preserving the sum as good as possible.

    The function loops over the elements of an array and collects the deviations to the
    nearest downward adjusted integer. Whenever the collected deviations reach a
    predefined threshold, +1 is added to the current element and the collected
    deviations are reduced by 1.

    Args:
        arr (numpy.ndarray): 1d numpy array.

    Returns:
        numpy.ndarray

    Example:
        >>> arr = np.full(10, 5.2)
        >>> _sum_preserving_round(arr)
        array([5., 5., 6., 5., 5., 5., 5., 6., 5., 5.])

        >>> arr = np.full(2, 1.9)
        >>> _sum_preserving_round(arr)
        array([2., 2.])

    """
    arr = arr.copy()

    threshold = 0.5
    deviation = 0

    for i in range(len(arr)):

        floor_value = int(arr[i])
        deviation += arr[i] - floor_value

        if deviation >= threshold:
            arr[i] = floor_value + 1
            deviation -= 1

        else:
            arr[i] = floor_value

    return arr
