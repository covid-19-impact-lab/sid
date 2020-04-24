import itertools

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
        contact_models (dict): See :ref:`contact_models`. They are already sorted.
        contact_policies (dict): See :ref:`policies`.
        states (pandas.DataFrame): See :ref:`states`.
        params (pandas.DataFrame): See :ref:`params`.
        period (int): Period of the model.

    Returns:
        contacts (np.ndarray): DataFrame with one column for each contact model that
            where model["model"] != "meet_group".

    """
    columns = []
    for model_name, model in contact_models.items():
        if model["model"] != "meet_group":
            loc = model.get("loc", params.index)
            func = model["model"]
            cont = func(states, params.loc[loc], period)

            if model_name in contact_policies:
                cp = contact_policies[model_name]
                if cp["start"] <= period <= cp["end"] and cp["is_active"](states):
                    cont *= cp["multiplier"]

        cont = _sum_preserving_round(cont.to_numpy()).astype(DTYPE_N_CONTACTS)
        columns.append(cont)

    contacts = np.column_stack(columns)

    return contacts


def calculate_infections(states, contacts, params, indexers, group_probs, seed):
    """Calculate infections from contacts.

    This function mainly converts the relevant parts from states and contacts into
    numpy arrays or other objects that are supported in numba nopython mode and
    then calls ``calculate_infections_numba``.

    Args:
        states (pandas.DataFrame): see :ref:`states`.
        contacts (pandas.DataFrame): One column per contact_model. Same index as states.
        params (pandas.DataFrame): See :ref:`params`.
        indexers (dict): Dict of numba.Typed.List The i_th entry of the lists are the
            indices of the i_th group.
        group_probs (dict): dict of arrays of shape
            n_group, n_groups. probs[i, j] is the probability that an individual from
            group i meets someone from group j.
        seed (itertools.count): Seed counter to control randomness.

    Returns:
        infected_sr (pd.Series): Boolean Series that is True for newly infected people.
        states (pandas.DataFrame): Copy of states with updated immune column.

    """
    is_meet_group = np.array([k not in group_probs for k in indexers])
    states = states.copy()
    infectious = states["infectious"].to_numpy(copy=True)
    immune = states["immune"].to_numpy(copy=True)
    group_codes = states[[f"group_codes_{cm}" for cm in indexers]].to_numpy()
    infect_probs = np.zeros((len(indexers), 2))
    infect_probs[:, 0] = [
        params.loc[("infection_prob", cm), "value"] for cm in indexers
    ]
    infect_probs[:, 1] = 1 - infect_probs[:, 0]

    group_probs_list = NumbaList()
    for gp in group_probs.values():
        group_probs_list.append(gp)

    indexers_list = NumbaList()
    for ind in indexers.values():
        indexers_list.append(ind)

    np.random.seed(next(seed))
    loop_entries = np.array(
        list(itertools.product(range(len(states)), range(len(indexers))))
    )

    indices = np.random.choice(len(loop_entries), replace=False, size=len(loop_entries))
    loop_order = loop_entries[indices]

    infected, infection_counter, immune, missed = _calculate_infections_numba(
        contacts,
        infectious,
        immune,
        group_codes,
        group_probs_list,
        indexers_list,
        infect_probs,
        next(seed),
        is_meet_group,
        loop_order,
    )

    infected_sr = pd.Series(infected, index=states.index, dtype=bool)
    states["infection_counter"] += infection_counter
    for i, contact_model in enumerate(group_probs):
        states[f"missed_{contact_model}"] = missed[:, i]

    states["immune"] = immune

    return infected_sr, states


@njit
def _calculate_infections_numba(
    contacts,
    infectious,
    immune,
    group_codes,
    group_probs_list,
    indexers_list,
    infection_probs,
    seed,
    is_meet_group,
    loop_order,
):
    """Match people, draw if they get infected and record who infected whom.

    Args:
        contacts (numpy.ndarray): 2d integer array with number of contacts per
            individual. There is one row per individual in the state and one column
            for each contact model where model["model"] != "meet_group".
        infectious (numpy.ndarray): 1d boolean array that indicates if a person is
            infectious. This is not directly changed after an infection.
        immune (numpy.ndarray): 1d boolean array that indicates if a person is immune.
        group_codes (numpy.ndarray): 2d integer array with the index of the group used
            in the first stage of matching.
        group_probs_list (numba.typed.List): List of arrays of shape n_group, n_groups.
            arr[i, j] is the probability that an individual from group i meets someone
            from group j.
        indexers_list (numba.typed.List): Nested typed list. The i_th entry of the inner
            lists are the indices of the i_th group. There is one inner list per contact
            model.
        infection_probs (numpy.ndarray): Array of shape (n_contact_models, 2). The first
            column contains the probability of infection for each contact model. The
            second column 1 minus that probability.
        seed (int): Seed value to control randomness.
        is_meet_group (numpy.ndarray): Boolean array of length n_contact_models.
        loop_orrder (np.ndarray): 2d numpy array with two columns. The first column
            indicates an individual. The second indicates a contact model.

    Returns:
        infected (numpy.ndarray): 1d boolean array that is True for individuals who got
            newly infected.
        infection_counter (numpy.ndarray): 1d integer array
        immune (numpy.ndarray): 1-D boolean array that indicates if a person is immune.
        missed (numpy.ndarray): 1d integer array with missed contacts. Same length as
            contacts.

    """
    np.random.seed(seed)

    infected = np.zeros(len(contacts))
    infection_counter = np.zeros(len(contacts))

    groups_list = [np.arange(len(gp)) for gp in group_probs_list]
    infection_events = np.array([True, False])

    # Loop over all individual-contact_model combinations
    for k in range(len(loop_order)):
        i, cm = loop_order[k]
        if is_meet_group[cm]:
            # We only check if i gets infected by someone else from his group. Whether
            # he infects some j is only checked, when the main loop arrives at j.
            group_i = group_codes[i, cm]
            if not immune[i] and group_i >= 0:
                others = indexers_list[cm][group_i]
                # we don't have to handle the case where j == i because if i is
                # infectious he is also immune, if he is not infectious, nothing
                # happens anyways.
                for j in others:
                    if infectious[j]:
                        is_infection = _choose_one_element(
                            infection_events, weights=infection_probs[cm]
                        )
                        if is_infection:
                            infection_counter[j] += 1
                            infected[i] = 1
                            immune[i] = True

        else:
            # get the probabilities for meeting another group which depend on the
            # individual's group.
            group_i = group_codes[i, cm]
            group_probs_i = group_probs_list[cm][group_i]

            # Loop over each contact the individual has, sample the contact's group and
            # compute the sum of possible contacts in this group.
            n_contacts = contacts[i, cm]
            for _ in range(n_contacts):
                contact_takes_place = True
                group_j = _choose_one_element(groups_list[cm], weights=group_probs_i)
                choice_indices = indexers_list[cm][group_j]
                contacts_j = contacts[choice_indices, cm]
                contacts_sum_j = contacts_j.sum()

                # If no individual in the contacted group can have another contact, no
                # meeting takes place.
                if contacts_sum_j == 0:
                    contact_takes_place = False

                # If a contact is possible, sample the index of the contact. If the
                # contact is the individual itself, no contact takes place.
                else:
                    p = contacts_j / contacts_sum_j
                    j = _choose_one_element(choice_indices, weights=p)
                    if i == j:
                        contact_takes_place = False

                # If a contact takes place, find out if one individual got infected.
                if contact_takes_place:
                    contacts[i, cm] -= 1
                    contacts[j, cm] -= 1

                    if infectious[i] and not immune[j]:
                        is_infection = _choose_one_element(
                            infection_events, weights=infection_probs[cm]
                        )
                        if is_infection:
                            infection_counter[i] += 1
                            infected[j] = 1
                            immune[j] = True

                    elif infectious[j] and not immune[i]:
                        is_infection = _choose_one_element(
                            infection_events, weights=infection_probs[cm]
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
