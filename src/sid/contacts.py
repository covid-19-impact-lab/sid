import itertools

import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List as NumbaList
from sid.config import DTYPE_INDEX
from sid.config import DTYPE_INFECTED
from sid.config import DTYPE_INFECTION_COUNTER
from sid.config import DTYPE_N_CONTACTS
from sid.shared import factorize_assortative_variables


def calculate_contacts(contact_models, contact_policies, states, params, date):
    """Calculate number of contacts of different types.

    Args:
        contact_models (dict): See :ref:`contact_models`. They are already sorted.
        contact_policies (dict): See :ref:`policies`.
        states (pandas.DataFrame): See :ref:`states`.
        params (pandas.DataFrame): See :ref:`params`.
        date (pandas.Timestamp): The current date.

    Returns:
        contacts (numpy.ndarray): DataFrame with one column for each contact model.

    """
    columns = []
    for model_name, model in contact_models.items():
        loc = model.get("loc", params.index)
        func = model["model"]
        cont = func(states, params.loc[loc])
        if model_name in contact_policies:
            cp = contact_policies[model_name]
            policy_start = pd.to_datetime(cp["start"])
            policy_end = pd.to_datetime(cp["end"])
            if policy_start <= date <= policy_end and cp["is_active"](states):
                cont *= cp["multiplier"]
        if not model["is_recurrent"]:
            cont = _sum_preserving_round(cont.to_numpy().astype(DTYPE_N_CONTACTS))
            cont = cont
        columns.append(cont)

    contacts = np.column_stack(columns).astype(DTYPE_N_CONTACTS)
    return contacts


def calculate_infections_by_contacts(
    states, contacts, params, indexers, group_cdfs, seed
):
    """Calculate infections from contacts.

    This function mainly converts the relevant parts from states and contacts into
    numpy arrays or other objects that are supported in numba nopython mode and
    then calls :func:`_calculate_infections_by_contacts_numba`.

    Args:
        states (pandas.DataFrame): see :ref:`states`.
        contacts (pandas.DataFrame): One column per contact_model. Same index as states.
        params (pandas.DataFrame): See :ref:`params`.
        indexers (dict): Dict of numba.Typed.List The i_th entry of the lists are the
            indices of the i_th group.
        group_cdfs (dict): dict of arrays of shape
            n_group, n_groups. probs[i, j] is the cumulative probability that an
            individual from group i meets someone from group j.
        seed (itertools.count): Seed counter to control randomness.

    Returns:
        (tuple): Tuple containing

            - infected_sr (pandas.Series): Boolean Series that is True for newly
              infected people.
            - states (pandas.DataFrame): Copy of states with updated immune column.

    """
    is_recurrent = np.array([k not in group_cdfs for k in indexers])
    states = states.copy()
    infectious = states["infectious"].to_numpy(copy=True)
    immune = states["immune"].to_numpy(copy=True)
    group_codes = states[[f"group_codes_{cm}" for cm in indexers]].to_numpy()
    infect_probs = np.array(
        [params.loc[("infection_prob", cm, cm), "value"] for cm in indexers]
    )

    group_cdfs_list = NumbaList()
    for gp in group_cdfs.values():
        group_cdfs_list.append(gp)
    # nopython mode fails, if we leave the list empty or put a 1d array inside the list.
    if len(group_cdfs_list) == 0:
        group_cdfs_list.append(np.zeros((0, 0)))

    indexers_list = NumbaList()
    for ind in indexers.values():
        indexers_list.append(ind)

    np.random.seed(next(seed))
    loop_entries = np.array(
        list(itertools.product(range(len(states)), range(len(indexers))))
    )

    indices = np.random.choice(len(loop_entries), replace=False, size=len(loop_entries))
    loop_order = loop_entries[indices]

    (
        infected,
        infection_counter,
        immune,
        missed,
    ) = _calculate_infections_by_contacts_numba(
        contacts,
        infectious,
        immune,
        group_codes,
        group_cdfs_list,
        indexers_list,
        infect_probs,
        next(seed),
        is_recurrent,
        loop_order,
    )

    infected_sr = pd.Series(infected, index=states.index)
    states["n_has_infected"] += infection_counter
    for i, contact_model in enumerate(group_cdfs):
        states[f"missed_{contact_model}"] = missed[:, i]

    states["immune"] = immune

    return infected_sr, states


@njit
def _calculate_infections_by_contacts_numba(
    contacts,
    infectious,
    immune,
    group_codes,
    group_cdfs,
    indexers_list,
    infection_probs,
    seed,
    is_recurrent,
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
        group_cdfs (numba.typed.List): List of arrays of shape n_group, n_groups.
            arr[i, j] is the cumulative probability that an individual from group i
            meets someone from group j.
        indexers_list (numba.typed.List): Nested typed list. The i_th entry of the inner
            lists are the indices of the i_th group. There is one inner list per contact
            model.
        infection_probs (numpy.ndarray): 1d array of length n_contact_models with the
            probability of infection for each contact model.
        seed (int): Seed value to control randomness.
        is_recurrent (numpy.ndarray): Boolean array of length n_contact_models.
        loop_orrder (numpy.ndarray): 2d numpy array with two columns. The first column
            indicates an individual. The second indicates a contact model.

    Returns:
        (tuple) Tuple containing

            - infected (numpy.ndarray): 1d boolean array that is True for individuals
              who got newly infected.
            - infection_counter (numpy.ndarray): 1d integer array
            - immune (numpy.ndarray): 1-D boolean array that indicates if a person is
              immune.
            - missed (numpy.ndarray): 1d integer array with missed contacts. Same length
              as contacts.

    """
    np.random.seed(seed)

    infected = np.zeros(len(contacts), dtype=DTYPE_INFECTED)
    infection_counter = np.zeros(len(contacts), dtype=DTYPE_INFECTION_COUNTER)
    groups_list = [np.arange(len(gp)) for gp in group_cdfs]

    # Loop over all individual-contact_model combinations
    for k in range(len(loop_order)):
        i, cm = loop_order[k]
        if is_recurrent[cm]:
            # We only check if i gets infected by someone else from his group. Whether
            # he infects some j is only checked, when the main loop arrives at j.
            group_i = group_codes[i, cm]
            # skip completely if i does not have a group or is already immune
            if not immune[i] and contacts[i, cm] > 0:
                others = indexers_list[cm][group_i]
                for j in others:
                    # There is no point in meeting oneself. It is not a pleasure.
                    if i == j:
                        pass
                    else:
                        if infectious[j] and not immune[i] and contacts[j, cm] > 0:
                            is_infection = _boolean_choice(infection_probs[cm])
                            if is_infection:
                                infection_counter[j] += 1
                                infected[i] = 1
                                immune[i] = True

        else:
            # get the probabilities for meeting another group which depend on the
            # individual's group.
            group_i = group_codes[i, cm]
            group_i_cdf = group_cdfs[cm][group_i]

            # Loop over each contact the individual has, sample the contact's group and
            # compute the sum of possible contacts in this group.
            n_contacts = contacts[i, cm]
            for _ in range(n_contacts):
                contact_takes_place = True
                group_j = _choose_other_group(groups_list[cm], cdf=group_i_cdf)
                choice_indices = indexers_list[cm][group_j]
                contacts_j = contacts[choice_indices, cm]

                j = _choose_other_individual(choice_indices, weights=contacts_j)

                if j < 0 or j == i:
                    contact_takes_place = False

                # If a contact takes place, find out if one individual got infected.
                if contact_takes_place:
                    contacts[i, cm] -= 1
                    contacts[j, cm] -= 1

                    if infectious[i] and not immune[j]:
                        is_infection = _boolean_choice(infection_probs[cm])
                        if is_infection:
                            infection_counter[i] += 1
                            infected[j] = 1
                            immune[j] = True

                    elif infectious[j] and not immune[i]:
                        is_infection = _boolean_choice(infection_probs[cm])
                        if is_infection:
                            infection_counter[j] += 1
                            infected[i] = 1
                            immune[i] = True

    missed = contacts

    return infected, infection_counter, immune, missed


@njit
def _choose_other_group(a, cdf):
    """Choose a group out of a, given cumulative choice probabilities."""
    u = np.random.uniform(0, 1)
    index = _get_index_refining_search(u, cdf)
    return a[index]


@njit
def _choose_other_individual(a, weights):
    """Return an element of a, if weights are not all zero, else return -1.

    Implementation is similar to `_choose_one_element`.

    :func:`numpy.argmax` returns the first index for multiple maximum values.

    Args:
        a (numpy.ndarray): 1d array of choices
        weights (numpy.ndarray): 1d array of weights.

    Returns:
        choice (int or float): An element of a or -1

    Example:
        >>> _choose_other_individual(np.arange(3), np.array([0, 0, 5]))
        2

        >>> _choose_other_individual(np.arange(3), np.zeros(3))
        -1


        >>> chosen = _choose_other_individual(np.arange(3), np.array([0.1, 0.5, 0.7]))
        >>> chosen in [0, 1, 2]
        True

    """
    cdf = weights.cumsum()
    sum_of_weights = cdf[-1]
    if sum_of_weights == 0:
        chosen = -1
    else:
        u = np.random.uniform(0, sum_of_weights)
        index = _get_index_refining_search(u, cdf)
        chosen = a[index]

    return chosen


@njit
def _get_index_refining_search(u, cdf):
    """Get the index of the first element in cdf that is larger than u.

    The algorithm does a refining search. We first iterate over cdf in
    larger steps to find a subset of cdf in which we have to look
    at each element.

    The step size in the first iteration is the square root of the
    length of cdf, which minimizes runtime in expectation if u is a
    uniform random variable.

    Args:
        u (float): A uniform random draw.
        cdf (numpy.ndarray): 1d array with cumulative probabilities.

    Returns:
        int: The selected index.

    Example:
        >>> cdf = np.array([0.1, 0.6, 1.0])
        >>> _get_index_refining_search(0, cdf)
        0
        >>> _get_index_refining_search(0.05, cdf)
        0
        >>> _get_index_refining_search(0.55, cdf)
        1
        >>> _get_index_refining_search(1, cdf)
        2

    """
    n_ind = len(cdf)
    highest_i = n_ind - 1
    i = 0
    step = int(np.sqrt(n_ind))

    while cdf[i] < u and i < highest_i:
        i = min(i + step, highest_i)

    i = max(0, i - step)

    while cdf[i] < u and i < highest_i:
        i += 1

    return i


@njit
def _boolean_choice(truth_prob):
    """Return True with probability truth_prob.

    Args:
        truth_prob (float): Must be between 0 and 1.

    Returns:
        bool

    Example:
        >>> _boolean_choice(1)
        True

        >>> _boolean_choice(0)
        False

    """
    u = np.random.uniform(0, 1)
    return u <= truth_prob


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
            # the keys of groups are not tuples if there was just one assort_by variable
            # but the group_codes_values are.
            group = group[0] if len(group) == 1 else group
            indexer.append(groups[group].to_numpy(dtype=DTYPE_INDEX))

    else:
        indexer = NumbaList()
        indexer.append(states.index.to_numpy(DTYPE_INDEX))

    return indexer


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
