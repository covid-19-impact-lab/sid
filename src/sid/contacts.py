from typing import Dict
from typing import List

import numba as nb
import numpy as np
import pandas as pd
from numba.typed import List as NumbaList
from sid.config import DTYPE_INDEX
from sid.config import DTYPE_INFECTED
from sid.config import DTYPE_INFECTION_COUNTER
from sid.config import DTYPE_N_CONTACTS
from sid.shared import boolean_choice
from sid.shared import factorize_assortative_variables
from sid.validation import validate_return_is_series_or_ndarray


def calculate_contacts(contact_models, contact_policies, states, params, date, seed):
    """Calculate number of contacts of different types.

    Args:
        contact_models (dict): See :ref:`contact_models`. They are already sorted.
        contact_policies (dict): See :ref:`policies`.
        states (pandas.DataFrame): See :ref:`states`.
        params (pandas.DataFrame): See :ref:`params`.
        date (pandas.Timestamp): The current date.
        seed (itertools.count)

    Returns:
        contacts (numpy.ndarray): DataFrame with one column for each contact model.

    """
    contacts = np.zeros((len(states), len(contact_models)), dtype=DTYPE_N_CONTACTS)

    for i, (model_name, model) in enumerate(contact_models.items()):
        loc = model.get("loc", params.index)
        func = model["model"]
        model_specific_contacts = func(
            states=states, params=params.loc[loc], seed=next(seed)
        )
        model_specific_contacts = validate_return_is_series_or_ndarray(
            model_specific_contacts, when=f"Contact model {model_name}"
        )
        for policy in contact_policies.values():
            if policy["affected_contact_model"] == model_name:
                if (policy["start"] <= date <= policy["end"]) and policy["is_active"](
                    states
                ):
                    if isinstance(policy["policy"], (float, int)):
                        model_specific_contacts *= policy["policy"]
                    else:
                        model_specific_contacts = policy["policy"](
                            states=states,
                            contacts=model_specific_contacts,
                            seed=next(seed),
                        )

        if not model["is_recurrent"]:
            model_specific_contacts = _sum_preserving_round(
                model_specific_contacts.to_numpy().astype(DTYPE_N_CONTACTS)
            )

        contacts[:, i] = model_specific_contacts

        # Dead people and ICU patients don't have contacts.
        contacts[states["needs_icu"] | states["dead"]] = 0

    return contacts


def calculate_infections_by_contacts(
    states,
    contacts,
    params,
    indexers,
    group_cdfs,
    code_to_contact_model,
    group_codes_names,
    seed,
):
    """Calculate infections from contacts.

    This function mainly converts the relevant parts from states and contacts into numpy
    arrays or other objects that are supported in numba nopython mode and then calls
    :func:`_calculate_infections_by_contacts_numba` to calculate the infections by
    contact.

    Args:
        states (pandas.DataFrame): see :ref:`states`.
        contacts (pandas.DataFrame): One column per contact_model. Same index as states.
        params (pandas.DataFrame): See :ref:`params`.
        indexers (dict): Dict of numba.Typed.List The i_th entry of the lists are the
            indices of the i_th group.
        group_cdfs (dict): dict of arrays of shape
            n_group, n_groups. probs[i, j] is the cumulative probability that an
            individual from group i meets someone from group j.
        group_codes_names (Dict[str, str]): The name of the group code column for each
            contact model.
        seed (itertools.count): Seed counter to control randomness.

    Returns:
        (tuple): Tuple containing

        - infected (pandas.Series): Boolean Series that is True for newly infected
          people.
        - n_has_additionally_infected (pandas.Series): A series with counts of people an
          individual has infected in this period by contact.
        - missed_contacts (pandas.DataFrame): Counts of missed contacts for each contact
          model.

    """
    is_recurrent = np.array([k not in group_cdfs for k in indexers])
    states = states.copy()
    infectious = states["infectious"].to_numpy(copy=True)
    immune = states["immune"].to_numpy(copy=True)
    group_codes = states[[group_codes_names[cm] for cm in indexers]].to_numpy()
    infect_probs = np.array(
        [params.loc[("infection_prob", cm, cm), "value"] for cm in indexers]
    )

    contacts = _reduce_contacts_with_infection_probs(
        contacts, is_recurrent, infect_probs, next(seed)
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

    (
        infected,
        infection_counter,
        immune,
        missed,
        was_infected_by,
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
    )

    infected = pd.Series(infected, index=states.index)
    n_has_additionally_infected = pd.Series(infection_counter, index=states.index)

    # Save missed contacts and set missed contacts of recurrent models to zero which
    # happens in :func:`_calculate_infections_by_contacts_numba` since ``missed`` is set
    # to ``contacts``.
    missed_contacts = pd.DataFrame(
        missed, columns=[f"missed_{name}" for name in indexers]
    )
    missed_contacts.loc[:, is_recurrent] = 0

    categories = {-1: "not_infected_by_contact", **code_to_contact_model}
    was_infected_by = pd.Series(
        pd.Categorical(was_infected_by, categories=list(categories)), index=states.index
    ).cat.rename_categories(new_categories=categories)

    return infected, n_has_additionally_infected, missed_contacts, was_infected_by


@nb.njit
def _reduce_contacts_with_infection_probs(contacts, is_recurrent, probs, seed):
    """Reduce the number of contacts stochastically.

    The remaining contacts have the interpretation that they would lead
    to an infection if one person is susceptible and one is infectious.

    Args:
        contacts (numpy.ndarray): 2d integer array with number of contacts per
            individual. There is one row per individual in the state and one column
            for each contact model where model["model"] != "meet_group".
        is_recurrent (numpy.ndarray): One entry per contact model.
        probs (numpy.ndarray): Infection probabilities. One entry per contact model.
        seed (int): The seed.

    Returns
        reduced_contacts (numpy.ndarray): Same shape as contacts. Equal to contacts for
            recurrent contact models. Less or equal to contacts otherwise.

    """

    contacts = contacts.copy()
    np.random.seed(seed)
    n_obs, n_contacts = contacts.shape
    for i in range(n_obs):
        for j in range(n_contacts):
            if not is_recurrent[j] and contacts[i, j] != 0:
                success = 0
                for _ in range(contacts[i, j]):
                    if boolean_choice(probs[j]):
                        success += 1
                contacts[i, j] = success
    return contacts


@nb.njit
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
    was_infected_by = np.full(len(contacts), -1, dtype=np.int16)

    n_obs, n_contact_models = contacts.shape
    # Loop over all individual-contact_model combinations
    for i in range(n_obs):
        for cm in range(n_contact_models):
            if is_recurrent[cm]:
                # We only check if i infects someone else from his group. Whether
                # he is infected by some j is only checked, when the main loop arrives
                # at j. This allows us to skip completely if i is not infectious or has
                # no contacts under contact model cm.
                group_i = group_codes[i, cm]
                if group_i >= 0 and infectious[i] and contacts[i, cm] > 0:
                    others = indexers_list[cm][group_i]
                    # extract infection probability into a variable for faster access
                    prob = infection_probs[cm]
                    for j in others:
                        # the case i == j is skipped by the next if condition because it
                        # never happens that i is infectious but not immune
                        if not immune[j] and contacts[j, cm] > 0:
                            is_infection = boolean_choice(prob)
                            if is_infection:
                                infection_counter[i] += 1
                                infected[j] = 1
                                immune[j] = True
                                was_infected_by[j] = cm

            else:
                # get the probabilities for meeting another group which depend on the
                # individual's group.
                group_i = group_codes[i, cm]
                group_i_cdf = group_cdfs[cm][group_i]

                # Loop over each contact the individual has, sample the contact's group
                # and compute the sum of possible contacts in this group.
                n_contacts = contacts[i, cm]
                for _ in range(n_contacts):
                    contact_takes_place = True
                    group_j = choose_other_group(groups_list[cm], cdf=group_i_cdf)
                    choice_indices = indexers_list[cm][group_j]
                    contacts_j = contacts[choice_indices, cm]

                    j = choose_other_individual(choice_indices, weights=contacts_j)

                    if j < 0 or j == i:
                        contact_takes_place = False

                    # If a contact takes place, find out if one individual got infected.
                    if contact_takes_place:
                        contacts[i, cm] -= 1
                        contacts[j, cm] -= 1

                        if infectious[i] and not immune[j]:
                            infection_counter[i] += 1
                            infected[j] = 1
                            immune[j] = True
                            was_infected_by[j] = cm

                        elif infectious[j] and not immune[i]:
                            infection_counter[j] += 1
                            infected[i] = 1
                            immune[i] = True
                            was_infected_by[i] = cm

    missed = contacts

    return infected, infection_counter, immune, missed, was_infected_by


@nb.njit
def choose_other_group(a, cdf):
    """Choose a group out of a, given cumulative choice probabilities.

    Note: This function is also used in sid-estimation.

    """
    u = np.random.uniform(0, 1)
    index = _get_index_refining_search(u, cdf)
    return a[index]


@nb.njit
def choose_other_individual(a, weights):
    """Return an element of a, if weights are not all zero, else return -1.

    Implementation is similar to `_choose_one_element`.

    :func:`numpy.argmax` returns the first index for multiple maximum values.

    Note: This function is also used in sid-estimation.

    Args:
        a (numpy.ndarray): 1d array of choices
        weights (numpy.ndarray): 1d array of weights.

    Returns:
        choice (int or float): An element of a or -1

    Example:
        >>> choose_other_individual(np.arange(3), np.array([0, 0, 5]))
        2

        >>> choose_other_individual(np.arange(3), np.zeros(3))
        -1


        >>> chosen = choose_other_individual(np.arange(3), np.array([0.1, 0.5, 0.7]))
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


@nb.njit
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


def create_group_indexer(
    states: pd.DataFrame, assort_by: Dict[str, List[str]], is_recurrent
) -> nb.typed.List:
    """Create the group indexer.

    The indexer is a list where the positions correspond to the group number defined by
    assortative variables. The values inside the list are one-dimensional integer arrays
    containing the indices of states belonging to the group.

    If there are no assortative variables, all individuals are assigned to a single
    group with code 0 and the indexer is a list where the first position contains all
    indices of states.

    For efficiency reasons, we assign each group a number instead of identifying by
    the values of the assort_by variables directly.

    Note: This function is also used in sid-estimation.

    Args:
        states (pandas.DataFrame): See :ref:`states`
        assort_by (List[str]): List of variables that influence matching probabilities.

    Returns:
        indexer (numba.typed.List): The i_th entry are the indices of the i_th group.

    """
    states = states.reset_index()
    if assort_by:
        groups = states.groupby(assort_by).groups
        _, group_codes_values = factorize_assortative_variables(
            states, assort_by, is_recurrent
        )

        indexer = NumbaList()
        for group in group_codes_values:
            # the keys of groups are not tuples if there was just one assort_by variable
            # but the group_codes_values are.
            group = group[0] if isinstance(group, tuple) and len(group) == 1 else group
            indexer.append(groups[group].to_numpy(dtype=DTYPE_INDEX))

    else:
        indexer = NumbaList()
        indexer.append(states.index.to_numpy(DTYPE_INDEX))

    return indexer


@nb.njit
def _sum_preserving_round(arr):
    """Round values in an array, preserving the sum as good as possible.

    The function loops over the elements of an array and collects the deviations to the
    nearest downward adjusted integer. Whenever the collected deviations reach a
    predefined threshold, +1 is added to the current element and the collected
    deviations are reduced by 1.

    Args:
        arr (numpy.ndarray): A one-dimensional array whose values should be rounded.

    Returns:
        arr (numpy.ndarray): Array with sum preserved rounded values.

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
