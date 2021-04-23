"""This module contains everything related to contacts and matching."""
import itertools
from typing import Any
from typing import Dict
from typing import Tuple

import numba as nb
import numpy as np
import pandas as pd
from numba.typed import List as NumbaList
from sid.config import DTYPE_CONTACT_MODEL_CODES
from sid.config import DTYPE_INDEX
from sid.config import DTYPE_INFECTION_COUNTER
from sid.config import DTYPE_N_CONTACTS
from sid.config import DTYPE_VIRUS_STRAIN
from sid.shared import boolean_choice
from sid.shared import separate_contact_model_names
from sid.validation import validate_return_is_series_or_ndarray
from sid.virus_strains import combine_first_factorized_infections
from sid.virus_strains import factorize_categorical_infections


def calculate_contacts(
    contact_models: Dict[str, Dict[str, Any]],
    states: pd.DataFrame,
    params: pd.DataFrame,
    seed: itertools.count,
) -> pd.DataFrame:
    """Calculate number of contacts of different types.

    Args:
        contact_models (Dict[str, Dict[str, Any]]): See :ref:`contact_models`. They are
            already sorted.
        states (pandas.DataFrame): See :ref:`states`.
        params (pandas.DataFrame): See :ref:`params`.
        seed (itertools.count): The seed counter.

    Returns:
        contacts (pandas.DataFrame): A DataFrame with columns for each active contact
            model and the number of contacts.

    """
    contacts = pd.DataFrame(index=states.index)

    for model_name, model in contact_models.items():
        loc = model.get("loc", params.index)
        func = model["model"]
        model_specific_contacts = func(
            states=states, params=params.loc[loc], seed=next(seed)
        )

        model_specific_contacts = validate_return_is_series_or_ndarray(
            model_specific_contacts, model_name, "contact_models", states.index
        )

        if model["is_recurrent"]:
            model_specific_contacts = model_specific_contacts.astype(bool)
        contacts[model_name] = model_specific_contacts

    return contacts


def calculate_infections_by_contacts(
    states: pd.DataFrame,
    recurrent_contacts: np.ndarray,
    random_contacts: np.ndarray,
    params: pd.DataFrame,
    indexers: Dict[str, nb.typed.List],
    assortative_matching_cum_probs: nb.typed.List,
    contact_models: Dict[str, Dict[str, Any]],
    group_codes_info: Dict[str, Dict[str, Any]],
    susceptibility_factor: np.ndarray,
    virus_strains: Dict[str, Any],
    seasonality_factor: float,
    seed: itertools.count,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Calculate infections from contacts.

    This function mainly converts the relevant parts from states and contacts into numpy
    arrays or other objects that are supported in numba nopython mode and then calls
    :func:`_calculate_infections_by_contacts_numba` to calculate the infections by
    contact.

    Args:
        states (pandas.DataFrame): see :ref:`states`.
        recurrent_contacts (numpy.ndarray): An array with boolean entries for each
            person and recurrent contact model.
        random_contacts (numpy.ndarray): An array with integer entries indicating the
            number of contacts for each person and random contact model.
        params (pandas.DataFrame): See :ref:`params`.
        indexers (Dict[str, numba.typed.List]): The indexer is a dictionary with one
            entry for recurrent and random contact models. The values are Numba lists
            containing Numba lists for each contact model. Each list holds indices for
            each group in the contact model.
        assortative_matching_cum_probs (numba.typed.List): The list contains one entry
            for each random contact model. Each entry holds a ``n_groups * n_groups``
            transition matrix where ``probs[i, j]`` is the cumulative probability that
            an individual from group ``i`` meets someone from group ``j``.
        contact_models (Dict[str, Dict[str, Any]]): The contact models.
        group_codes_info (Dict[str, Dict[str, Any]]): The name of the group code column
            for each contact model.
        susceptibility_factor (np.ndarray): A multiplier which scales the infection
            probability due to susceptibility.
        virus_strains (Dict[str, Any]): A dictionary with the keys ``"names"`` and
            ``"factors"`` holding the different contagiousness factors of multiple
            viruses.
        seasonality_factor (float): A multiplier which scales the infection
            probabilities due to seasonality.
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
    states = states.copy()
    infectious = states["infectious"].to_numpy(copy=True)
    immune = states["immune"].to_numpy(copy=True)
    virus_strain, _ = factorize_categorical_infections(
        states["virus_strain"], virus_strains["names"]
    )

    recurrent_models, random_models = separate_contact_model_names(contact_models)

    group_codes_recurrent = states[
        [group_codes_info[cm]["name"] for cm in recurrent_models]
    ].to_numpy()
    group_codes_random = states[
        [group_codes_info[cm]["name"] for cm in random_models]
    ].to_numpy()

    seasonal_infection_probabilities_recurrent = (
        np.array(
            [params.loc[("infection_prob", cm, cm), "value"] for cm in recurrent_models]
        )
        * seasonality_factor
    )
    seasonal_infection_probabilities_random = (
        np.array(
            [params.loc[("infection_prob", cm, cm), "value"] for cm in random_models]
        )
        * seasonality_factor
    )

    infection_counter = np.zeros(len(states), dtype=DTYPE_INFECTION_COUNTER)

    if recurrent_models:
        (
            newly_infected_recurrent,
            infection_counter,
            immune,
            was_infected_by_recurrent,
        ) = _calculate_infections_by_recurrent_contacts(
            recurrent_contacts=recurrent_contacts,
            infectious=infectious,
            immune=immune,
            virus_strain=virus_strain,
            group_codes=group_codes_recurrent,
            indexers=indexers["recurrent"],
            infection_probs=seasonal_infection_probabilities_recurrent,
            susceptibility_factor=susceptibility_factor,
            contagiousness_factor=virus_strains["factors"],
            infection_counter=infection_counter,
            seed=next(seed),
        )
    else:
        was_infected_by_recurrent = None
        newly_infected_recurrent = np.full(len(states), -1, dtype=DTYPE_VIRUS_STRAIN)

    if random_models:
        random_contacts = _reduce_random_contacts_with_infection_probs(
            random_contacts, seasonal_infection_probabilities_random, next(seed)
        )

        (
            newly_infected_random,
            infection_counter,
            immune,
            missed,
            was_infected_by_random,
        ) = _calculate_infections_by_random_contacts(
            random_contacts=random_contacts,
            infectious=infectious,
            immune=immune,
            virus_strain=virus_strain,
            group_codes=group_codes_random,
            assortative_matching_cum_probs=assortative_matching_cum_probs,
            indexers=indexers["random"],
            susceptibility_factor=susceptibility_factor,
            contagiousness_factor=virus_strains["factors"],
            infection_counter=infection_counter,
            seed=next(seed),
        )

        missed_contacts = pd.DataFrame(
            missed, columns=[f"missed_{name}" for name in random_models]
        )
    else:
        missed_contacts = None
        was_infected_by_random = None
        newly_infected_random = np.full(len(states), -1, dtype=DTYPE_VIRUS_STRAIN)

    was_infected_by = _consolidate_reason_of_infection(
        was_infected_by_recurrent, was_infected_by_random, contact_models
    )
    was_infected_by.index = states.index
    n_has_additionally_infected = pd.Series(infection_counter, index=states.index)

    combined_newly_infected = combine_first_factorized_infections(
        newly_infected_recurrent, newly_infected_random
    )
    newly_infected = pd.Series(combined_newly_infected, index=states.index)

    return newly_infected, n_has_additionally_infected, missed_contacts, was_infected_by


@nb.njit  # pragma: no cover
def _reduce_random_contacts_with_infection_probs(
    random_contacts: np.ndarray, probs: np.ndarray, seed: int
) -> np.ndarray:
    """Reduce the number of random contacts stochastically.

    The remaining random contacts have the interpretation that they would lead to an
    infection if one person is infectious and the other person is susceptible, the
    person has the highest susceptibility in the population according to the
    ``susceptibility_factor``, and the infected person is affected by the
    most contagious virus strain according to the ``virus_strain``.

    The copy is necessary as we need the original random contacts for debugging.

    Args:
        random_contacts (numpy.ndarray): An integer array containing the number of
            contacts per individual for each random (non-recurrent) contact model.
        probs (numpy.ndarray): An array containing one infection probability for each
            random contact model.
        seed (int): The seed.

    Returns
        random_contacts (numpy.ndarray): Same shape as contacts. Equal to contacts for
            recurrent contact models. Less or equal to contacts otherwise.

    """
    np.random.seed(seed)
    random_contacts = random_contacts.copy()

    n_obs, n_contacts = random_contacts.shape
    for i in range(n_obs):
        for j in range(n_contacts):
            if random_contacts[i, j] != 0:
                success = 0
                for _ in range(random_contacts[i, j]):
                    if boolean_choice(probs[j]):
                        success += 1
                random_contacts[i, j] = success
    return random_contacts


@nb.njit  # pragma: no cover
def _calculate_infections_by_recurrent_contacts(
    recurrent_contacts: np.ndarray,
    infectious: np.ndarray,
    immune: np.ndarray,
    virus_strain: np.ndarray,
    group_codes: np.ndarray,
    indexers: nb.typed.List,
    infection_probs: np.ndarray,
    susceptibility_factor: np.ndarray,
    contagiousness_factor: np.ndarray,
    infection_counter: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray]:
    """Match recurrent contacts and record infections.

    Args:
        recurrent_contacts (numpy.ndarray): 2d integer array with number of contacts per
            individual. There is one row per individual in the state and one column
            for each contact model where model["model"] != "meet_group".
        infectious (numpy.ndarray): 1d boolean array that indicates if a person is
            infectious. This is not directly changed after an infection.
        immune (numpy.ndarray): 1d boolean array that indicates if a person is immune.
        virus_strain (numpy.ndarray)
        group_codes (numpy.ndarray): 2d integer array with the index of the group used
            in the first stage of matching.
        indexers (numba.typed.List): Nested typed list. The i_th entry of the inner
            lists are the indices of the i_th group. There is one inner list per contact
            model.
        infection_probs (numpy.ndarray): An array containing the infection probabilities
            for each recurrent contact model.
        susceptibility_factor (np.ndarray): A multiplier which scales the infection
            probability.
        virus_strains_multiplier
        infection_counter (numpy.ndarray): An array counting infection caused by an
            individual.
        seed (int): Seed value to control randomness.

    Returns:
        (tuple) Tuple containing

        - newly_infected (numpy.ndarray): Boolean array that is True for individuals
          who got newly infected.
        - infection_counter (numpy.ndarray): 1d integer array
        - immune (numpy.ndarray): 1-D boolean array that indicates if a person is
          immune.
        - was_infected_by (numpy.ndarray): An array indicating the contact model which
          caused the infection.

    """
    np.random.seed(seed)

    n_individuals, n_recurrent_contact_models = recurrent_contacts.shape
    was_infected_by = np.full(n_individuals, -1, dtype=DTYPE_CONTACT_MODEL_CODES)
    newly_infected = np.full(n_individuals, -1, dtype=DTYPE_VIRUS_STRAIN)

    for i in range(n_individuals):
        for cm in range(n_recurrent_contact_models):
            # We only check if i infects someone else from his group. Whether
            # he is infected by some j is only checked, when the main loop arrives
            # at j. This allows us to skip completely if i is not infectious or has
            # no contacts under contact model cm.
            group_i = group_codes[i, cm]
            if group_i >= 0 and infectious[i] and recurrent_contacts[i, cm] > 0:
                others = indexers[cm][group_i]
                # extract infection probability into a variable for faster access
                base_probability = infection_probs[cm]
                for j in others:
                    # the case i == j is skipped by the next if condition because it
                    # never happens that i is infectious but not immune
                    if not immune[j] and recurrent_contacts[j, cm] > 0:
                        # j is infected depending on its own susceptibility.
                        virus_strain_i = virus_strain[i]
                        contagiousness_factor_i = contagiousness_factor[virus_strain_i]
                        individual_infection_risk = (
                            base_probability
                            * susceptibility_factor[j]
                            * contagiousness_factor_i
                        )

                        is_infection = boolean_choice(individual_infection_risk)
                        if is_infection:
                            infection_counter[i] += 1
                            newly_infected[j] = virus_strain_i
                            immune[j] = True
                            was_infected_by[j] = cm

    return newly_infected, infection_counter, immune, was_infected_by


@nb.njit  # pragma: no cover
def _calculate_infections_by_random_contacts(
    random_contacts: np.ndarray,
    infectious: np.ndarray,
    immune: np.ndarray,
    virus_strain: np.ndarray,
    group_codes: np.ndarray,
    assortative_matching_cum_probs: nb.typed.List,
    indexers: nb.typed.List,
    susceptibility_factor: np.ndarray,
    contagiousness_factor: np.ndarray,
    infection_counter: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray]:
    """Match random contacts and record infections.

    Args:
        random_contacts (numpy.ndarray): 2d integer array with number of contacts per
            individual. There is one row per individual in the state and one column
            for each contact model where model["model"] != "meet_group".
        infectious (numpy.ndarray): 1d boolean array that indicates if a person is
            infectious. This is not directly changed after an infection.
        immune (numpy.ndarray): 1d boolean array that indicates if a person is immune.
        virus_strain
        group_codes (numpy.ndarray): 2d integer array with the index of the group used
            in the first stage of matching.
        assortative_matching_cum_probs (numba.typed.List): List of arrays of shape
            n_group, n_groups. arr[i, j] is the cumulative probability that an
            individual from group i meets someone from group j.
        indexers (numba.typed.List): Nested typed list. The i_th entry of the inner
            lists are the indices of the i_th group. There is one inner list per contact
            model.
        susceptibility_factor (np.ndarray): A multiplier which scales the infection
            probability.
        contagiousness_factor
        infection_counter (numpy.ndarray): An array counting infection caused by an
            individual.
        seed (int): Seed value to control randomness.

    Returns:
        (tuple) Tuple containing

        - newly_infected (numpy.ndarray): Indicates newly infected individuals.
        - infection_counter (numpy.ndarray): Counts the number of infected individuals.
        - immune (numpy.ndarray): Indicates immune individuals.
        - missed (numpy.ndarray): Matrix which contains unmatched random contacts.

    """
    np.random.seed(seed)

    n_individuals, n_random_contact_models = random_contacts.shape
    groups_list = [np.arange(len(gp)) for gp in assortative_matching_cum_probs]
    was_infected_by = np.full(n_individuals, -1, dtype=DTYPE_CONTACT_MODEL_CODES)
    newly_infected = np.full(n_individuals, -1, dtype=DTYPE_VIRUS_STRAIN)

    # Loop over all individual-contact_model combinations
    for i in range(n_individuals):
        for cm in range(n_random_contact_models):
            # get the probabilities for meeting another group which depends on the
            # individual's group.
            group_i = group_codes[i, cm]
            group_i_cdf = assortative_matching_cum_probs[cm][group_i]

            # Loop over each contact the individual has, sample the contact's group
            # and compute the sum of possible contacts in this group.
            n_contacts = random_contacts[i, cm]
            for _ in range(n_contacts):
                contact_takes_place = True
                group_j = choose_other_group(groups_list[cm], cdf=group_i_cdf)
                choice_indices = indexers[cm][group_j]
                contacts_j = random_contacts[choice_indices, cm]

                j = choose_other_individual(choice_indices, weights=contacts_j)

                if j < 0 or j == i:
                    contact_takes_place = False

                # If a contact takes place, find out if one individual got infected.
                if contact_takes_place:
                    random_contacts[i, cm] -= 1
                    random_contacts[j, cm] -= 1

                    if infectious[i] and not immune[j]:
                        virus_strain_i = virus_strain[i]
                        contagiousness_factor_i = contagiousness_factor[virus_strain_i]
                        adjusted_individual_infection_risk_j = (
                            susceptibility_factor[j] * contagiousness_factor_i
                        )

                        is_infection = boolean_choice(
                            adjusted_individual_infection_risk_j
                        )
                        if is_infection:
                            infection_counter[i] += 1
                            newly_infected[j] = virus_strain_i
                            immune[j] = True
                            was_infected_by[j] = cm

                    elif infectious[j] and not immune[i]:
                        virus_strain_j = virus_strain[j]
                        contagiousness_factor_j = contagiousness_factor[virus_strain_j]
                        adjusted_individual_infection_risk_i = (
                            susceptibility_factor[i] * contagiousness_factor_j
                        )

                        is_infection = boolean_choice(
                            adjusted_individual_infection_risk_i
                        )
                        if is_infection:
                            infection_counter[j] += 1
                            newly_infected[i] = virus_strain_j
                            immune[i] = True
                            was_infected_by[i] = cm

    missed = random_contacts

    return newly_infected, infection_counter, immune, missed, was_infected_by


@nb.njit  # pragma: no cover
def choose_other_group(a, cdf):
    """Choose a group out of a, given cumulative choice probabilities.

    Note: This function is also used in sid-germany.

    """
    u = np.random.uniform(0, 1)
    index = _get_index_refining_search(u, cdf)
    return a[index]


@nb.njit  # pragma: no cover
def choose_other_individual(a, weights):
    """Return an element of a, if weights are not all zero, else return -1.

    Implementation is similar to `_choose_one_element`.

    :func:`numpy.argmax` returns the first index for multiple maximum values.

    Note: This function is also used in sid-germany.

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


@nb.njit  # pragma: no cover
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
    states: pd.DataFrame,
    group_code_name: str,
) -> nb.typed.List:
    """Create the group indexer.

    The indexer is a list where the positions correspond to the group number defined by
    assortative variables. The values inside the list are one-dimensional integer arrays
    containing the indices of states belonging to the group.

    If there are no assortative variables, all individuals are assigned to a single
    group with code 0 and the indexer is a list where the first position contains all
    indices of states.

    When an assortative variable is factorized, missing values receive -1 as the group
    key. Thus, we remove all negative group keys from the indexer.

    Args:
        states (pandas.DataFrame): See :ref:`states`
        group_code_name (str): The name of the group codes belonging to this contact
            model.

    Returns:
        indexer (numba.typed.List): The i_th entry are the indices of the i_th group.

    """
    indices = states.groupby(group_code_name, sort=True).indices
    reduced_indices = {idx: indices[idx] for idx in sorted(indices) if idx >= 0}

    indexer = NumbaList()
    for indices in reduced_indices.values():
        indexer.append(indices.astype(DTYPE_INDEX))

    return indexer


def post_process_contacts(contacts, states, contact_models):
    """Post-process contacts.

    The number of contacts for random models is rounded such that the sum of contacts is
    preserved.

    """
    recurrent_models, random_models = separate_contact_model_names(contact_models)

    # Dead people and ICU patients don't have contacts.
    has_no_contacts = states["needs_icu"] | states["dead"]
    contacts.loc[has_no_contacts, random_models] = 0
    contacts.loc[has_no_contacts, recurrent_models] = False

    if random_models:
        integers = (
            contacts[random_models].select_dtypes(include=np.integer).columns.tolist()
        )
        if integers:
            contacts[integers] = contacts[integers].astype(DTYPE_N_CONTACTS)

        floats = (
            contacts[random_models].select_dtypes(include=np.floating).columns.tolist()
        )
        for float_col in floats:
            contacts.loc[:, float_col] = _sum_preserving_round(
                contacts[float_col].to_numpy()
            ).astype(DTYPE_N_CONTACTS)

        no_integers = (
            contacts[random_models].select_dtypes(exclude=np.integer).columns.tolist()
        )
        if no_integers:
            dtype_mapping = contacts[no_integers].dtypes.to_dict()
            raise ValueError(
                "The following contacts should be integers or floats, but they have a "
                f"different dtype.\n\n{dtype_mapping}"
            )
        random_contacts = contacts[random_models]
    else:
        random_contacts = None

    if recurrent_models:
        recurrent_contacts = contacts[recurrent_models].astype(dtype="bool")
    else:
        recurrent_contacts = None

    return recurrent_contacts, random_contacts


@nb.njit  # pragma: no cover
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


def _consolidate_reason_of_infection(
    was_infected_by_recurrent: np.ndarray,
    was_infected_by_random: np.ndarray,
    contact_models: Dict[str, Dict[str, Any]],
) -> pd.Series:
    """Consolidate reason of infection."""
    n_individuals = (
        len(was_infected_by_recurrent)
        if was_infected_by_recurrent is not None
        else len(was_infected_by_random)
    )
    was_infected_by = np.full(n_individuals, -1)
    contact_model_to_code = {c: i for i, c in enumerate(contact_models)}

    if was_infected_by_random is not None:
        random_pos_to_code = {
            i: contact_model_to_code[c]
            for i, c in enumerate(contact_models)
            if not contact_models[c]["is_recurrent"]
        }

        mask = was_infected_by_random >= 0
        was_infected_by[mask] = _numpy_replace(
            was_infected_by_random[mask], random_pos_to_code
        )

    if was_infected_by_recurrent is not None:
        recurrent_pos_to_code = {
            i: contact_model_to_code[c]
            for i, c in enumerate(contact_models)
            if contact_models[c]["is_recurrent"]
        }

        mask = was_infected_by_recurrent >= 0
        was_infected_by[mask] = _numpy_replace(
            was_infected_by_recurrent[mask], recurrent_pos_to_code
        )

    categories = {-1: "not_infected_by_contact", **dict(enumerate(contact_models))}
    was_infected_by = pd.Series(
        pd.Categorical(was_infected_by, categories=list(categories))
    ).cat.rename_categories(new_categories=categories)

    return was_infected_by


def _numpy_replace(x: np.ndarray, replace_to: Dict[Any, Any]):
    """Replace values in a NumPy array with a dictionary."""
    sort_idx = np.argsort(list(replace_to))
    idx = np.searchsorted(list(replace_to), x, sorter=sort_idx)
    out = np.asarray(list(replace_to.values()))[sort_idx][idx]
    return out
