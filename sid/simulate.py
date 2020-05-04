import warnings
from itertools import count

import numpy as np
import pandas as pd

from sid.config import BOOLEAN_STATE_COLUMNS
from sid.config import COUNTDOWNS
from sid.config import DTYPE_COUNTER
from sid.config import STATES_INDEX_DEFAULT_NAME
from sid.contacts import calculate_contacts
from sid.contacts import calculate_infections
from sid.contacts import create_group_indexer
from sid.contacts import create_group_transition_probs
from sid.parse_model import parse_duration
from sid.pathogenesis import draw_course_of_disease
from sid.shared import factorize_assortative_variables
from sid.update_states import update_states


def simulate(
    params,
    initial_states,
    initial_infections,
    contact_models,
    duration=None,
    contact_policies=None,
    testing_policies=None,
    seed=None,
):
    """Simulate the spread of an infectious disease.

    Args:
        params (pandas.DataFrame): DataFrame with parameters that influence the number
            of contacts, contagiousness and dangerousness of the disease, ... .
        initial_states (pandas.DataFrame): See :ref:`states`. Cannot contain the
            columnns "id", "date" or "period" because those are used internally. The
            index of initial_states will be used as "id".
        initial_infections (pandas.Series): Series with the same index as states with
            initial infections.
        contact_models (dict): Dictionary of dictionaries where each dictionary
            describes a channel by which contacts can be formed.
            See :ref:`contact_models`.
        contact_policies (dict): Dict of dicts with contact. See :ref:`policies`.
        testing_policies (dict): Dict of dicts with testing policies. See
            :ref:`policies`.
        duration (dict or None): Duration is a dictionary containing kwargs for
            :func:`pandas.date_range`.
        seed (int, optional): Seed is used as the starting point of a sequence of seeds
            used to control randomness internally.

    Returns:
        simulation_results (pandas.DataFrame): The simulation results in form of a long
            DataFrame. The DataFrame contains the states of each period (see
            :ref:`states`) and a column called infections. The index has two levels. The
            first is the period. The second is the id. Id is the index of
            initial_states.

    """

    contact_policies = {} if contact_policies is None else contact_policies
    testing_policies = {} if testing_policies is None else testing_policies
    seed = count(np.random.randint(0, 1_000_000)) if seed is None else count(seed)

    _check_inputs(
        params,
        initial_states,
        initial_infections,
        contact_models,
        contact_policies,
        testing_policies,
    )

    contact_models = _sort_contact_models(contact_models)
    assort_bys = _process_assort_bys(contact_models)
    states, index_names = _process_initial_states(initial_states, assort_bys)
    duration = parse_duration(duration)

    states = draw_course_of_disease(states, params, seed)
    contact_policies = {
        key: _add_defaults_to_policy_dict(val, duration)
        for key, val in contact_policies.items()
    }
    states = update_states(states, initial_infections, params, seed)

    indexers, first_probs = _prepare_assortative_matching(
        states, assort_bys, params, contact_models
    )

    to_concat = []
    for period, date in enumerate(duration["dates"]):
        states["date"] = date
        states["period"] = period

        contacts = calculate_contacts(
            contact_models, contact_policies, states, params, date
        )
        infections, states = calculate_infections(
            states, contacts, params, indexers, first_probs, seed,
        )
        states = update_states(states, infections, params, seed)

        for i, contact_model in enumerate(first_probs):
            states[contact_model] = contacts[:, i]
        states["infections"] = infections
        to_concat.append(states.copy(deep=True))

    simulation_results = _process_simulation_results(to_concat, index_names)

    return simulation_results


def _sort_contact_models(contact_models):
    """Sort the contact_models.

    First we have the contact models where model["model"] != "meet_group" in
    alphabetical order. Then the ones where model["model"] == "meet_group" in
    alphabetical order.

    Args:
        contact_models (dict): see :ref:`contact_models`

    Returns:
        dict: sorted copy of contact_models.

    """
    sorted_ = sorted(
        name for name, mod in contact_models.items() if mod["model"] != "meet_group"
    )
    sorted_ += sorted(
        name for name, mod in contact_models.items() if mod["model"] == "meet_group"
    )
    return {name: contact_models[name] for name in sorted_}


def _process_assort_bys(contact_models):
    """Set default values for assort_by variables and extract them into a dict.

    Args:
        contact_models (dict): see :ref:`contact_models`

    Returns:
        assort_bys (dict): Keys are names of contact models, values are lists with the
            assort_by variables of the model.

    """
    assort_bys = {}
    for model_name, model in contact_models.items():
        assort_by = model.get("assort_by", None)
        if assort_by is None:
            warnings.warn(
                "Not specifying 'assort_by' significantly raises runtime. "
                "You can silence this warning by setting 'assort_by' to False."
                f"in contact model {model_name}"
            )
            assort_by = []
        elif not assort_by:
            assort_by = []
        elif isinstance(assort_by, str):
            assort_by = [assort_by]
        elif isinstance(assort_by, list):
            pass
        else:
            raise ValueError(
                f"'assort_by' for '{model_name}' must be False str, or list."
            )

        assort_bys[model_name] = assort_by

    return assort_bys


def _check_inputs(
    params,
    initial_states,
    initial_infections,
    contact_models,
    contact_policies,
    testing_policies,
):
    """Check the user inputs."""
    if not isinstance(params, pd.DataFrame):
        raise ValueError("params must be a DataFrame.")

    if params.index.names != ["category", "subcategory", "name"]:
        raise ValueError(
            "params must have the index levels 'category', 'subcategory' and 'name'."
        )

    cd_names = sorted(COUNTDOWNS)
    gb = params.loc[cd_names].groupby(["category", "subcategory"])
    prob_sums = gb["value"].sum()
    problematic = prob_sums[~prob_sums.between(1 - 1e-08, 1 + 1e-08)].index.tolist()
    assert (
        len(problematic) == 0
    ), f"The following countdown probabilities don't add up to 1: {problematic}"

    if not isinstance(initial_states, pd.DataFrame):
        raise ValueError("initial_states must be a DataFrame.")

    if not isinstance(initial_infections, pd.Series):
        raise ValueError("initial_infections must be a pandas Series.")

    if not initial_infections.index.equals(initial_states.index):
        raise ValueError("initial_states and initial_infections must have same index.")

    if not isinstance(contact_models, dict):
        raise ValueError("contact_models must be a dictionary.")

    for cm_name, cm in contact_models.items():
        if not isinstance(cm, dict):
            raise ValueError(f"Each contact model must be a dictionary: {cm_name}.")

    if not isinstance(contact_policies, dict):
        raise ValueError("policies must be a dictionary.")

    for name, pol in contact_policies.items():
        if not isinstance(pol, dict):
            raise ValueError(f"Each policy must be a dictionary: {name}.")
        if name not in contact_models:
            raise KeyError(
                f"contact_policy refers to non existent contact model: {name}."
            )

    if testing_policies != {}:
        raise NotImplementedError


def _prepare_assortative_matching(states, assort_bys, params, contact_models):
    """Create indexers and first stage probabilities for assortative matching.

    Args:
        states (pd.DataFrame): see :ref:`states`.
        assort_bys (dict): Keys are names of contact models, values are lists with the
            assort_by variables of the model.
        params (pd.DataFrame): see :ref:`params`.
        contact_models (dict): see :ret:`contact_models`.

    returns:
        indexers (dict): Dict of numba.Typed.List The i_th entry of the lists are the
            indices of the i_th group.
        first_probs (dict): dict of arrays of shape
            n_group, n_groups. probs[i, j] is the probability that an individual from
            group i meets someone from group j.

    """
    indexers = {}
    first_probs = {}
    for model_name, assort_by in assort_bys.items():
        indexers[model_name] = create_group_indexer(states, assort_by)
        if contact_models[model_name]["model"] != "meet_group":
            first_probs[model_name] = create_group_transition_probs(
                states, assort_by, params, model_name
            )
    return indexers, first_probs


def _add_defaults_to_policy_dict(pol_dict, duration):
    """Add defaults to a policy dictionary."""
    default = {
        "start": duration["start"],
        "end": duration["end"],
        "is_active": lambda states: True,
    }
    default.update(pol_dict)

    return default


def _process_initial_states(states, assort_bys):
    """Process the initial states given by the user.

    Args:
        states (pandas.DataFrame): The user-defined initial states.
        assort_bys (list, optional): List of variable names. Contacts are assortative
            by these variables.

    Returns:
        states (pandas.DataFrame): Processed states.

    """
    states = states.copy()

    if np.any(states.isna()):
        raise ValueError("'initial_states' are not allowed to contain NaNs.")

    states = states.sort_index()
    if isinstance(states.index, pd.MultiIndex):
        index_names = states.index.names
    else:
        if not states.index.name:
            states.index.name = STATES_INDEX_DEFAULT_NAME
        index_names = [states.index.name]

    # reset the index because having a sorted range index could speed up things.
    states = states.reset_index()

    for col in BOOLEAN_STATE_COLUMNS:
        if col not in states.columns:
            states[col] = False

    for col in COUNTDOWNS:
        if col not in states.columns:
            states[col] = -1
        states[col] = states[col].astype(DTYPE_COUNTER)

    states["infection_counter"] = 0

    for model_name, assort_by in assort_bys.items():
        states[f"group_codes_{model_name}"], _ = factorize_assortative_variables(
            states, assort_by
        )

    return states, index_names


def _process_simulation_results(to_concat, index_names):
    """Process the simulation results."""
    df = pd.concat(to_concat).set_index(["date"] + index_names)

    return df
