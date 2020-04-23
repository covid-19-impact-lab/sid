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
    duration,
    contact_policies=None,
    testing_policies=None,
    assort_by=None,
    seed=None,
):
    """Simulate the spread of an infectious disease.

    Args:
        params (pandas.DataFrame): DataFrame with parameters that influence the number
            of contacts, contagiousness and dangerousness of the disease, ... .
        initial_states (pandas.DataFrame): See :ref:`states`. Cannot contain the
            columnns "id" or "period" because those are used internally.
            The index of initial_states will be used as "id".
        initial_infections (pandas.Series): Series with the same index as states with
            initial infections.
        contact_models (dict): Dictionary of dictionaries where each dictionary
            describes a channel by which contacts can be formed.
            See :ref:`contact_models`.
        contact_policies (dict): Dict of dicts with contact. See :ref:`policies`.
        testing_policies (dict): Dict of dicts with testing policies. See
            :ref:`policies`.
        duration (int or dict): Duration can be an integer which will simulate data for
            `range(0, duration)` periods. It can also be a dictionary containing kwargs
            for :func:`pandas.date_range`.
        assort_by (list, optional): List of variable names. Contacts are assortative by
            these variables. These variables must be in the initial_states.
        seed (int, optional): Seed is used as the starting point of a sequence of seeds
            used to control randomness internally.

    Returns:
        simulation_results (pandas.DataFrame): The simulation results in form of a long
            DataFrame. The DataFrame contains the states of each period (see
            :ref:`states`) and a column called infections. The index has two levels. The
            first is the period. The second is the id. Id is the index of
            initial_states.

    """
    if assort_by is None:
        warnings.warn(
            "Specifying no variables in 'assort_by' significantly raises runtime. "
            "You can silence this warning by setting 'assort_by' to False."
        )

    assort_by = [] if not assort_by else assort_by
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
        assort_by,
    )

    duration = parse_duration(duration)

    states, index_names = _process_initial_states(initial_states, assort_by)
    states = draw_course_of_disease(states, params, seed)
    contact_policies = {
        key: _add_defaults_to_policy_dict(val, duration)
        for key, val in contact_policies.items()
    }
    states = update_states(states, initial_infections, params, seed)
    indexer = create_group_indexer(states, assort_by)
    first_probs = create_group_transition_probs(states, assort_by, params)

    to_concat = []
    for period in duration["iterable"]:
        states[duration["column_name"]] = period

        contacts = calculate_contacts(
            contact_models, contact_policies, states, params, period
        )
        infections, states = calculate_infections(
            states, contacts, params, indexer, first_probs, seed,
        )
        states = update_states(states, infections, params, seed)

        for contact_type in contacts.columns:
            states[contact_type] = contacts[contact_type]
        states["infections"] = infections
        to_concat.append(states.copy(deep=True))

    simulation_results = _process_simulation_results(to_concat, index_names, duration)

    return simulation_results


def _check_inputs(
    params,
    initial_states,
    initial_infections,
    contact_models,
    contact_policies,
    testing_policies,
    assort_by,
):
    """Check the user inputs."""
    if not isinstance(params, pd.DataFrame):
        raise ValueError("params must be a DataFrame.")

    if params.index.names != ["category", "name"]:
        raise ValueError("params must have an index with levels 'category' and 'name'.")

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

    for var in assort_by:
        if var not in initial_states.columns:
            raise KeyError(f"assort_by variable is not in initial states: {var}.")


def _add_defaults_to_policy_dict(pol_dict, duration):
    """Add defaults to a policy dictionary."""
    default = {
        "start": duration["start"],
        "end": duration["end"],
        "is_active": lambda states: True,
    }
    default.update(pol_dict)

    return default


def _process_initial_states(states, assort_by):
    """Process the initial states given by the user.

    Args:
        states (pandas.DataFrame): The user-defined initial states.
        assort_by (list, optional): List of variable names. Contacts are assortative by
            these variables.

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

    # Shuffle the states and reset the index because having a sorted range index could
    # speed up things.
    states = states.sample(frac=1, replace=False).reset_index()

    for col in BOOLEAN_STATE_COLUMNS:
        if col not in states.columns:
            states[col] = False

    for col in COUNTDOWNS:
        if col not in states.columns:
            states[col] = -1
        states[col] = states[col].astype(DTYPE_COUNTER)

    states["infection_counter"] = 0

    states["group_codes"], _ = factorize_assortative_variables(states, assort_by)

    return states, index_names


def _process_simulation_results(to_concat, index_names, duration):
    """Process the simulation results."""
    df = pd.concat(to_concat).set_index([duration["column_name"]] + index_names)

    return df
