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
    n_periods,
    contact_policies=None,
    testing_policies=None,
    assort_by=None,
):
    """Simulate the spread of covid-19.

    Args:
        params (pd.DataFrame): DataFrame with parameters that influence the number of
            contacts, contagiousness and dangerousness of the disease, ...
        initial_states (pd.DataFrame): See :ref:`states`. Cannot contain the columnns
            "id" or "period" because those are used internally.
        initial_infectios (pd.Series): Series with the same index as states with
            initial infections.
        contact_models (dict): List of dictionaries where each dictionary describes a
            channel by which contacts can be formed. See :ref:`contact_models`.
        contact_policies (dict): Dict of dicts with contact. See :ref:`policies`.
        testing_policies (dict): Dict of dicts with testing policies. See :ref:`policies`.
        n_periods (int): Number of periods to simulate.
        assort_by (list, optional): List of variable names. Contacts are assortative by
            these variables.


    Returns:
        pd.DataFrame: The simulation results in form of a long DataFrame. The DataFrame
            contains the states of each period (see :ref:`states`) and a column called
            infections. The index has two levels. The first is period. The second is id.
            Id is the index of initial_states.

    """
    assort_by = [] if assort_by is None else assort_by
    contact_policies = {} if contact_policies is None else contact_policies
    testing_policies = {} if testing_policies is None else testing_policies

    _check_inputs(
        params,
        initial_states,
        initial_infections,
        contact_models,
        contact_policies,
        testing_policies,
        n_periods,
        assort_by,
    )

    states = _process_states(initial_states, assort_by)
    states = draw_course_of_disease(states, params)
    contact_policies = {
        key: _process_pol_dict(val, n_periods) for key, val in contact_policies.items()
    }
    states = update_states(states, initial_infections, params)
    indexer = create_group_indexer(states, assort_by)
    first_probs = create_group_transition_probs(states, assort_by, params)

    to_concat = []
    for period in range(n_periods):
        contacts = calculate_contacts(
            contact_models, contact_policies, states, params, period
        )
        infections, states = calculate_infections(
            states, contacts, params, indexer, first_probs
        )
        states = update_states(states, infections, params)
        period_res = states.copy()
        period_res["period"] = period
        period_res["infections"] = infections
        to_concat.append(period_res.copy(deep=True))

    simulation_results = pd.concat(to_concat).set_index(["period", "id"])

    return simulation_results


def _check_inputs(
    params,
    initial_states,
    initial_infections,
    contact_models,
    contact_policies,
    testing_policies,
    n_periods,
    assort_by,
):
    if not isinstance(params, pd.DataFrame):
        raise ValueError("params must be a DataFrame")

    if list(params.index.names) != ["category", "name"]:
        raise ValueError("params must have an index with levels 'category' and 'name'")

    if not isinstance(initial_states, pd.DataFrame):
        raise ValueError("initial_states must be a DataFrame")

    if initial_states.index.nlevels != 1:
        raise ValueError("initial_states cannot have a MultiIndex")

    if not isinstance(initial_infections, pd.Series):
        raise ValueError("initial_infections must be a pandas Series.")

    if not initial_infections.index.equals(initial_states.index):
        raise ValueError("initial_states and initial_infections must have same index.")

    if not isinstance(contact_models, dict):
        raise ValueError("contact_models must be a dictionary.")

    for cm_name, cm in contact_models.items():
        if not isinstance(cm, dict):
            raise ValueError(f"Each contact model must be a dictionary: {cm_name}")

    if not isinstance(contact_policies, dict):
        raise ValueError("policies must be a dictionary.")

    for name, pol in contact_policies.items():
        if not isinstance(pol, dict):
            raise ValueError(f"Each policy must be a dictionary: {name}")
        if name not in contact_models:
            raise KeyError(
                f"contact_policy refers to non existent contact model: {name}"
            )

    if testing_policies != {}:
        raise NotImplementedError

    if not isinstance(n_periods, int) or n_periods <= 0:
        raise ValueError("n_periods must be a strictly positive integer.")

    for var in assort_by:
        if var not in initial_states.columns:
            raise KeyError(f"assort_by variable is not in initial states: {var}")


def _process_pol_dict(pol_dict, n_periods):
    default = {
        "start": 0,
        "end": n_periods,
        "is_active": lambda states: True,
    }
    default.update(pol_dict)
    return default


def _process_states(states, assort_by):
    # reset the index because having a sorted range index could speed up things
    states = states.copy()
    states.index.name = "id"
    states = states.sample(frac=1, replace=False).reset_index(drop=False)

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
