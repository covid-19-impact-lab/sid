import functools
import itertools as it
import shutil
import warnings
from pathlib import Path

import dask.dataframe as dd
import numpy as np
from sid.config import BOOLEAN_STATE_COLUMNS
from sid.config import SAVED_COLUMNS
from sid.contacts import calculate_contacts
from sid.contacts import calculate_infections_by_contacts
from sid.contacts import create_group_indexer
from sid.countdowns import COUNTDOWNS
from sid.events import calculate_infections_by_events
from sid.matching_probabilities import create_group_transition_probs
from sid.parse_model import parse_duration
from sid.shared import factorize_assortative_variables
from sid.shared import process_optional_state_columns
from sid.testing_allocation import allocate_tests
from sid.testing_allocation import update_pending_tests
from sid.testing_demand import calculate_demand_for_tests
from sid.testing_processing import process_tests
from sid.update_states import update_states
from sid.validation import validate_models
from sid.validation import validate_params
from sid.validation import validate_prepared_initial_states


def get_simulate_func(
    params,
    initial_states,
    contact_models,
    duration=None,
    events=None,
    contact_policies=None,
    testing_demand_models=None,
    testing_allocation_models=None,
    testing_processing_models=None,
    seed=None,
    path=None,
    saved_columns=None,
    optional_state_columns=None,
):
    """Get a function that simulates the spread of an infectious disease.

    The resulting function only depends on parameters. The computational time it takes
    to process the user input is only incurred once in :func:`get_simulate_func` and not
    when the resulting function is called.

    Args:
        params (pandas.DataFrame): DataFrame with parameters that influence the number
            of contacts, contagiousness and dangerousness of the disease, ... .
        initial_states (pandas.DataFrame): The output of
            :func:`sid.spread_infections.spreat_out_infections` or from a
            past simulation or estimation.
        contact_models (dict): Dictionary of dictionaries where each dictionary
            describes a channel by which contacts can be formed. See
            :ref:`contact_models`.
        duration (dict or None): Duration is a dictionary containing kwargs for
            :func:`pandas.date_range`.
        events (dict or None): Dictionary of events which cause infections.
        contact_policies (dict): Dict of dicts with contact. See :ref:`policies`.
        testing_demand_models (dict): Dict of dicts with demand models for tests. See
            :ref:`testing_demand_models` for more information.
        testing_allocation_models (dict): Dict of dicts with allocation models for
            tests. See :ref:`testing_allocation_models` for more information.
        testing_processing_models (dict): Dict of dicts with processing models for
            tests. See :ref:`testing_processing_models` for more information.
        seed (int, optional): Seed is used as the starting point of a sequence of seeds
            used to control randomness internally.
        path (str or pathlib.Path): Path to the directory where the simulated data is
            stored.
        saved_columns (dict or None): Dictionary with categories of state columns.
            The corresponding values can be True, False or Lists with columns that
            should be saved. Typically, during estimation you only want to save exactly
            what you need to calculate moments to make the simulation and calculation
            of moments faster. The categories are "initial_states", "disease_states",
            "testing_states", "countdowns", "contacts", "countdown_draws", "group_codes"
            "infection_reason" and "other".
        optional_state_columns (dict): Dictionary with categories of state columns
            that can additionally be added to the states dataframe, either for use in
            contact models and policies or to be saved. Most types of columns are added
            by default, but some of them are costly to add and thus only added when
            needed. Columns that are not in the state but specified in ``saved_columns``
            will not be saved. The categories are "contacts" and "reason_for_infection".

    Returns:
        callable: Simulates dataset based on parameters.

    """
    events = {} if events is None else events
    contact_policies = {} if contact_policies is None else contact_policies
    if testing_demand_models is None:
        testing_demand_models = {}
    if testing_allocation_models is None:
        testing_allocation_models = {}
    if testing_processing_models is None:
        testing_processing_models = {}

    params = params.copy(deep=True)
    initial_states = initial_states.copy(deep=True)

    validate_params(params)
    validate_prepared_initial_states(initial_states)
    validate_models(
        contact_models,
        contact_policies,
        testing_demand_models,
        testing_allocation_models,
        testing_processing_models,
    )

    contact_models = _sort_contact_models(contact_models)
    assort_bys = _process_assort_bys(contact_models)
    initial_states = _process_initial_states(initial_states, assort_bys)
    duration = parse_duration(duration)
    contact_policies = {
        key: _add_defaults_to_policy_dict(val, duration)
        for key, val in contact_policies.items()
    }

    indexers = _prepare_assortative_matching_indexers(initial_states, assort_bys)

    optional_state_columns = process_optional_state_columns(optional_state_columns)
    cols_to_keep = _process_saved_columns(
        saved_columns, initial_states.columns, contact_models, optional_state_columns
    )

    path = _create_output_directory(path)

    sim_func = functools.partial(
        _simulate,
        initial_states=initial_states,
        assort_bys=assort_bys,
        contact_models=contact_models,
        duration=duration,
        events=events,
        contact_policies=contact_policies,
        testing_demand_models=testing_demand_models,
        testing_allocation_models=testing_allocation_models,
        testing_processing_models=testing_processing_models,
        seed=seed,
        path=path,
        columns_to_keep=cols_to_keep,
        indexers=indexers,
        optional_state_columns=optional_state_columns,
    )
    return sim_func


def _simulate(
    params,
    initial_states,
    assort_bys,
    contact_models,
    duration,
    events,
    contact_policies,
    testing_demand_models,
    testing_allocation_models,
    testing_processing_models,
    seed,
    path,
    columns_to_keep,
    indexers,
    optional_state_columns,
):
    """Simulate the spread of an infectious disease.

    Args:
        params (pandas.DataFrame): DataFrame with parameters that influence the number
            of contacts, contagiousness and dangerousness of the disease, ... .
        initial_states (pandas.DataFrame): The output of
            :func:`sid.preparation.prepare_initial_states` or states from a former
            simulation or estimation.
        contact_models (dict): Dictionary of dictionaries where each dictionary
            describes a channel by which contacts can be formed. See
            :ref:`contact_models`.
        duration (dict): Duration is a dictionary containing kwargs for
            :func:`pandas.date_range`.
        events (dict): Dictionary of events which cause infections.
        contact_policies (dict): Dict of dicts with policies. See :ref:`policies`.
        testing_demand_models (dict): Dict of dicts with demand models for tests. See
            :ref:`testing_demand_models` for more information.
        testing_allocation_models (dict): Dict of dicts with allocation models for
            tests. See :ref:`testing_allocation_models` for more information.
        testing_processing_models (dict): Dict of dicts with processing models for
            tests. See :ref:`testing_processing_models` for more information.
        seed (int): Seed is used as the starting point of a sequence of seeds
            used to control randomness internally.
        path (pathlib.Path): Path to the directory where the simulated data is stored.
        columns_to_keep (list): Columns of states that will be saved in each period.
        optional_state_columns (dict): Dictionary with categories of state columns
            that can additionally be added to the states dataframe, either for use in
            contact models and policies or to be saved. Most types of columns are added
            by default, but some of them are costly to add and thus only added when
            needed. Columns that are not in the state but specified in ``saved_columns``
            will not be saved. The categories are "contacts" and "reason_for_infection".

    Returns:
        simulation_results (dask.dataframe): The simulation results in form of a long
            :class:`dask.dataframe`. The DataFrame contains the states of each period
            (see :ref:`states`) and a column called newly_infected.

    """
    seed = it.count(np.random.randint(0, 1_000_000)) if seed is None else it.count(seed)
    states = initial_states

    cum_probs = _prepare_assortative_matching_probabilities(
        states, assort_bys, params, contact_models
    )

    code_to_contact_model = dict(zip(range(len(contact_models)), list(contact_models)))

    for date in duration["dates"]:
        states["date"] = date

        contacts = calculate_contacts(
            contact_models=contact_models,
            contact_policies=contact_policies,
            states=states,
            params=params,
            date=date,
        )

        (
            newly_infected_contacts,
            n_has_additionally_infected,
            newly_missed_contacts,
            was_infected_by_model,
        ) = calculate_infections_by_contacts(
            states=states,
            contacts=contacts,
            params=params,
            indexers=indexers,
            group_cdfs=cum_probs,
            code_to_contact_model=code_to_contact_model,
            seed=seed,
        )
        newly_infected_events = calculate_infections_by_events(states, params, events)

        if testing_demand_models:
            demands_test = calculate_demand_for_tests(
                states, testing_demand_models, params, date, seed
            )
            allocated_tests = allocate_tests(
                states, testing_allocation_models, demands_test, params, date
            )

            states = update_pending_tests(states, allocated_tests)

            to_be_processed_tests = process_tests(
                states, testing_processing_models, params, date
            )
        else:
            to_be_processed_tests = None

        states = update_states(
            states=states,
            newly_infected_contacts=newly_infected_contacts,
            newly_infected_events=newly_infected_events,
            params=params,
            seed=seed,
            optional_state_columns=optional_state_columns,
            n_has_additionally_infected=n_has_additionally_infected,
            indexers=indexers,
            contacts=contacts,
            to_be_processed_test=to_be_processed_tests,
            was_infected_by_model=was_infected_by_model,
        )

        _dump_periodic_states(states, columns_to_keep, path, date)

    categoricals = {
        column: initial_states[column].cat.categories.shape[0]
        for column in initial_states.select_dtypes("category").columns
        if column in columns_to_keep
    }
    simulation_results = _return_dask_dataframe(path, categoricals)

    return simulation_results


def _create_output_directory(path):
    """Determine the output directory for the data.

    The user can provide a path or a default path is chosen. If the user's path leads to
    an non-empty directory, it is removed and newly created.

    Args:
        path (pathlib.Path or None): Path to the output directory.

    Returns:
        output_directory (pathlib.Path): Path to the created output directory.

    """
    if path is None:
        path = Path.cwd() / ".sid"

    output_directory = Path(path)

    if output_directory.exists() and not output_directory.is_dir():
        raise ValueError(f"{path} is a file instead of an directory.")
    elif output_directory.exists():
        shutil.rmtree(output_directory)

    output_directory.mkdir(parents=True, exist_ok=True)

    return output_directory


def _sort_contact_models(contact_models):
    """Sort the contact_models.

    First we have non recurrent, then recurrent contacts models. Within each group
    the models are sorted alphabetically.

    Args:
        contact_models (dict): see :ref:`contact_models`

    Returns:
        dict: sorted copy of contact_models.

    """
    sorted_ = sorted(
        name for name, mod in contact_models.items() if not mod["is_recurrent"]
    )
    sorted_ += sorted(
        name for name, mod in contact_models.items() if mod["is_recurrent"]
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


def _prepare_assortative_matching_indexers(states, assort_bys):
    """Create indexers and first stage probabilities for assortative matching.

    Args:
        states (pd.DataFrame): see :ref:`states`.
        assort_bys (dict): Keys are names of contact models, values are lists with the
            assort_by variables of the model.

    returns:
        indexers (dict): Dict of numba.Typed.List The i_th entry of the lists are the
            indices of the i_th group.

    """
    indexers = {}
    for model_name, assort_by in assort_bys.items():
        indexers[model_name] = create_group_indexer(states, assort_by)

    return indexers


def _prepare_assortative_matching_probabilities(
    states, assort_bys, params, contact_models
):
    """Create indexers and first stage probabilities for assortative matching.

    Args:
        states (pd.DataFrame): see :ref:`states`.
        assort_bys (dict): Keys are names of contact models, values are lists with the
            assort_by variables of the model.
        params (pd.DataFrame): see :ref:`params`.
        contact_models (dict): see :ref:`contact_models`.

    returns:
        first_probs (dict): dict of arrays of shape n_group, n_groups with probabilities
        for the first stage of sampling when matching contacts. probs[i, j] is the
        cumulative probability that an individual from group i meets someone from
        group j.

    """
    first_probs = {}
    for model_name, assort_by in assort_bys.items():
        if not contact_models[model_name]["is_recurrent"]:
            first_probs[model_name] = create_group_transition_probs(
                states, assort_by, params, model_name
            )
    return first_probs


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
    # Check if all assort_by columns are categoricals. This is important to save memory.
    assort_by_variables = list(
        set(it.chain.from_iterable(a_b for a_b in assort_bys.values()))
    )
    for assort_by in assort_by_variables:
        if states[assort_by].dtype.name != "category":
            states[assort_by] = states[assort_by].astype("category")

    # Sort index for deterministic shuffling and reset index because otherwise it will
    # be dropped while writing to parquet. Parquet stores an efficient range index
    # instead.
    states = states.sort_index().reset_index()

    for model_name, assort_by in assort_bys.items():
        states[f"group_codes_{model_name}"], _ = factorize_assortative_variables(
            states, assort_by
        )

    return states


def _dump_periodic_states(states, columns_to_keep, output_directory, date):
    states = states[columns_to_keep]
    states.to_parquet(output_directory / f"{date.date()}.parquet", engine="fastparquet")


def _return_dask_dataframe(output_directory, categoricals):
    """Process the simulation results.

    Args:
        output_directory (pathlib.Path): Path to output directory.
        categoricals (list): List of variable names which are categoricals.
    Returns:
        df (dask.dataframe): A dask DataFrame which contains the simulation results.

    """
    return dd.read_parquet(
        output_directory, categories=categoricals, engine="fastparquet"
    )


def _process_saved_columns(
    saved_columns, initial_state_columns, contact_models, optional_state_columns
):
    saved_columns = (
        SAVED_COLUMNS if saved_columns is None else {**SAVED_COLUMNS, **saved_columns}
    )

    all_columns = {
        "initial_states": initial_state_columns,
        "disease_states": [col for col in BOOLEAN_STATE_COLUMNS if "test" not in col],
        "testing_states": [col for col in BOOLEAN_STATE_COLUMNS if "test" in col]
        + ["pending_test_date"],
        "countdowns": list(COUNTDOWNS),
        "contacts": [f"n_contacts_{model}" for model in contact_models],
        "countdown_draws": [f"{cd}_draws" for cd in COUNTDOWNS],
        "group_codes": [f"group_codes_{model}" for model in contact_models],
    }

    keep = ["date"]
    for category in all_columns:
        keep += _combine_column_lists(saved_columns[category], all_columns[category])

    if isinstance(saved_columns["other"], list):
        keep += saved_columns["other"]

    all_columns["contacts"] = _combine_column_lists(
        optional_state_columns["contacts"], all_columns["contacts"]
    )

    if not optional_state_columns["reason_for_infection"]:
        keep = [k for k in keep if k != "reason_for_infection"]

    # drop duplicates
    keep = list(set(keep))

    return keep


def _combine_column_lists(user_entries, all_entries):
    if isinstance(user_entries, bool) and user_entries:
        res = all_entries
    elif isinstance(user_entries, list):
        res = [e for e in user_entries if e in all_entries]
    else:
        res = []
    res = list(res)
    return res
