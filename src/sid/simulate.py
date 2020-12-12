import functools
import itertools
import itertools as it
import logging
import shutil
import warnings
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import dask.dataframe as dd
import numba as nb
import numpy as np
import pandas as pd
from sid.config import BOOLEAN_STATE_COLUMNS
from sid.config import DTYPE_COUNTDOWNS
from sid.config import DTYPE_INFECTION_COUNTER
from sid.config import OPTIONAL_STATE_COLUMNS
from sid.config import SAVED_COLUMNS
from sid.contacts import calculate_contacts
from sid.contacts import calculate_infections_by_contacts
from sid.contacts import create_group_indexer
from sid.countdowns import COUNTDOWNS
from sid.events import calculate_infections_by_events
from sid.initial_conditions import (
    sample_initial_distribution_of_infections_and_immunity,
)
from sid.matching_probabilities import create_group_transition_probs
from sid.parse_model import parse_duration
from sid.parse_model import parse_initial_conditions
from sid.parse_model import parse_share_known_cases
from sid.pathogenesis import draw_course_of_disease
from sid.shared import factorize_assortative_variables
from sid.testing_allocation import allocate_tests
from sid.testing_allocation import update_pending_tests
from sid.testing_demand import calculate_demand_for_tests
from sid.testing_processing import process_tests
from sid.time import timestamp_to_sid_period
from sid.update_states import update_states
from sid.validation import validate_initial_conditions
from sid.validation import validate_initial_states
from sid.validation import validate_models
from sid.validation import validate_params
from sid.validation import validate_prepared_initial_states
from tqdm import tqdm


logger = logging.getLogger("sid")


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
    seed: Optional[int] = None,
    path: Union[str, Path, None] = None,
    saved_columns: Optional[Dict[str, Union[bool, str, List[str]]]] = None,
    optional_state_columns=None,
    initial_conditions: Optional[Dict[str, Any]] = None,
    share_known_cases: Optional[float, pd.Series] = None,
):
    """Get a function that simulates the spread of an infectious disease.

    The resulting function only depends on parameters. The computational time it takes
    to process the user input is only incurred once in :func:`get_simulate_func` and not
    when the resulting function is called.

    Args:
        params (pandas.DataFrame): DataFrame with parameters that influence the number
            of contacts, contagiousness and dangerousness of the disease, ... .
        initial_states (pandas.DataFrame): See :ref:`states`. Cannot contain the column
            "date" because it is used internally.
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
        seed (Optional[int]): The seed is used as the starting point for two seed
            sequences where one is used to set up the simulation function and the other
            seed sequence is used within the simulation and reset every parameter
            evaluation. If you pass ``None`` as a seed, an internal seed is sampled to
            set up the simulation function. The seed for the simulation is sampled at
            the beginning of the simulation function and can be influenced by setting
            :class:`numpy.random.seed` right before the call.
        path (Union[str, pathlib.Path, None]): Path to the directory where the simulated
            data is stored.
        saved_columns (Option[Dict[str, Union[bool, str, List[str]]]]): Dictionary with
            categories of state columns. The corresponding values can be True, False or
            Lists with columns that should be saved. Typically, during estimation you
            only want to save exactly what you need to calculate moments to make the
            simulation and calculation of moments faster. The categories are
            "initial_states", "disease_states", "testing_states", "countdowns",
            "contacts", "countdown_draws", "group_codes" and "other".
        optional_state_columns (dict): Dictionary with categories of state columns
            that can additionally be added to the states DataFrame, either for use in
            contact models and policies or to be saved. Most types of columns are added
            by default, but some of them are costly to add and thus only added when
            needed. Columns that are not in the state but specified in ``saved_columns``
            will not be saved. The sole category is currently "contacts".
        initial_conditions (Optional[Dict[str, Any]]): The initial conditions allow you
            to govern the distribution of infections and immunity and the heterogeneity
            of courses of disease at the start of the simulation. Use ``None`` to assume
            no heterogeneous courses of diseases and 1% infections. Otherwise,
            ``initial_conditions`` is a dictionary containing the following entries:

            - ``assort_by`` (Optional[Union[str, List[str]]]): The relative infections
              is preserved between the groups formed by ``assort_by`` variables. By
              default, no group is formed and infections spread across the whole
              population.
            - ``burn_in_periods`` (int): The number of periods over which infections are
              distributed and can progress. The default is one period.
            - ``growth_rate`` (float): The growth rate specifies the increase of
              infections from one burn-in period to the next. For example, two indicates
              doubling case numbers every period. The value must be greater than or
              equal to one. Default is one which is no distribution over time.
            - ``initial_immunity`` (Union[int, float, pandas.Series]): The n_people who
              are immune in the beginning can be specified as an integer for the number,
              a float between 0 and 1 for the share, and a :class:`pandas.Series` with
              the same index as states. Note that infected individuals are also immune.
              For a 10% pre-existing immunity with 2% currently infected people, set the
              key to 0.12. By default, only infected individuals indicated by the
              initial infections are immune.
            - ``initial_infections`` (Union[int, float, pandas.Series,
              pandas.DataFrame]): The initial infections can be given as an integer
              which is the number of randomly infected individuals, as a float for the
              share or as a :class:`pandas.Series` which indicates whether an
              individuals is infected. If initial infections are a
              :class:`pandas.DataFrame`, then, the index is the same as ``states``,
              columns are dates or periods which can be sorted, and values are infected
              individuals on that date. This step will skip upscaling and distributing
              infections over days and directly jump to the evolution of states. By
              default, 1% of individuals is infected.
            - ``known_cases_multiplier`` (int): The factor can be used to scale up the
              initial infections while keeping shares between ``assort_by`` variables
              constant. This is helpful if official numbers are underreporting the
              number of cases.
        share_known_cases (Optional[float, pd.Series]): Share of known cases to all
            cases. The argument is a float or a series with :class:`pd.DatetimeIndex`
            which covers the whole simulation period and yields the ratio of known
            infections to all infections.

            This feature can be used instead of testing models which are hard to
            calibrate to data.

    Returns:
        callable: Simulates dataset based on parameters.

    """
    startup_seed, simulation_seed = _generate_seeds(seed)

    events = {} if events is None else events
    contact_policies = {} if contact_policies is None else contact_policies
    if testing_demand_models is None:
        testing_demand_models = {}
    if testing_allocation_models is None:
        testing_allocation_models = {}
    if testing_processing_models is None:
        testing_processing_models = {}

    initial_states = initial_states.copy(deep=True)
    params = params.copy(deep=True)

    validate_params(params)
    validate_models(
        contact_models,
        contact_policies,
        testing_demand_models,
        testing_allocation_models,
        testing_processing_models,
    )

    optional_state_columns = _process_optional_state_columns(optional_state_columns)
    user_state_columns = initial_states.columns

    path = _create_output_directory(path)
    contact_models = _sort_contact_models(contact_models)
    assort_bys = _process_assort_bys(contact_models)
    duration = parse_duration(duration)
    contact_policies = {
        key: _add_defaults_to_policy_dict(val, duration)
        for key, val in contact_policies.items()
    }

    initial_conditions = parse_initial_conditions(initial_conditions, duration["start"])
    validate_initial_conditions(initial_conditions)
    share_known_cases = parse_share_known_cases(
        share_known_cases, duration, initial_conditions["burn_in_periods"]
    )

    if _are_states_prepared(initial_states):
        if initial_conditions is not None:
            raise ValueError(
                "You passed both, prepared states to resume a simulation and initial "
                "conditions, which is not possible. Either resume a simulation or "
                "start a new one."
            )
        validate_prepared_initial_states(initial_states, duration)
    else:
        validate_initial_states(initial_states)
        initial_states = _process_initial_states(initial_states, assort_bys)
        initial_states = draw_course_of_disease(
            initial_states, params, next(startup_seed)
        )
        initial_states = sample_initial_distribution_of_infections_and_immunity(
            initial_states, params, initial_conditions, share_known_cases, startup_seed
        )

    indexers = _prepare_assortative_matching_indexers(initial_states, assort_bys)

    cols_to_keep = _process_saved_columns(
        saved_columns, user_state_columns, contact_models, optional_state_columns
    )

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
        seed=simulation_seed,
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
        initial_states (pandas.DataFrame): See :ref:`states`. Cannot contain the column
            "date" because it is used internally.
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
        seed (int, optional): The seed is used as the starting point for two seed
            sequences where one is used to set up the simulation function and the other
            seed sequence is used within the simulation and reset every parameter
            evaluation. If you pass ``None`` as a seed, an internal seed is sampled to
            set up the simulation function. The seed for the simulation is sampled at
            the beginning of the simulation function and can be influenced by setting
            :class:`numpy.random.seed` right before the call.
        path (pathlib.Path): Path to the directory where the simulated data is stored.
        columns_to_keep (list): Columns of states that will be saved in each period.
        optional_state_columns (dict): Dictionary with categories of state columns
            that can additionally be added to the states dataframe, either for use in
            contact models and policies or to be saved. Most types of columns are added
            by default, but some of them are costly to add and thus only added when
            needed. Columns that are not in the state but specified in ``saved_columns``
            will not be saved. The sole category is currently "contacts".

    Returns:
        result (dict): The simulation result which includes the following keys:

        - ``time_series`` (:class:`dask.dataframe`): The DataFrame contains the states
          of each period (see :ref:`states`).
        - ``last_states`` (:class:`dask.dataframe`): The states of the last simulated
          period to resume the simulation.

    """
    seed = np.random.randint(0, 1_000_000) if seed is None else seed
    seed = itertools.count(seed)

    cum_probs = _prepare_assortative_matching_probabilities(
        initial_states, assort_bys, params, contact_models
    )

    code_to_contact_model = dict(enumerate(contact_models))
    states = initial_states

    if states.columns.isin(["date", "period"]).any():
        logger.info("Resume the simulation...")
    else:
        logger.info("Start the simulation...")

    pbar = tqdm(duration["dates"])
    for date in pbar:
        pbar.set_description(f"{date.date()}")

        states["date"] = date
        states["period"] = timestamp_to_sid_period(date)

        contacts = calculate_contacts(
            contact_models=contact_models,
            contact_policies=contact_policies,
            states=states,
            params=params,
            date=date,
            seed=seed,
        )

        (
            newly_infected_contacts,
            n_has_additionally_infected,
            newly_missed_contacts,
            channel_infected_by_contact,
        ) = calculate_infections_by_contacts(
            states=states,
            contacts=contacts,
            params=params,
            indexers=indexers,
            group_cdfs=cum_probs,
            code_to_contact_model=code_to_contact_model,
            seed=seed,
        )
        (
            newly_infected_events,
            channel_infected_by_event,
        ) = calculate_infections_by_events(states, params, events)

        if testing_demand_models:
            demands_test, channel_demands_test = calculate_demand_for_tests(
                states,
                testing_demand_models,
                params,
                date,
                optional_state_columns,
                seed,
            )
            allocated_tests = allocate_tests(
                states, testing_allocation_models, demands_test, params, date
            )

            states = update_pending_tests(states, allocated_tests)

            to_be_processed_tests = process_tests(
                states, testing_processing_models, params, date
            )
        else:
            demands_test = None
            channel_demands_test = None
            allocated_tests = None
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
            channel_infected_by_contact=channel_infected_by_contact,
            channel_infected_by_event=channel_infected_by_event,
            channel_demands_test=channel_demands_test,
        )

        _dump_periodic_states(states, columns_to_keep, path, date)

    results = _prepare_simulation_result(path, columns_to_keep, states)

    return results


def _generate_seeds(seed: Optional[int]):
    """Generate seeds for startup and simulation.

    We use the user provided seed or a random seed to generate two other seeds. The
    first seed will be turned to a seed sequence and used to control randomness during
    the preparation of the simulate function. The second seed is for the randomness in
    the simulation, but stays an integer so that the seed sequence can be rebuild every
    iteration.

    If the seed is ``None``, only the start-up seed is sampled and the seed for
    simulation is set to ``None``. This seed will be sampled in :func:`_simulate` and
    can be influenced by setting ``np.random.seed(seed) right before the call.

    Args:
        seed (Optional[int]): The seed provided by the user.

    Returns:
        startup_seed (itertools.count): The seed sequence for the startup.
        simulation_seed (int): The starting point for the seed sequence in the
            simulation.

    """
    internal_seed = np.random.randint(0, 1_000_000) if seed is None else seed

    np.random.seed(internal_seed)

    startup_seed = itertools.count(np.random.randint(0, 10_000))
    simulation_seed = (
        np.random.randint(100_000, 1_000_000) if seed is not None else None
    )

    return startup_seed, simulation_seed


def _create_output_directory(path: Union[str, Path, None]) -> Path:
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
    output_directory.joinpath("last_states").mkdir(parents=True, exist_ok=True)
    output_directory.joinpath("time_series").mkdir(parents=True, exist_ok=True)

    return output_directory


def _sort_contact_models(contact_models: Dict[str, Any]) -> Dict[str, Any]:
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


def _process_assort_bys(contact_models: Dict[str, Any]) -> Dict[str, List[str]]:
    """Set default values for assort_by variables and extract them into a dict.

    Args:
        contact_models (Dict[str, Any]): see :ref:`contact_models`

    Returns:
        assort_bys (Dict[str, List[str]]): Keys are names of contact models, values are
            lists with the assort_by variables of the model.

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
                f"'assort_by' for '{model_name}' must one of False, str, or list."
            )

        assort_bys[model_name] = assort_by

    return assort_bys


def _prepare_assortative_matching_indexers(
    states: pd.DataFrame, assort_bys: Dict[str, List[str]]
) -> Dict[str, nb.typed.List]:
    """Create indexers and first stage probabilities for assortative matching.

    Args:
        states (pd.DataFrame): see :ref:`states`.
        assort_bys (Dict[str, List[str]]): Keys are names of contact models, values are
            lists with the assort_by variables of the model.

    returns:
        indexers (Dict[str, numba.typed.List]): The i_th entry of the lists are the
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
    if np.any(states.isna()):
        raise ValueError("'initial_states' are not allowed to contain NaNs.")

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

    for col in BOOLEAN_STATE_COLUMNS:
        if col not in states.columns:
            states[col] = False

    for col in COUNTDOWNS:
        if col not in states.columns:
            states[col] = -1
        states[col] = states[col].astype(DTYPE_COUNTDOWNS)

    states["n_has_infected"] = DTYPE_INFECTION_COUNTER(0)
    states["pending_test_date"] = pd.NaT

    for model_name, assort_by in assort_bys.items():
        states[f"group_codes_{model_name}"], _ = factorize_assortative_variables(
            states, assort_by
        )

    return states


def _dump_periodic_states(states, columns_to_keep, output_directory, date):
    states = states[columns_to_keep]
    states.to_parquet(
        output_directory / "time_series" / f"{date.date()}.parquet",
        engine="fastparquet",
    )


def _prepare_simulation_result(output_directory, columns_to_keep, last_states):
    """Process the simulation results.

    Args:
        output_directory (pathlib.Path): Path to output directory.
        columns_to_keep (list): List of variables which should be kept.
        last_states (pandas.DataFrame): States of the last period.

    Returns:
        result (dict): The simulation result which includes the following keys:

            - ``time_series`` (dask.dataframe): The DataFrame contains the states of
              each period (see :ref:`states`).
            - ``last_states`` (dask.dataframe): The states of the last simulated period
              to resume the simulation.

    """
    categoricals = {
        column: last_states[column].cat.categories.shape[0]
        for column in last_states.select_dtypes("category").columns
    }

    last_states.to_parquet(output_directory / "last_states" / "last_states.parquet")
    last_states = dd.read_parquet(
        output_directory / "last_states" / "last_states.parquet",
        categories=categoricals,
        engine="fastparquet",
    )

    reduced_categoricals = {
        k: v for k, v in categoricals.items() if k in columns_to_keep
    }

    time_series = dd.read_parquet(
        output_directory / "time_series",
        categories=reduced_categoricals,
        engine="fastparquet",
    )

    return {"time_series": time_series, "last_states": last_states}


def _process_saved_columns(
    saved_columns, initial_state_columns, contact_models, optional_state_columns
):
    saved_columns = (
        SAVED_COLUMNS if saved_columns is None else {**SAVED_COLUMNS, **saved_columns}
    )

    all_columns = {
        "time": ["date", "period"],
        "initial_states": initial_state_columns,
        "disease_states": [col for col in BOOLEAN_STATE_COLUMNS if "test" not in col],
        "testing_states": (
            [col for col in BOOLEAN_STATE_COLUMNS if "test" in col]
            + ["pending_test_date"]
        ),
        "countdowns": list(COUNTDOWNS),
        "contacts": [f"n_contacts_{model}" for model in contact_models],
        "countdown_draws": [f"{cd}_draws" for cd in COUNTDOWNS],
        "group_codes": [f"group_codes_{model}" for model in contact_models],
        "channels": [
            "channel_infected_by_contact",
            "channel_infected_by_event",
            "channel_demands_test",
        ],
    }

    keep = []
    for category in all_columns:
        keep += _combine_column_lists(saved_columns[category], all_columns[category])

    if isinstance(saved_columns["other"], list):
        keep += saved_columns["other"]

    keep += _combine_column_lists(
        optional_state_columns["contacts"], all_columns["contacts"]
    )

    keep += _combine_column_lists(
        optional_state_columns["channels"], all_columns["channels"]
    )

    # drop duplicates
    keep = list(set(keep))

    return keep


def _combine_column_lists(user_entries, all_entries):
    if isinstance(user_entries, bool) and user_entries:
        res = all_entries
    elif isinstance(user_entries, list):
        res = [e for e in user_entries if e in all_entries]
    elif isinstance(user_entries, str):
        res = [user_entries] if user_entries in all_entries else []
    else:
        res = []
    res = list(res)
    return res


def _process_optional_state_columns(opt_state_cols):
    res = (
        OPTIONAL_STATE_COLUMNS
        if opt_state_cols is None
        else {**OPTIONAL_STATE_COLUMNS, **opt_state_cols}
    )
    return res


def _are_states_prepared(states):
    """Are states prepared.

    If the states include information on the period or date, we assume that the states
    are prepared.

    """
    return states.columns.isin(["date", "period"]).any()
