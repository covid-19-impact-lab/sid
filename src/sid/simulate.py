import itertools as it
import shutil
import warnings
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from sid.config import BOOLEAN_STATE_COLUMNS
from sid.config import DTYPE_COUNTDOWNS
from sid.config import DTYPE_INFECTION_COUNTER
from sid.config import DTYPE_PERIOD
from sid.config import INDEX_NAMES
from sid.config import USELESS_COLUMNS
from sid.contacts import calculate_contacts
from sid.contacts import calculate_infections_by_contacts
from sid.contacts import create_group_indexer
from sid.countdowns import COUNTDOWNS
from sid.events import calculate_infections_by_events
from sid.matching_probabilities import create_group_transition_probs
from sid.parse_model import parse_duration
from sid.pathogenesis import draw_course_of_disease
from sid.shared import factorize_assortative_variables
from sid.testing_allocation import allocate_tests
from sid.testing_allocation import update_pending_tests
from sid.testing_demand import calculate_demand_for_tests
from sid.testing_processing import process_tests
from sid.update_states import add_debugging_information
from sid.update_states import update_states


def simulate(
    params,
    initial_states,
    initial_infections,
    contact_models,
    duration=None,
    events=None,
    contact_policies=None,
    testing_demand_models=None,
    testing_allocation_models=None,
    testing_processing_models=None,
    seed=None,
    path=None,
    debug=False,
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
        debug (bool): Indicates whether the debug modus is turned on which means that
            the simulated data contains more (technical) information to debug a model.

    Returns:
        simulation_results (dask.dataframe): The simulation results in form of a long
            :class:`dask.dataframe`. The DataFrame contains the states of each period
            (see :ref:`states`) and a column called newly_infected.

    """
    events = {} if events is None else events
    contact_policies = {} if contact_policies is None else contact_policies
    testing_demand_models = (
        {} if testing_demand_models is None else testing_demand_models
    )
    testing_allocation_models = (
        {} if testing_allocation_models is None else testing_allocation_models
    )
    testing_processing_models = (
        {} if testing_processing_models is None else testing_processing_models
    )
    seed = it.count(np.random.randint(0, 1_000_000)) if seed is None else it.count(seed)
    initial_states = initial_states.copy(deep=True)
    params = _prepare_params(params)

    output_directory = _create_output_directory(path)

    _check_inputs(
        params,
        initial_states,
        initial_infections,
        contact_models,
        contact_policies,
        testing_demand_models,
        testing_allocation_models,
        testing_processing_models,
    )

    contact_models = _sort_contact_models(contact_models)
    assort_bys = _process_assort_bys(contact_models)
    states = _process_initial_states(initial_states, assort_bys)
    duration = parse_duration(duration)

    states = draw_course_of_disease(states, params, seed)
    contact_policies = {
        key: _add_defaults_to_policy_dict(val, duration)
        for key, val in contact_policies.items()
    }
    states = update_states(states, initial_infections, initial_infections, params, seed)

    indexers, cum_probs = _prepare_assortative_matching(
        states, assort_bys, params, contact_models
    )

    for period, date in enumerate(duration["dates"]):
        states["date"] = date
        states["period"] = DTYPE_PERIOD(period)

        contacts = calculate_contacts(
            contact_models, contact_policies, states, params, date
        )

        (
            newly_infected_contacts,
            n_has_additionally_infected,
            newly_missed_contacts,
        ) = calculate_infections_by_contacts(
            states,
            contacts,
            params,
            indexers,
            cum_probs,
            seed,
        )
        newly_infected_events = calculate_infections_by_events(states, params, events)

        if testing_demand_models:
            demands_test, demands_test_reason = calculate_demand_for_tests(
                states, testing_demand_models, params, seed
            )
            allocated_tests = allocate_tests(
                states, testing_allocation_models, demands_test, params
            )

            states = update_pending_tests(states, allocated_tests)

            to_be_processed_tests = process_tests(
                states, testing_processing_models, params
            )
        else:
            demands_test = None
            allocated_tests = None
            to_be_processed_tests = None

        states = update_states(
            states,
            newly_infected_contacts,
            newly_infected_events,
            params,
            seed,
            n_has_additionally_infected,
            indexers,
            contacts,
            to_be_processed_tests,
        )

        if debug:
            states = add_debugging_information(
                states,
                newly_missed_contacts,
                demands_test,
                allocated_tests,
                to_be_processed_tests,
            )

        _dump_periodic_states(states, output_directory, date, debug)

    categoricals = {
        column: initial_states[column].cat.categories.shape[0]
        for column in initial_states.select_dtypes("category").columns
    }
    simulation_results = _return_dask_dataframe(output_directory, categoricals)

    return simulation_results


def _prepare_params(params):
    """Check the supplied params and set the index if not done."""
    if not isinstance(params, pd.DataFrame):
        raise ValueError("params must be a DataFrame.")

    params = params.copy()
    if not (
        isinstance(params.index, pd.MultiIndex) and params.index.names == INDEX_NAMES
    ):
        raise ValueError(
            "params must have the index levels 'category', 'subcategory' and 'name'."
        )

    if np.any(params.index.to_frame().isna()):
        raise ValueError(
            "No NaNs allowed in the params index. Repeat the previous index level "
            "instead."
        )

    if params.index.duplicated().any():
        raise ValueError("No duplicates in the params index allowed.")

    if params["value"].isna().any():
        raise ValueError("The 'value' column of params must not contain NaNs.")

    return params


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


def _check_inputs(
    params,
    initial_states,
    initial_infections,
    contact_models,
    contact_policies,
    testing_demand_models,
    testing_allocation_models,
    testing_processing_models,
):
    """Check the user inputs."""
    cd_names = sorted(COUNTDOWNS)
    gb = params.loc[cd_names].groupby(INDEX_NAMES[:2])
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

    for testing_model in [
        testing_demand_models,
        testing_allocation_models,
        testing_processing_models,
    ]:
        for name in testing_model:
            if not isinstance(testing_model[name], dict):
                raise ValueError(f"Each testing model must be a dictionary: {name}.")

            if "model" not in testing_model[name]:
                raise ValueError(
                    f"Each testing model must have a 'model' entry: {name}."
                )

    first_levels = params.index.get_level_values("category")
    assort_prob_matrices = [
        x for x in first_levels if x.startswith("assortative_matching_")
    ]
    for name in assort_prob_matrices:
        meeting_prob = params.loc[name]["value"].unstack()
        assert len(meeting_prob.index) == len(
            meeting_prob.columns
        ), f"assortative probability matrices must be square but isn't for {name}."
        assert (
            meeting_prob.index == meeting_prob.columns
        ).all(), (
            f"assortative probability matrices must be square but isn't for {name}."
        )
        assert (meeting_prob.sum(axis=1) > 0.9999).all() & (
            meeting_prob.sum(axis=1) < 1.00001
        ).all(), (
            f"the meeting probabilities of {name} do not add up to one in every row."
        )


def _prepare_assortative_matching(states, assort_bys, params, contact_models):
    """Create indexers and first stage probabilities for assortative matching.

    Args:
        states (pd.DataFrame): see :ref:`states`.
        assort_bys (dict): Keys are names of contact models, values are lists with the
            assort_by variables of the model.
        params (pd.DataFrame): see :ref:`params`.
        contact_models (dict): see :ref:`contact_models`.

    returns:
        indexers (dict): Dict of numba.Typed.List The i_th entry of the lists are the
            indices of the i_th group.
        first_probs (dict): dict of arrays of shape
            n_group, n_groups. probs[i, j] is the cumulative probability that an
            individual from group i meets someone from group j.

    """
    indexers = {}
    first_probs = {}
    for model_name, assort_by in assort_bys.items():
        indexers[model_name] = create_group_indexer(states, assort_by)
        if not contact_models[model_name]["is_recurrent"]:
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
    states["pending_test_period"] = np.nan

    for model_name, assort_by in assort_bys.items():
        states[f"group_codes_{model_name}"], _ = factorize_assortative_variables(
            states, assort_by
        )

    return states


def _dump_periodic_states(states, output_directory, date, debug):
    if not debug:
        group_codes = states.filter(like="group_codes_").columns.tolist()
        useless_columns = USELESS_COLUMNS + group_codes
        states = states.drop(columns=useless_columns)

    states.to_parquet(output_directory / f"{date.date()}.parquet")


def _return_dask_dataframe(output_directory, categoricals):
    """Process the simulation results.

    Args:
        output_directory (pathlib.Path): Path to output directory.
        categoricals (list): List of variable names which are categoricals.
    Returns:
        df (dask.dataframe): A dask DataFrame which contains the simulation results.

    """
    return dd.read_parquet(output_directory, categories=categoricals)
