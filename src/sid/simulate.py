import functools
import itertools
import itertools as it
import logging
import shutil
import warnings
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import dask.dataframe as dd
import numba as nb
import numpy as np
import pandas as pd
from sid.config import BOOLEAN_STATE_COLUMNS
from sid.config import DTYPE_COUNTDOWNS
from sid.config import DTYPE_GROUP_CODE
from sid.config import DTYPE_INFECTION_COUNTER
from sid.config import SAVED_COLUMNS
from sid.contacts import calculate_contacts
from sid.contacts import calculate_infections_by_contacts
from sid.contacts import create_group_indexer
from sid.contacts import post_process_contacts
from sid.countdowns import COUNTDOWNS
from sid.events import calculate_infections_by_events
from sid.initial_conditions import (
    sample_initial_distribution_of_infections_and_immunity,
)
from sid.matching_probabilities import create_cumulative_group_transition_probabilities
from sid.parse_model import parse_duration
from sid.parse_model import parse_initial_conditions
from sid.parse_model import parse_virus_strains
from sid.pathogenesis import draw_course_of_disease
from sid.policies import apply_contact_policies
from sid.rapid_tests import apply_reactions_to_rapid_tests
from sid.rapid_tests import perform_rapid_tests
from sid.seasonality import prepare_seasonality_factor
from sid.shared import factorize_assortative_variables
from sid.shared import separate_contact_model_names
from sid.susceptibility import prepare_susceptibility_factor
from sid.testing import perform_testing
from sid.time import timestamp_to_sid_period
from sid.update_states import update_states
from sid.vaccination import vaccinate_individuals
from sid.validation import validate_contact_models
from sid.validation import validate_contact_policies
from sid.validation import validate_initial_states
from sid.validation import validate_params
from sid.validation import validate_prepared_initial_states
from sid.validation import validate_testing_models
from sid.validation import validate_vaccination_models
from tqdm import tqdm


logger = logging.getLogger("sid")


def get_simulate_func(
    params: pd.DataFrame,
    initial_states: pd.DataFrame,
    contact_models: Dict[str, Any],
    duration: Optional[Dict[str, Any]] = None,
    events: Optional[Dict[str, Any]] = None,
    contact_policies: Optional[Dict[str, Any]] = None,
    testing_demand_models: Optional[Dict[str, Any]] = None,
    testing_allocation_models: Optional[Dict[str, Any]] = None,
    testing_processing_models: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    path: Union[str, Path, None] = None,
    saved_columns: Optional[Dict[str, Union[bool, str, List[str]]]] = None,
    initial_conditions: Optional[Dict[str, Any]] = None,
    susceptibility_factor_model: Optional[Callable] = None,
    virus_strains: Optional[List[str]] = None,
    vaccination_models: Optional[Callable] = None,
    rapid_test_models: Optional[Dict[str, Dict[str, Any]]] = None,
    rapid_test_reaction_models: Optional[Dict[str, Dict[str, Any]]] = None,
    seasonality_factor_model: Optional[Callable] = None,
    derived_state_variables: Optional[Dict[str, str]] = None,
):
    """Get a function that simulates the spread of an infectious disease.

    The resulting function only depends on parameters. The computational time it takes
    to process the user input is only incurred once in :func:`get_simulate_func` and not
    when the resulting function is called.

    Args:
        params (pandas.DataFrame): ``params`` is a DataFrame with a three-level index
            which contains parameters for various aspects of the model. For example,
            infection probabilities of contact models, multiplier effects of policies,
            determinants of the course of the disease. More information can be found in
            :ref:`params`.
        initial_states (pandas.DataFrame): The initial states are a DataFrame which
            contains individuals and their characteristics. More information can be
            found in :ref:`states`.
        contact_models (Dict[str, Any]): A dictionary of dictionaries where each
            dictionary describes a channel by which contacts can be formed. More
            information can be found in :ref:`contact_models`.
        duration (Optional[Dict[str, Any]]): A dictionary which contains keys and values
            suited to be passed to :func:`pandas.date_range`. Only the first three
            arguments, ``"start"``, ``"end"``, and ``"periods"``, are allowed.
        events (Optional[Dict[str, Any]]): Dictionary of events which cause infections.
        contact_policies (Optional[Dict[str, Any]]): Dict of dicts with contact. See
            :ref:`policies`.
        testing_demand_models (Optional[Dict[str, Any]]): Dict of dicts with demand
            models for tests. See :ref:`testing_demand_models` for more information.
        testing_allocation_models (Optional[Dict[str, Any]]): Dict of dicts with
            allocation models for tests. See :ref:`testing_allocation_models` for more
            information.
        testing_processing_models (Optional[Dict[str, Any]]): Dict of dicts with
            processing models for tests. See :ref:`testing_processing_models` for more
            information.
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
            - ``virus_shares`` (Union[dict, pandas.Series]): A mapping between the names
              of the virus strains and their share among newly infected individuals in
              each burn-in period.
        susceptibility_factor_model (Optional[Callable]): A function which
            takes the states and parameters and returns an infection probability
            multiplier for each individual.
        virus_strains (Optional[List[str]]): A list of names indicating the different
            virus strains used in the model. Their different contagiousness factors are
            looked up in the params DataFrame. By default, only one virus strain is
            used.
        vaccination_models (Optional[Dict[str, Dict[str, Any]): A dictionary of models
            which allow to vaccinate individuals. The ``"model"`` key holds a function
            with arguments ``states``, ``params``, and a ``seed`` which returns boolean
            indicators for individuals who received a vaccination.
        rapid_test_models (Optional[Dict[str, Dict[str, Any]]]: A dictionary of
            dictionaries containing models for rapid tests. Each model for rapid tests
            can have a ``"start"`` and ``"end"`` date. It must have a function under
            ``"model"`` which accepts ``states``, ``params``, ``receives_rapid_test``,
            ``"contacts"``  and ``seed`` and returns a boolean series indicating
            individuals who  received a rapid test. The difference to other test models
            is that rapid tests are performed after planned contacts are calculated
            (i.e. contact models and policies are evaluated) but before they actually
            take place. This allows people to use more rapid tests on days  with many
            planned contacts and to react to the test outcome in
            ``rapid_test_reaction_models``.
        rapid_test_reaction_models (Optional[Dict[str, Dict[str, Any]]]): A dictionary
            holding rapid tests reaction models which allow to change calculated
            contacts based on the results of rapid tests. Each model can have a
            ``"start"`` and ``"end"`` date. It must have a function under ``"model"``
            which accepts ``states``, ``params``,  ``"contacts"``  and ``seed`` and
            returns a modified copy of contacts.
        seasonality_factor_model (Optional[Callable]): A model which takes in and
            ``params`` and ``dates`` signaling the whole duration of the simulation and
            returns a factor for each day which scales all infection probabilities.
        derived_state_variables (Optional[Dict[str, str]]): A dictionary that maps
            names of state variables to pandas evaluation strings that generate derived
            state variables, i.e. state variables that can be calculated from the
            existing state variables.

    Returns:
        Callable: Simulates dataset based on parameters.

    """
    startup_seed, simulation_seed = _generate_seeds(seed)

    events = {} if events is None else events
    contact_policies = {} if contact_policies is None else contact_policies
    if (
        testing_demand_models is None
        or testing_allocation_models is None
        or testing_processing_models is None
    ):
        testing_demand_models = {}
        testing_allocation_models = {}
        testing_processing_models = {}

    if rapid_test_reaction_models is None or rapid_test_models is None:
        rapid_test_reaction_models = {}
    if rapid_test_models is None:
        rapid_test_models = {}
    if vaccination_models is None:
        vaccination_models = {}
    if derived_state_variables is None:
        derived_state_variables = {}

    initial_states = initial_states.copy(deep=True)
    params = params.copy(deep=True)

    validate_params(params)
    validate_contact_models(contact_models)
    validate_contact_policies(contact_policies, contact_models)
    validate_testing_models(
        testing_demand_models, testing_allocation_models, testing_processing_models
    )
    validate_vaccination_models(vaccination_models)

    user_state_columns = initial_states.columns

    path = _create_output_directory(path)
    contact_models = _sort_contact_models(contact_models)
    assort_bys = _process_assort_bys(contact_models)
    duration = parse_duration(duration)
    virus_strains = parse_virus_strains(virus_strains, params)

    contact_policies = _add_default_duration_to_models(contact_policies, duration)

    initial_conditions = parse_initial_conditions(
        initial_conditions, duration["start"], virus_strains
    )

    # Testing and vaccination models are used in the initial conditions and should be
    # activated during the burn-in phase if the starting date is not defined.
    default_duration_w_burn_in = {
        "start": initial_conditions["burn_in_periods"][0],
        "end": duration["end"],
    }
    testing_demand_models = _add_default_duration_to_models(
        testing_demand_models, default_duration_w_burn_in
    )
    testing_allocation_models = _add_default_duration_to_models(
        testing_allocation_models, default_duration_w_burn_in
    )
    testing_processing_models = _add_default_duration_to_models(
        testing_processing_models, default_duration_w_burn_in
    )
    vaccination_models = _add_default_duration_to_models(
        vaccination_models, default_duration_w_burn_in
    )

    rapid_test_models = _add_default_duration_to_models(rapid_test_models, duration)
    rapid_test_reaction_models = _add_default_duration_to_models(
        rapid_test_reaction_models, duration
    )

    if _are_states_prepared(initial_states):
        validate_prepared_initial_states(initial_states, duration)
    else:
        validate_initial_states(initial_states)
        initial_states = _process_initial_states(
            initial_states, assort_bys, virus_strains
        )
        initial_states = draw_course_of_disease(
            initial_states, params, next(startup_seed)
        )
        initial_states = sample_initial_distribution_of_infections_and_immunity(
            states=initial_states,
            params=params,
            initial_conditions=initial_conditions,
            testing_demand_models=testing_demand_models,
            testing_allocation_models=testing_allocation_models,
            testing_processing_models=testing_processing_models,
            virus_strains=virus_strains,
            vaccination_models=vaccination_models,
            seed=startup_seed,
            derived_state_variables=derived_state_variables,
        )

    initial_states, group_codes_info = _create_group_codes_and_info(
        initial_states, assort_bys, contact_models
    )

    indexers = _prepare_assortative_matching_indexers(
        initial_states, contact_models, group_codes_info
    )

    cols_to_keep = _process_saved_columns(
        saved_columns, user_state_columns, group_codes_info, contact_models
    )

    sim_func = functools.partial(
        _simulate,
        initial_states=initial_states,
        assort_bys=assort_bys,
        contact_models=contact_models,
        group_codes_info=group_codes_info,
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
        susceptibility_factor_model=susceptibility_factor_model,
        virus_strains=virus_strains,
        vaccination_models=vaccination_models,
        rapid_test_models=rapid_test_models,
        rapid_test_reaction_models=rapid_test_reaction_models,
        seasonality_factor_model=seasonality_factor_model,
        derived_state_variables=derived_state_variables,
    )
    return sim_func


def _simulate(
    params,
    initial_states,
    assort_bys,
    contact_models,
    group_codes_info,
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
    susceptibility_factor_model,
    virus_strains,
    vaccination_models,
    rapid_test_models,
    rapid_test_reaction_models,
    seasonality_factor_model,
    derived_state_variables,
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
        susceptibility_factor_model (Callable): A function which takes the
            states and parameters and returns an infection probability multiplier for
            each individual.
        virus_strains (Dict[str, Any]): A dictionary with the keys ``"names"`` and
            ``"factors"`` holding the different contagiousness factors of multiple
            viruses.
        vaccination_models (Optional[Dict[str, Dict[str, Any]): A dictionary of models
            which allow to vaccinate individuals. The ``"model"`` key holds a function
            with arguments ``states``, ``params``, and a ``seed`` which returns boolean
            indicators for individuals who received a vaccination.
        rapid_test_models (Optional[Dict[str, Dict[str, Any]]]: A dictionary of
            dictionaries containing models for rapid tests. Each model for rapid tests
            can have a ``"start"`` and ``"end"`` date. It must have a function under
            ``"model"`` which accepts ``states``, ``params``, ``receives_rapid_test``
            and ``seed`` and returns a boolean series indicating individuals who
            received a rapid test. The difference to other test models is that rapid
            tests are performed before contacts are calculated to allow that people can
            use rapid tests before they meet other people and so that normal tests can
            re-test individuals with positive rapid tests.
        rapid_test_reaction_models (Optional[Dict[str, Dict[str, Any]]]): A dictionary
            holding rapid tests reaction models which allow to change calculated
            contacts based on the results of rapid tests.
        seasonality_factor_model (Optional[Callable]): A model which takes in and
            ``params`` and ``dates`` signaling the whole duration of the simulation and
            returns a factor for each day which scales all infection probabilities.
        derived_state_variables (Dict[str, str]): A dictionary that maps
            names of state variables to pandas evaluation strings that generate derived
            state variables, i.e. state variables that can be calculated from the
            existing state variables.

    Returns:
        result (Dict[str, dask.dataframe]): The simulation result which includes the
            following keys:

            - **time_series** (:class:`dask.dataframe`): The DataFrame contains the
              states of each period (see :ref:`states`).
            - **last_states** (:class:`dask.dataframe`): The states of the last
              simulated period to resume the simulation.

    """
    seed = np.random.randint(0, 1_000_000) if seed is None else seed
    seed = itertools.count(seed)

    assortative_matching_cum_probs = (
        _prepare_assortative_matching_cumulative_probabilities(
            initial_states, assort_bys, params, contact_models, group_codes_info
        )
    )

    susceptibility_factor = prepare_susceptibility_factor(
        susceptibility_factor_model=susceptibility_factor_model,
        initial_states=initial_states,
        params=params,
        seed=seed,
    )

    seasonality_factor = prepare_seasonality_factor(
        seasonality_factor_model=seasonality_factor_model,
        params=params,
        dates=duration["dates"],
        seed=seed,
    )

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
            states=states,
            params=params,
            seed=seed,
        )

        contacts = apply_contact_policies(
            contact_policies=contact_policies,
            contacts=contacts,
            states=states,
            date=date,
            seed=seed,
        )

        states = perform_rapid_tests(
            date=date,
            states=states,
            params=params,
            rapid_test_models=rapid_test_models,
            contacts=contacts,
            seed=seed,
        )

        contacts = apply_reactions_to_rapid_tests(
            date=date,
            states=states,
            params=params,
            rapid_test_reaction_models=rapid_test_reaction_models,
            contacts=contacts,
            seed=seed,
        )

        recurrent_contacts, random_contacts = post_process_contacts(
            contacts=contacts,
            states=states,
            contact_models=contact_models,
        )

        recurrent_contacts_np = (
            None
            if recurrent_contacts is None
            else recurrent_contacts.to_numpy(copy=True)
        )
        random_contacts_np = (
            None if random_contacts is None else random_contacts.to_numpy(copy=True)
        )

        (
            newly_infected_contacts,
            n_has_additionally_infected,
            newly_missed_contacts,
            channel_infected_by_contact,
        ) = calculate_infections_by_contacts(
            states=states,
            recurrent_contacts=recurrent_contacts_np,
            random_contacts=random_contacts_np,
            params=params,
            indexers=indexers,
            assortative_matching_cum_probs=assortative_matching_cum_probs,
            contact_models=contact_models,
            group_codes_info=group_codes_info,
            susceptibility_factor=susceptibility_factor,
            virus_strains=virus_strains,
            seasonality_factor=seasonality_factor[date],
            seed=seed,
        )
        (
            newly_infected_events,
            channel_infected_by_event,
        ) = calculate_infections_by_events(states, params, events, virus_strains, seed)

        states, channel_demands_test, to_be_processed_tests = perform_testing(
            date=date,
            states=states,
            params=params,
            testing_demand_models=testing_demand_models,
            testing_allocation_models=testing_allocation_models,
            testing_processing_models=testing_processing_models,
            seed=seed,
            columns_to_keep=columns_to_keep,
        )

        newly_vaccinated = vaccinate_individuals(
            date, vaccination_models, states, params, seed
        )

        states = update_states(
            states=states,
            newly_infected_contacts=newly_infected_contacts,
            newly_infected_events=newly_infected_events,
            params=params,
            to_be_processed_tests=to_be_processed_tests,
            virus_strains=virus_strains,
            newly_vaccinated=newly_vaccinated,
            seed=seed,
            derived_state_variables=derived_state_variables,
        )

        states = _add_additional_information_to_states(
            states=states,
            columns_to_keep=columns_to_keep,
            n_has_additionally_infected=n_has_additionally_infected,
            random_contacts=random_contacts,
            recurrent_contacts=recurrent_contacts,
            channel_infected_by_contact=channel_infected_by_contact,
            channel_infected_by_event=channel_infected_by_event,
            channel_demands_test=channel_demands_test,
            susceptibility_factor=susceptibility_factor,
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
    can be influenced by setting ``np.random.seed(seed)`` right before the call.

    Args:
        seed (Optional[int]): The seed provided by the user.

    Returns:
        out (tuple): A tuple containing

        - **startup_seed** (:class:`itertools.count`): The seed sequence for the
          startup.
        - **simulation_seed** (:class:`int`): The starting point for the seed sequence
          in the simulation.

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
        path (Union[str, Path, None]): Path to the output directory.

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

    for directory in [
        output_directory,
        output_directory / "last_states",
        output_directory / "time_series",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    return output_directory


def _sort_contact_models(contact_models: Dict[str, Any]) -> Dict[str, Any]:
    """Sort the contact_models.

    First we have non recurrent, then recurrent contacts models. Within each group
    the models are sorted alphabetically.

    Args:
        contact_models (Dict[str, Any]): See :ref:`contact_models`

    Returns:
        Dict[str, Any]: Sorted copy of contact_models.

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


def _create_group_codes_names(
    contact_models: Dict[str, Any], assort_bys: Dict[str, List[str]]
) -> Dict[str, str]:
    """Create a name for each contact models group codes.

    The group codes are either found in the initial states or are a factorization of one
    or multiple variables in the initial states.

    ``"is_factorized"`` can be set in contact models to indicate that the assortative
    variable is already factorized which saves memory.

    """
    group_codes_names = {}
    for name, model in contact_models.items():
        is_factorized = model.get("is_factorized", False)
        n_assort_bys = len(assort_bys[name])
        if is_factorized and n_assort_bys != 1:
            raise ValueError(
                f"'is_factorized' is 'True' for contact model {name}, but there is not "
                f"one assortative variable, but {n_assort_bys}."
            )
        elif is_factorized:
            group_codes_names[name] = assort_bys[name][0]
        else:
            group_codes_names[name] = f"group_codes_{name}"

    return group_codes_names


def _prepare_assortative_matching_indexers(
    states: pd.DataFrame,
    contact_models: Dict[str, Dict[str, Any]],
    group_codes_info: Dict[str, Dict[str, Any]],
) -> Dict[str, nb.typed.List]:
    """Create indexers for matching individuals within contact models.

    For each contact model, :func:`create_group_indexer` returns a Numba list where each
    position contains a :class:`numpy.ndarray` with all the indices of individuals
    belonging to the same group given by the index.

    The indexer has one Numba list for recurrent and random models. Each list has one
    entry per contact model which holds the result of :func:`create_group_indexer`.

    Args:
        states (pandas.DataFrame): see :ref:`states`.
        contact_models (Dict[str, Dict[str, Any]]): The contact models.
        group_codes_info (Dict[str, Dict[str, Any]]): A dictionary where keys are names
          of contact models and values are dictionaries containing the name and the
          original codes of the assortative variables.

    Returns:
        indexers (Dict[str, numba.typed.List]): The indexer is a dictionary with one
            entry for recurrent and random contact models. The values are Numba lists
            containing Numba lists for each contact model. Each list holds indices for
            each group in the contact model.

    """
    recurrent_models, random_models = separate_contact_model_names(contact_models)

    indexers = {"recurrent": nb.typed.List(), "random": nb.typed.List()}
    for cm in recurrent_models:
        indexer = create_group_indexer(states, group_codes_info[cm]["name"])
        indexers["recurrent"].append(indexer)
    for cm in random_models:
        indexer = create_group_indexer(states, group_codes_info[cm]["name"])
        indexers["random"].append(indexer)

    return indexers


def _prepare_assortative_matching_cumulative_probabilities(
    states: pd.DataFrame,
    assort_bys: Dict[str, List[str]],
    params: pd.DataFrame,
    contact_models: Dict[str, Dict[str, Any]],
    group_codes_info: Dict[str, Dict[str, Any]],
) -> nb.typed.List:
    """Create first stage probabilities for assortative matching with random contacts.

    Args:
        states (pandas.DataFrame): See :ref:`states`.
        assort_bys (Dict[str, List[str]]): Keys are names of contact models, values are
            lists with the assort_by variables of the model.
        params (pandas.DataFrame): See :ref:`params`.
        contact_models (dict): see :ref:`contact_models`.
        group_codes_info (Dict[str, Dict[str, Any]]): A dictionary where keys are names
          of contact models and values are dictionaries containing the name and the
          original codes of the assortative variables.

    Returns:
        probabilities (numba.typed.List): The list contains one entry for each random
            contact model. Each entry holds a ``n_groups * n_groups`` transition matrix
            where ``probs[i, j]`` is the cumulative probability that an individual from
            group ``i`` meets someone from group ``j``.

    """
    probabilities = nb.typed.List()
    for model_name, assort_by in assort_bys.items():
        if not contact_models[model_name]["is_recurrent"]:
            probs = create_cumulative_group_transition_probabilities(
                states,
                assort_by,
                params,
                model_name,
                group_codes_info[model_name]["groups"],
            )
            probabilities.append(probs)

    # The nopython mode fails while calculating infections, if we leave the list empty
    # or put a 1d array inside the list.
    if len(probabilities) == 0:
        probabilities.append(np.zeros((0, 0)))

    return probabilities


def _add_default_duration_to_models(
    dictionaries: Dict[str, Dict[str, Any]], duration: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Add default durations to models."""
    for name, model in dictionaries.items():
        start = pd.Timestamp(model.get("start", duration["start"]))
        end = pd.Timestamp(model.get("end", duration["end"]))

        m = "The {} date of model '{}' could not be converted to a valid pd.Timestamp."
        if pd.isna(start):
            raise ValueError(m.format("start", name))
        if pd.isna(end):
            raise ValueError(m.format("end", name))

        if end < start:
            raise ValueError(
                f"The end date of model '{name}' is before the start date."
            )

        dictionaries[name] = {
            **model,
            "start": start,
            "end": end,
        }

    return dictionaries


def _process_initial_states(
    states: pd.DataFrame,
    assort_bys: Dict[str, List[str]],
    virus_strains: Dict[str, Any],
) -> pd.DataFrame:
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

    for col in BOOLEAN_STATE_COLUMNS:
        if col not in states.columns:
            states[col] = False

    for col in COUNTDOWNS:
        if col not in states.columns:
            states[col] = DTYPE_COUNTDOWNS(-99)
        states[col] = states[col].astype(DTYPE_COUNTDOWNS)

    states["n_has_infected"] = DTYPE_INFECTION_COUNTER(0)
    states["pending_test_date"] = pd.NaT
    states["virus_strain"] = pd.Categorical(
        [pd.NA] * len(states), categories=virus_strains["names"]
    )

    return states


def _create_group_codes_and_info(
    states: pd.DataFrame,
    assort_bys: Dict[str, List[str]],
    contact_models: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Create group codes and additional information.

    Args:
        states (pd.DataFrame): The states.
        assort_bys (Dict[str, List[str]]): The assortative variables for each contact
            model.
        contact_models (Dict[str, Dict[str, Any]]): The contact models.

    Returns:
        A tuple containing:

        - states (pandas.DataFrame): The states.
        - group_codes_info (Dict[str, Dict[str, Any]]): A dictionary where keys are
          names of contact models and values are dictionaries containing the name and
          the original codes of the assortative variables.

    """
    group_codes_names = _create_group_codes_names(contact_models, assort_bys)

    group_codes_info = {}

    for model_name, assort_by in assort_bys.items():
        is_factorized = contact_models[model_name].get("is_factorized", False)
        group_code_name = group_codes_names[model_name]

        # Create the group code column if it is not available or if it exists - meaning
        # we are resuming a simulation - to recover the groups.
        if (group_code_name not in states.columns) or (
            group_code_name in states.columns and not is_factorized
        ):
            states[group_code_name], groups = factorize_assortative_variables(
                states, assort_by
            )
        elif group_code_name in states.columns and is_factorized:
            states[group_code_name] = states[group_code_name].astype(DTYPE_GROUP_CODE)
            unsorted_groups = states[group_code_name].unique()
            groups = np.sort(unsorted_groups[unsorted_groups != -1])
        else:
            groups = states[group_code_name].cat.categories
            if is_factorized:
                groups = groups[groups != -1]

        group_codes_info[model_name] = {"name": group_code_name, "groups": groups}

    return states, group_codes_info


def _dump_periodic_states(states, columns_to_keep, output_directory, date) -> None:
    """Dump the states of one period."""
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
    saved_columns: Union[None, Dict[str, Union[bool, str, List[str]]]],
    initial_state_columns: List[str],
    group_codes_info: Dict[str, str],
    contact_models: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Process saved columns.

    This functions combines the user-defined ``saved_columns`` with the default and
    produces a list of columns names which should be kept in the periodic states.

    The list is also used to check whether additional information should be computed and
    then stored in the periodic states.

    Args:
        saved_columns (Union[None, Dict[str, Union[bool, str, List[str]]]]): The columns
            the user decided to save in the simulation output.
        initial_state_columns (List[str]): The columns available in the initial states
            passed by the user.
        group_codes_info (Dict[str, str]): A dictionary which contains the name and
            groups for each group code variable.
        contact_models (Dict[str, Dict[str, Any]]): The contact models.

    Returns:
        keep (List[str]): A list of columns names which should be kept in the states.

    """
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
        "group_codes": [
            group_codes_info[model]["name"]
            for model in group_codes_info
            if group_codes_info[model]["name"].startswith("group_codes_")
        ],
        "channels": [
            "channel_infected_by_contact",
            "channel_infected_by_event",
            "channel_demands_test",
        ],
        "rapid_tests": ["is_tested_positive_by_rapid_test"],
    }

    keep = []
    for category in all_columns:
        keep += _combine_column_lists(saved_columns[category], all_columns[category])

    if isinstance(saved_columns["other"], list):
        keep += saved_columns["other"]

    # drop duplicates
    keep = list(set(keep))

    return keep


def _combine_column_lists(user_entries, all_entries):
    """Combine multiple lists."""
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


def _are_states_prepared(states: pd.DataFrame) -> bool:
    """Are states prepared.

    If the states include information on the period or date, we assume that the states
    are prepared.

    """
    return states.columns.isin(["date", "period"]).any()


def _add_additional_information_to_states(
    states: pd.DataFrame,
    columns_to_keep: List[str],
    n_has_additionally_infected: pd.Series,
    random_contacts: np.ndarray,
    recurrent_contacts: np.ndarray,
    channel_infected_by_contact: pd.Series,
    channel_infected_by_event: pd.Series,
    channel_demands_test: pd.Series,
    susceptibility_factor: np.ndarray,
):
    """Add additional but optional information to states.

    Args:
        states (pandas.DataFrame): The states of one period.
        columns_to_keep (List[str]): A list of columns names which should be kept.
        n_has_additionally_infected (Optional[pandas.Series]): Additionally infected
            persons by this individual.
        contacts (numpy.ndarray): Matrix with number of contacts for each contact model.
        channel_infected_by_contact (pandas.Series): A categorical series containing the
            information which contact model lead to the infection.
        channel_infected_by_event (pandas.Series): A categorical series containing the
            information which event model lead to the infection.
        susceptibility_factor (numpy.ndarray): An array containing infection
            probability multiplier for each individual.

    Returns:
        states (pandas.DataFrame): The states with additional information.

    """
    if recurrent_contacts is not None:
        for name in recurrent_contacts.columns:
            states[f"n_contacts_{name}"] = recurrent_contacts[name]
    if random_contacts is not None:
        for name in random_contacts.columns:
            states[f"n_contacts_{name}"] = random_contacts[name]

    if (
        channel_infected_by_contact is not None
        and "channel_infected_by_contact" in columns_to_keep
    ):
        states["channel_infected_by_contact"] = channel_infected_by_contact

    if (
        channel_infected_by_event is not None
        and "channel_infected_by_event" in columns_to_keep
    ):
        states["channel_infected_by_event"] = channel_infected_by_event

    if channel_demands_test is not None and "channel_demands_test" in columns_to_keep:
        states["channel_demands_test"] = channel_demands_test

    if n_has_additionally_infected is not None:
        states["n_has_infected"] += n_has_additionally_infected

    if "susceptibility_factor" in columns_to_keep:
        states["susceptibility_factor"] = susceptibility_factor

    return states
