import itertools
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd
from sid.config import RELATIVE_POPULATION_PARAMETER
from sid.countdowns import COUNTDOWNS
from sid.shared import fast_condition_evaluator
from sid.virus_strains import categorize_factorized_infections
from sid.virus_strains import combine_first_factorized_infections


def update_states(
    states: pd.DataFrame,
    newly_infected_contacts: pd.Series,
    newly_infected_events: pd.Series,
    params: pd.DataFrame,
    virus_strains: Dict[str, Any],
    to_be_processed_tests: Optional[pd.Series],
    newly_vaccinated: pd.Series,
    seed: itertools.count,
    derived_state_variables,
):
    """Update the states with new infections and advance it by one period.

    States are changed in place to save copying!

    Args:
        states (pandas.DataFrame): See :ref:`states`.
        newly_infected_contacts (pandas.Series): Boolean series indicating individuals
            infected by contacts. There can be an overlap with infections by events.
        newly_infected_events (pandas.Series): Boolean series indicating individuals
            infected by events. There can be an overlap with infections by contacts.
        params (pandas.DataFrame): See :ref:`params`.
        virus_strains (Dict[str, Any]): A dictionary with the keys ``"names"`` and
            ``"factors"`` holding the different contagiousness factors of multiple
            viruses.
        to_be_processed_tests (pandas.Series): Tests which are going to be processed.
        newly_vaccinated (Optional[pandas.Series]): A series which indicates newly
            vaccinated people.
        seed (itertools.count): Seed counter to control randomness.
        derived_state_variables (Dict[str, str]): A dictionary that maps
            names of state variables to pandas evaluation strings that generate derived
            state variables, i.e. state variables that can be calculated from the
            existing state variables.


    Returns: states (pandas.DataFrame): Updated states with reduced countdown lengths,
        newly started countdowns, and killed people over the ICU limit.

    """
    states = _update_countdowns(states)

    states = _update_info_on_newly_infected(
        states, newly_infected_contacts, newly_infected_events, virus_strains
    )

    states = _kill_people_over_icu_limit(states, params, next(seed))

    # important: this has to be called after _kill_people_over_icu_limit!
    states["newly_deceased"] = states["cd_dead_true"] == 0

    if to_be_processed_tests is not None:
        states = _update_info_on_new_tests(states, to_be_processed_tests)

    states = _update_info_on_new_vaccinations(states, newly_vaccinated)

    states = update_derived_state_variables(states, derived_state_variables)

    return states


def _update_countdowns(states):
    """Update countdowns."""
    # Reduce all existing countdowns by 1.
    for countdown in COUNTDOWNS:
        states[countdown] -= 1

    # Make changes where the countdown is zero.
    for countdown, info in COUNTDOWNS.items():
        locs = states.index[states[countdown] == 0]
        for to_change, new_val in info.get("changes", {}).items():
            states.loc[locs, to_change] = new_val

        for new_countdown in info.get("starts", []):
            states.loc[locs, new_countdown] = states.loc[locs, f"{new_countdown}_draws"]

    return states


def _update_info_on_newly_infected(
    states, newly_infected_contacts, newly_infected_events, virus_strains
):
    """Update information with newly infected individuals."""
    # Update states with new infections and add corresponding countdowns.
    states["newly_infected"] = (newly_infected_contacts >= 0) | (
        newly_infected_events >= 0
    )

    combined_newly_infected = combine_first_factorized_infections(
        newly_infected_contacts, newly_infected_events
    )
    newly_virus_strain = categorize_factorized_infections(
        combined_newly_infected, virus_strains
    )

    needs_replacement = newly_virus_strain.notnull() & states["virus_strain"].isnull()
    states["virus_strain"] = states["virus_strain"].where(
        ~needs_replacement, newly_virus_strain
    )

    locs = states["newly_infected"]
    states.loc[states["newly_infected"], "immune"] = True
    states.loc[locs, "ever_infected"] = True
    states.loc[locs, "cd_ever_infected"] = 0
    states.loc[locs, "cd_immune_false"] = states.loc[locs, "cd_immune_false_draws"]
    states.loc[locs, "cd_infectious_true"] = states.loc[
        locs, "cd_infectious_true_draws"
    ]

    return states


def _kill_people_over_icu_limit(states, params, seed):
    """Kill people over the ICU limit."""
    np.random.seed(seed)

    relative_limit = params.loc[
        ("health_system", "icu_limit_relative", "icu_limit_relative"), "value"
    ]
    absolute_limit = int(relative_limit * len(states) * RELATIVE_POPULATION_PARAMETER)
    need_icu_locs = states.index[states["needs_icu"]]
    if absolute_limit < len(need_icu_locs):
        excess = int(len(need_icu_locs) - absolute_limit)
        to_kill = np.random.choice(need_icu_locs, size=excess, replace=False)
        for to_change, new_val in COUNTDOWNS["cd_dead_true"]["changes"].items():
            states.loc[to_kill, to_change] = new_val
        states.loc[to_kill, "cd_dead_true"] = 0

    return states


def _update_info_on_new_tests(
    states: pd.DataFrame, to_be_processed_tests: pd.Series
) -> pd.DataFrame:
    # Remove information on pending tests for tests which are processed.
    states.loc[to_be_processed_tests, "pending_test_date"] = pd.NaT

    # Start the countdown for processed tests.
    states.loc[to_be_processed_tests, "cd_received_test_result_true"] = states.loc[
        to_be_processed_tests, "cd_received_test_result_true_draws"
    ]

    # For everyone who received a test result, the countdown for the test processing
    # has expired. If you have a positive test result (received_test_result &
    # immune) you will leave the state of knowing until your immunity expires.
    states["new_known_case"] = states["received_test_result"] & states["immune"]
    states.loc[states["new_known_case"], "knows_immune"] = True
    states.loc[states["new_known_case"], "cd_knows_immune_false"] = states.loc[
        states["new_known_case"], "cd_immune_false"
    ]

    new_knows_infectious = (
        states["knows_immune"] & states["infectious"] & states["new_known_case"]
    )
    states.loc[new_knows_infectious, "knows_infectious"] = True
    states.loc[new_knows_infectious, "cd_knows_infectious_false"] = states.loc[
        new_knows_infectious, "cd_infectious_false"
    ]

    # Everyone looses ``received_test_result == True`` because it is passed to the
    # more specific knows attributes.
    states.loc[states["received_test_result"], "received_test_result"] = False

    return states


def _update_info_on_new_vaccinations(
    states: pd.DataFrame, newly_vaccinated: pd.Series
) -> pd.DataFrame:
    """Activate the counter for immunity by vaccinations."""
    states["newly_vaccinated"] = newly_vaccinated
    states.loc[newly_vaccinated, "ever_vaccinated"] = True
    states.loc[newly_vaccinated, "cd_is_immune_by_vaccine"] = states.loc[
        newly_vaccinated, "cd_is_immune_by_vaccine_draws"
    ]

    return states


def update_derived_state_variables(states, derived_state_variables):
    for var, condition in derived_state_variables.items():
        states[var] = fast_condition_evaluator(states, condition)
    return states
