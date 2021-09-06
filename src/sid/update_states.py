import itertools
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd
from sid.config import DTYPE_IMMUNITY
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
        virus_strains (Dict[str, Any]): A dictionary with the keys ``"names"``,
            ``"contagiousness_factor"`` and ``"immunity_resistance_factor"`` holding the
            different contagiousness factors and immunity resistance factors of multiple
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

    # important: this has to be called after _update_info_on_newly_infected and
    # _update_info_on_new_vaccinations, as it consolidates information in states that is
    # created by these two functions!
    states = _update_immunity_level(states, params)

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
    states.loc[locs, "ever_infected"] = True
    states.loc[locs, "cd_ever_infected"] = 0
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
    states["new_known_case"] = states["received_test_result"] & (
        states["infectious"] | states["symptomatic"]
    )
    states.loc[states["new_known_case"], "knows_immune"] = True

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
    states.loc[newly_vaccinated, "cd_ever_vaccinated"] = 0
    return states


def update_derived_state_variables(states, derived_state_variables):
    for var, condition in derived_state_variables.items():
        states[var] = fast_condition_evaluator(states, condition)
    return states


def _update_immunity_level(states: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    """Update immunity levels from infection and vaccination."""
    days_since_infection = -states["cd_ever_infected"]
    days_since_vaccination = -states["cd_ever_vaccinated"]

    immunity_from_infection = _compute_waning_immunity(
        params, days_since_event=days_since_infection, event="infection"
    )
    immunity_from_vaccination = _compute_waning_immunity(
        params, days_since_event=days_since_vaccination, event="vaccination"
    )

    states["immunity"] = np.maximum(immunity_from_infection, immunity_from_vaccination)
    return states


def _compute_waning_immunity(
    params: pd.DataFrame, days_since_event: pd.Series, event: str
) -> pd.Series:
    """Compute waning immunity level.

    The immunity level is a simple piece-wise function parametrized over the maximum
    level of achievable immunity, the number of days it takes to achieve this maximum
    and the linear slope determining how fast the immunity level decreases after
    reaching the maximum. Before the maximum is achieved the function is modeled as a
    third-degree polynomial and afterwards as a linear function. The coefficients are
    adjusted to the parameters and set in ``_get_waning_immunity_coefficients``.

    Args:
        params (pandas.DataFrame): See :ref:`params`.
        days_since_event (pandas.Series): Series containing days since event occurred
            for each individual in the state.
        event (str): Reason for immunity. Must be in {"infection", "vaccination"}.

    Returns:
        immunity (pandas.Series): Adjusted immunity level.

    """
    coef = _get_waning_immunity_coefficients(params, event)
    immunity = pd.Series(0, index=days_since_event.index, dtype=DTYPE_IMMUNITY)

    before_maximum = (days_since_event > 0) & (
        days_since_event < coef["time_to_reach_maximum"]
    )
    after_maximum = days_since_event >= coef["time_to_reach_maximum"]

    # increase immunity level for individuals who have not reached their maximum level
    immunity[before_maximum] = coef["slope_before_maximum"] * (
        days_since_event[before_maximum] ** 3
    )

    # decrease immunity level for individuals who are beyond their maximum level
    immunity[after_maximum] = (
        coef["intercept"]
        + coef["slope_after_maximum"] * days_since_event[after_maximum]
    )

    # make sure that immunity level is non-negative, which is not automatically enforced
    # by the linear function specification
    immunity = immunity.clip(lower=0)
    return immunity


def _get_waning_immunity_coefficients(
    params: pd.DataFrame, event: str
) -> Dict[str, float]:
    """Transform high-level arguments for waning immunity to low-level coefficients.

    Coefficients are calibrated to parameters in params.

    Args:
        params (pandas.DataFrame): See :ref:`params`.
        event (str): Reason for immunity. Must be in {"infection", "vaccination"}.

    Returns:
        coef (Dict[str, float]): The coefficients.

    """
    maximum_immunity = params.loc[
        ("immunity", "immunity_level", f"from_{event}"), "value"
    ]
    time_to_reach_maximum = params.loc[
        ("immunity", "immunity_waning", f"time_to_reach_maximum_{event}"), "value"
    ]
    slope_after_maximum = params.loc[
        ("immunity", "immunity_waning", f"slope_after_maximum_{event}"), "value"
    ]

    slope_before_maximum = maximum_immunity / (time_to_reach_maximum ** 3)
    intercept = maximum_immunity - slope_after_maximum * time_to_reach_maximum
    coef = {
        "time_to_reach_maximum": time_to_reach_maximum,
        "slope_before_maximum": slope_before_maximum,
        "slope_after_maximum": slope_after_maximum,
        "intercept": intercept,
    }
    return coef
