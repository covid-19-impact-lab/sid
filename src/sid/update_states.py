import itertools
from typing import Optional

import numpy as np
import pandas as pd
from sid.config import IS_NEWLY_DECEASED
from sid.config import KNOWS_INFECTIOUS
from sid.config import RECEIVES_POSITIVE_TEST
from sid.config import RELATIVE_POPULATION_PARAMETER
from sid.countdowns import COUNTDOWNS


def update_states(
    states: pd.DataFrame,
    newly_infected_contacts: pd.Series,
    newly_infected_events: pd.Series,
    params: pd.DataFrame,
    share_known_cases: Optional[float],
    to_be_processed_tests: Optional[pd.Series],
    seed: itertools.count,
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
        share_known_cases (Optional[float]): Share of known cases.
        to_be_processed_tests (pandas.Series): Tests which are going to be processed.
        seed (itertools.count): Seed counter to control randomness.

    Returns: states (pandas.DataFrame): Updated states with reduced countdown lengths,
        newly started countdowns, and killed people over the ICU limit.

    """
    states = _update_countdowns(states)

    states = _update_info_on_newly_infected(
        states, newly_infected_contacts, newly_infected_events
    )

    states = _kill_people_over_icu_limit(states, params, next(seed))

    # important: this has to be called after _kill_people_over_icu_limit!
    states["newly_deceased"] = states.eval(IS_NEWLY_DECEASED)

    if share_known_cases is not None:
        to_be_processed_tests = _compute_new_tests_with_share_known_cases(
            states, share_known_cases
        )

    if to_be_processed_tests is not None:
        states = _update_info_on_new_tests(states, to_be_processed_tests)

    return states


def _update_countdowns(states):
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
    states, newly_infected_contacts, newly_infected_events
):
    # Update states with new infections and add corresponding countdowns.
    states["newly_infected"] = newly_infected_contacts | newly_infected_events

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


def _compute_new_tests_with_share_known_cases(
    states: pd.DataFrame, share_known_cases: float
) -> pd.Series:
    """Compute the new tests based on the share of known cases.

    The share of known cases is based on newly infected individuals.

    """
    n_new_known_cases = int(states["newly_infected"].sum() * share_known_cases)

    if n_new_known_cases > 0:
        ilocs = np.arange(len(states))[states["newly_infected"]]
        sampled_ilocs = np.random.choice(ilocs, size=n_new_known_cases, replace=False)
    else:
        sampled_ilocs = slice(0)

    new_tests = pd.Series(index=states.index, data=False)
    new_tests.iloc[sampled_ilocs] = True

    return new_tests


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
    states["new_known_case"] = states.eval(RECEIVES_POSITIVE_TEST)
    states.loc[states["new_known_case"], "knows_immune"] = True
    states.loc[states["new_known_case"], "cd_knows_immune_false"] = states.loc[
        states["new_known_case"], "cd_immune_false"
    ]

    new_knows_infectious = states.eval(KNOWS_INFECTIOUS) & states["new_known_case"]
    states.loc[new_knows_infectious, "knows_infectious"] = True
    states.loc[new_knows_infectious, "cd_knows_infectious_false"] = states.loc[
        new_knows_infectious, "cd_infectious_false"
    ]

    # Everyone looses ``received_test_result == True`` because it is passed to the
    # more specific knows attributes.
    states.loc[states["received_test_result"], "received_test_result"] = False

    return states
