import numpy as np
import pandas as pd
from sid.countdowns import COUNTDOWNS


def update_states(
    states,
    newly_infected_contacts,
    newly_infected_events,
    params,
    seed,
    n_has_additionally_infected=None,
    indexers=None,
    contacts=None,
    to_be_processed_test=None,
):
    """Update the states with new infections and advance it by one period.

    States are changed in place to save copying!

    Args:
        states (pandas.DataFrame): See :ref:`states`.
        newly_infected (pandas.Series): Boolean Series with same index as states.
        params (pandas.DataFrame): See :ref:`params`.
        seed (itertools.count): Seed counter to control randomness.
        n_has_additionally_infected (pandas.Series): Additionally infected persons by
            this individual.
        indexers (dict): Dictionary with contact models as keys in the same order as the
            contacts matrix.
        contacts (numpy.ndarray): Matrix with number of contacts for each contact model.
        to_be_processed_test (pandas.Series): Tests which are going to be processed.

    Returns: states (pandas.DataFrame): Updated states with reduced countdown lengths,
        newly started countdowns, and killed people over the ICU limit.

    """
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

    states["newly_infected"] = newly_infected_contacts | newly_infected_events
    states["immune"] = states["immune"] | states["newly_infected"]

    # Save channel of infection.
    if "newly_infected_reason" not in states:
        states["newly_infected_reason"] = pd.Categorical(
            np.full(len(states), np.nan),
            categories=["contact", "contact or event", "event"],
        )
    for condition, label in [
        (states.index, "contact or event"),
        (newly_infected_contacts & ~newly_infected_events, "contact"),
        (newly_infected_events & ~newly_infected_contacts, "event"),
    ]:
        states.loc[condition, "newly_infected_reason"] = label
    states["newly_infected_reason"] = states["newly_infected_reason"].astype("category")

    # Update states with new infections and add corresponding countdowns.
    locs = states.query("newly_infected").index
    states.loc[locs, "ever_infected"] = True
    states.loc[locs, "cd_ever_infected"] = 0
    states.loc[locs, "cd_immune_false"] = states.loc[locs, "cd_immune_false_draws"]
    states.loc[locs, "cd_infectious_true"] = states.loc[
        locs, "cd_infectious_true_draws"
    ]

    states = _kill_people_over_icu_limit(states, params, seed)

    # Add additional information.
    if indexers is not None and contacts is not None:
        for i, contact_model in enumerate(indexers):
            states[f"n_contacts_{contact_model}"] = contacts[:, i]

    if n_has_additionally_infected is not None:
        states["n_has_infected"] += n_has_additionally_infected

    # Perform steps if testing is enabled.
    if to_be_processed_test is not None:
        # Remove information on pending tests for tests which are processed.
        states.loc[to_be_processed_test, "pending_test_date"] = pd.NaT
        states.loc[to_be_processed_test, "pending_test_period"] = np.nan

        # Start the countdown for processed tests.
        states.loc[to_be_processed_test, "cd_received_test_result_true"] = states.loc[
            to_be_processed_test, "cd_received_test_result_true_draws"
        ]

        # For everyone who received a test result, the countdown for the test processing
        # has expired. If you have a positive test result (received_test_result &
        # immune) you will leave the state of knowing until your immunity expires.
        knows_immune = states.received_test_result & states.immune
        states.loc[knows_immune, "cd_knows_immune_false"] = states.loc[
            knows_immune, "cd_immune_false"
        ]
        states.loc[knows_immune, "knows_immune"] = True

        knows_infectious = knows_immune & states.infectious
        states.loc[knows_infectious, "cd_knows_infectious_false"] = states.loc[
            knows_infectious, "cd_infectious_false"
        ]
        states.loc[knows_infectious, "knows_infectious"] = True

        # Everyone looses ``received_test_result == True`` because it is passed to the
        # more specific knows attributes.
        states.loc[states.received_test_result, "received_test_result"] = False

    return states


def add_debugging_information(
    states, newly_missed_contacts, demands_test, allocated_tests, to_be_processed_tests
):
    """Add some information to states which is more useful to debug the model."""
    for column in newly_missed_contacts:
        states[column] = newly_missed_contacts[column]
    if demands_test is not None:
        states["demands_test"] = demands_test
        states["allocated_test"] = allocated_tests
        states["to_be_processed_test"] = to_be_processed_tests

    return states


def _kill_people_over_icu_limit(states, params, seed):
    """Kill people over the ICU limit."""
    np.random.seed(next(seed))

    rel_limit = params.loc[
        ("health_system", "icu_limit_relative", "icu_limit_relative"), "value"
    ]
    abs_limit = rel_limit * len(states)
    need_icu_locs = states.index[states["needs_icu"]]
    if abs_limit < len(need_icu_locs):
        excess = int(len(need_icu_locs) - abs_limit)
        to_kill = np.random.choice(need_icu_locs, size=excess, replace=False)
        for to_change, new_val in COUNTDOWNS["cd_dead_true"]["changes"].items():
            states.loc[to_kill, to_change] = new_val
        states.loc[to_kill, "cd_dead_true"] = 0

    return states
