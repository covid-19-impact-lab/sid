import itertools
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd
from sid.config import RELATIVE_POPULATION_PARAMETER
from sid.countdowns import COUNTDOWNS


def update_states(
    states: pd.DataFrame,
    newly_infected_contacts: pd.Series,
    newly_infected_events: pd.Series,
    params: pd.DataFrame,
    seed: itertools.count,
    optional_state_columns: Dict[str, Any] = None,
    n_has_additionally_infected: Optional[pd.Series] = None,
    indexers: Optional[Dict[int, np.ndarray]] = None,
    contacts: Optional[np.ndarray] = None,
    to_be_processed_test: Optional[pd.Series] = None,
    channel_infected_by_contact: Optional[pd.Series] = None,
    channel_infected_by_event: Optional[pd.Series] = None,
    channel_demands_test: Optional[pd.Series] = None,
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
        seed (itertools.count): Seed counter to control randomness.
        optional_state_columns (dict): Dictionary with categories of state columns
            that can additionally be added to the states dataframe, either for use in
            contact models and policies or to be saved. Most types of columns are added
            by default, but some of them are costly to add and thus only added when
            needed. Columns that are not in the state but specified in ``saved_columns``
            will not be saved. The categories are "contacts" and "reason_for_infection".
        n_has_additionally_infected (Optional[pandas.Series]): Additionally infected
            persons by this individual.
        indexers (dict): Dictionary with contact models as keys in the same order as the
            contacts matrix.
        contacts (numpy.ndarray): Matrix with number of contacts for each contact model.
        to_be_processed_test (pandas.Series): Tests which are going to be processed.
        channel_infected_by_contact (pandas.Series): A categorical series containing the
            information which contact model lead to the infection.
        channel_infected_by_event (pandas.Series): A categorical series containing the
            information which event model lead to the infection.

    Returns: states (pandas.DataFrame): Updated states with reduced countdown lengths,
        newly started countdowns, and killed people over the ICU limit.

    """
    if optional_state_columns is None:
        optional_state_columns = {"reason_for_infection": False, "contacts": False}

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

    if channel_infected_by_contact is not None:
        states["channel_infected_by_contact"] = channel_infected_by_contact

    if channel_infected_by_event is not None:
        states["channel_infected_by_event"] = channel_infected_by_event

    # Update states with new infections and add corresponding countdowns.
    locs = states.query("newly_infected").index
    states.loc[locs, "ever_infected"] = True
    states.loc[locs, "cd_ever_infected"] = 0
    states.loc[locs, "cd_immune_false"] = states.loc[locs, "cd_immune_false_draws"]
    states.loc[locs, "cd_infectious_true"] = states.loc[
        locs, "cd_infectious_true_draws"
    ]

    states = _kill_people_over_icu_limit(states, params, next(seed))

    # important: this has to be called after _kill_people_over_icu_limit!
    states["newly_deceased"] = states["cd_dead_true"] == 0

    # Add additional information.
    if optional_state_columns["contacts"]:
        if isinstance(optional_state_columns, list):
            cols_to_add = optional_state_columns["contacts"]
        else:
            cols_to_add = [f"n_contacts_{model}" for model in indexers]
        if indexers is not None and contacts is not None:
            for i, contact_model in enumerate(indexers):
                if f"n_contacts_{contact_model}" in cols_to_add:
                    states[f"n_contacts_{contact_model}"] = contacts[:, i]

    if channel_demands_test is not None and optional_state_columns["channels"]:
        states["channel_demands_test"] = channel_demands_test

    if n_has_additionally_infected is not None:
        states["n_has_infected"] += n_has_additionally_infected

    # Perform steps if testing is enabled.
    if to_be_processed_test is not None:
        # Remove information on pending tests for tests which are processed.
        states.loc[to_be_processed_test, "pending_test_date"] = pd.NaT

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

        states["new_known_case"] = (
            states["cd_received_test_result_true"] == 0
        ) & states["immune"]

        # Everyone looses ``received_test_result == True`` because it is passed to the
        # more specific knows attributes.
        states.loc[states.received_test_result, "received_test_result"] = False

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
