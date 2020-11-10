import numpy as np
import pandas as pd
from sid.config import RELATIVE_POPULATION_PARAMETER
from sid.countdowns import COUNTDOWNS


def update_states(
    states,
    newly_infected_contacts,
    newly_infected_events,
    params,
    seed,
    optional_state_columns,
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
        optional_state_columns (dict): Dictionary with categories of state columns
            that can additionally be added to the states dataframe, either for use in
            contact models and policies or to be saved. Most types of columns are added
            by default, but some of them are costly to add and thus only added when
            needed. Columns that are not in the state but specified in ``saved_columns``
            will not be saved. The categories are "contacts" and "reason_for_infection".

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

    # Save channel of infection; For speed reasons start with integer labels and
    # convert to string labels later
    if optional_state_columns["reason_for_infection"]:
        labels = {0: "contact or event", 1: "contact", 2: "event"}
        channel = np.zeros(len(states))
        newly_infected_contacts = newly_infected_contacts.to_numpy()
        newly_infected_events = newly_infected_events.to_numpy()
        channel[newly_infected_contacts & ~newly_infected_events] = 1
        channel[newly_infected_events & ~newly_infected_contacts] = 2
        # set categories is necessary in case one of the categories was not present in
        # the data. Setting them via set_categories is much faster than passing them
        # into pd.Categorical directly
        states["newly_infected_reason"] = (
            pd.Categorical(channel).set_categories([0, 1, 2]).rename_categories(labels)
        )

    # Update states with new infections and add corresponding countdowns.
    locs = states.query("newly_infected").index
    states.loc[locs, "ever_infected"] = True
    states.loc[locs, "cd_ever_infected"] = 0
    states.loc[locs, "cd_immune_false"] = states.loc[locs, "cd_immune_false_draws"]
    states.loc[locs, "cd_infectious_true"] = states.loc[
        locs, "cd_infectious_true_draws"
    ]

    states = _kill_people_over_icu_limit(states, params, seed)

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
    np.random.seed(next(seed))

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
