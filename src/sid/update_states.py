import numpy as np
from sid.config import COUNTDOWNS


def update_states(
    states,
    newly_infected_contacts,
    newly_infected_events,
    params,
    seed,
    n_has_additionally_infected=None,
    cum_probs=None,
    contacts=None,
):
    """Update the states with new infections and advance it by one period.

    States are changed in place to save copying!

    Args:
        states (pandas.DataFrame): See :ref:`states`.
        newly_infected (pandas.Series): Boolean Series with same index as states.
        params (pandas.DataFrame): See :ref:`params`.
        seed (itertools.count): Seed counter to control randomness.

    Returns: states (pandas.DataFrame): Updated states with reduced countdown lengths,
        newly started countdowns, and killed people over the ICU limit.

    """
    # Reduce all existing countdowns by 1.
    for countdown in COUNTDOWNS:
        # TODO: Investigate whether countdowns should all be reduced by 1 or only active
        # ones.
        states[countdown] -= 1

    # Make changes where the countdown is zero.
    for countdown, info in COUNTDOWNS.items():
        locs = states.index[states[countdown] == 0]
        for to_change, new_val in info["changes"].items():
            states.loc[locs, to_change] = new_val

        for new_countdown in info.get("starts", []):
            states.loc[locs, new_countdown] = states.loc[locs, f"{new_countdown}_draws"]

    states["newly_infected"] = newly_infected_contacts | newly_infected_events
    states["immune"] = states["immune"] | states["newly_infected"]

    # Save channel of infection.
    states["newly_infected_reason"] = "contact or event"
    states.loc[
        newly_infected_contacts & ~newly_infected_events, "newly_infected_reason"
    ] = "contact"
    states.loc[
        newly_infected_events & ~newly_infected_contacts, "newly_infected_reason"
    ] = "event"
    states["newly_infected_reason"] = states["newly_infected_reason"].astype("category")

    # Update states with new infections and add corresponding countdowns.
    locs = states.query("newly_infected").index
    states.loc[locs, "ever_infected"] = True
    states.loc[locs, "cd_immune_false"] = states.loc[locs, "cd_immune_false_draws"]
    states.loc[locs, "cd_infectious_true"] = states.loc[
        locs, "cd_infectious_true_draws"
    ]

    states = _kill_people_over_icu_limit(states, params, seed)

    # Add additional information.
    if cum_probs is not None and contacts is not None:
        for i, contact_model in enumerate(cum_probs):
            states[f"n_contacts_{contact_model}"] = contacts[:, i]

    if n_has_additionally_infected is not None:
        states["n_has_infected"] += n_has_additionally_infected

    return states


def add_debugging_information(states, newly_missed_contacts):
    """Add some information to states which is more useful to debug the model."""
    for column in newly_missed_contacts:
        states[column] = newly_missed_contacts[column]
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
