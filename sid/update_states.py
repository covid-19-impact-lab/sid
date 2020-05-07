import numpy as np

from sid.config import COUNTDOWNS


def update_states(states, newly_infected, params, seed):
    """Update the states with new infections and advance it by one period.

    States are changed in place to save copying!

    Args:
        states (pandas.DataFrame): See :ref:`states`.
        newly_infected (pandas.Series): Boolean Series with same index as states.
        params (pandas.DataFrame): See :ref:`params`.
        seed (itertools.count): Seed counter to control randomness.

    Returns: states (pandas.DataFrame): Updated states with reduced countdown lengths,
        newly started countdowns, and killed people over the icu limit.

    """
    np.random.seed(next(seed))

    # Reduce all existing countdowns by 1.
    for countdown in COUNTDOWNS:
        states[countdown] -= 1

    # Make changes where the countdown is zero.
    for countdown, info in COUNTDOWNS.items():
        locs = states.index[states[countdown] == 0]
        for to_change, new_val in info["changes"].items():
            states.loc[locs, to_change] = new_val

        for new_countdown in info.get("starts", []):
            states.loc[locs, new_countdown] = states.loc[locs, f"{new_countdown}_draws"]

    # Update states with new infections and add corresponding countdowns.
    locs = newly_infected[newly_infected].index
    states.loc[locs, "ever_infected"] = True
    states.loc[locs, "cd_immune_false"] = states.loc[locs, "cd_immune_false_draws"]
    states.loc[locs, "cd_infectious_true"] = states.loc[
        locs, "cd_infectious_true_draws"
    ]

    # Kill people over icu_limit.
    rel_limit = params.loc[("health_system", "icu_limit_relative", None), "value"]
    abs_limit = rel_limit * len(states)
    need_icu_locs = states.index[states["needs_icu"]]
    if abs_limit < len(need_icu_locs):
        excess = int(len(need_icu_locs) - abs_limit)
        to_kill = np.random.choice(need_icu_locs, size=excess, replace=False)
        for to_change, new_val in COUNTDOWNS["cd_dead"]["changes"].items():
            states.loc[to_kill, to_change] = new_val
        states.loc[to_kill, "cd_dead"] = 0

    return states
