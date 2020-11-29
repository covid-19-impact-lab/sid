"""This module contains everything related to initial conditions.

The initial conditions can be used to create a more diverse pattern of disease processes
among individuals in the beginning of the simulation. This is achieved by assuming a
growth rate for infections over a period of time and advancing individuals health
statuses in time.

"""
import numba as nb
import numpy as np
import pandas as pd
from sid.config import INITIAL_CONDITIONS
from sid.contacts import boolean_choice
from sid.update_states import update_states


def scale_and_spread_initial_infections(
    states, initial_infections, params, initial_conditions, seed
):
    """Scale up and spread initial infections."""
    initial_conditions = _parse_initial_conditions(initial_conditions)

    scaled_infections = _scale_up_initial_infections(
        initial_infections=initial_infections,
        states=states,
        assort_by=initial_conditions["assort_by"],
        known_cases_multiplier=initial_conditions["known_cases_multiplier"],
        seed=seed,
    )

    spread_out_infections = _spread_out_initial_infections(
        scaled_infections=scaled_infections,
        burn_in_periods=initial_conditions["burn_in_periods"],
        growth_rate=initial_conditions["growth_rate"],
        seed=seed,
    )

    for infections in spread_out_infections:
        states = update_states(
            states=states,
            newly_infected_contacts=infections,
            newly_infected_events=infections,
            params=params,
            seed=seed,
        )

    return states


def _parse_initial_conditions(initial_conditions):
    """Parse the initial conditions."""
    initial_conditions = (
        INITIAL_CONDITIONS
        if initial_conditions in [None, False]
        else {**INITIAL_CONDITIONS, **initial_conditions}
    )

    if isinstance(initial_conditions["assort_by"], str):
        initial_conditions["assort_by"] = [initial_conditions["assort_by"]]

    return initial_conditions


def _scale_up_initial_infections(
    initial_infections, states, assort_by, known_cases_multiplier, seed
):
    r"""Increase number of infections by a multiplier taken from params.

    If no ``assort_by`` variables are provided, infections are simply scaled up in the
    whole population without regarding any inter-group differences.

    If ``assort_by`` variables are passed to the function, the probability for each
    individual to become infectious depends on the share of infected people in their
    group, :math:`\mu` and the growth factor, :math:`r`.

    .. math::

        p_i = \frac{\mu * (r - 1)}{1 - \mu}

    The formula ensures that relative number of cases between groups defined by the
    variables in ``assort_by`` is preserved.

    Args:
        initial_infections (pandas.Series): Boolean array indicating initial infections.
        states (pandas.DataFrame): The states DataFrame.
        assort_by (Optional[List[str]]): A list of ``assort_by`` variables if infections
            should be proportional between groups or ``None`` if no groups are used.
        known_cases_multiplier (float): The multiplier which can be used to scale
            infections from observed infections to the real number of infections.
        seed (itertools.count): A seed counter.

    Returns:
        scaled_up (pandas.Series): A boolean series with upscaled infections.

    """
    states["known_infections"] = initial_infections

    if assort_by is None:
        share_infections = pd.Series(
            index=states.index, data=states["known_infections"].mean()
        )
    else:
        share_infections = states.groupby(assort_by)["known_infections"].transform(
            "mean"
        )
    states.drop(columns="known_infections", inplace=True)

    prob_numerator = share_infections * (known_cases_multiplier - 1)
    prob_denominator = 1 - share_infections
    prob = prob_numerator / prob_denominator

    scaled_up_arr = _scale_up_initial_infections_numba(
        initial_infections.to_numpy(), prob.to_numpy(), next(seed)
    )
    scaled_up = pd.Series(scaled_up_arr, index=states.index)
    return scaled_up


@nb.njit
def _scale_up_initial_infections_numba(initial_infections, probabilities, seed):
    """Scale up initial infections.

    Args:
        initial_infections (numpy.ndarray): Boolean array indicating initial infections.
        probabilities (numpy.ndarray): Probabilities for becoming infected.
        seed (int): Seed to control randomness.

    Returns:
        scaled_infections (numpy.ndarray): Upscaled infections.

    """
    np.random.seed(seed)
    n_obs = initial_infections.shape[0]
    scaled_infections = initial_infections.copy()
    for i in range(n_obs):
        if not scaled_infections[i]:
            scaled_infections[i] = boolean_choice(probabilities[i])
    return scaled_infections


def _spread_out_initial_infections(
    scaled_infections, burn_in_periods, growth_rate, seed
):
    """Spread out initial infections over several periods, given a growth rate.

    Args:
        scaled_infections (pandas.Series):
        burn_in_periods (int): Number of burn-in periods.
        growth_rate (float): Growth rate.
        seed (itertools.count): The seed counter.

    Return:
        spread_infections (List[ArrayLike[bool]]): A list of boolean arrays which
            indicate new infections for each day of the burn-in period.

    """
    np.random.seed(next(seed))

    if burn_in_periods > 1:
        scaled_infections = scaled_infections.to_numpy()
        shares = -np.diff(1 / growth_rate ** np.arange(burn_in_periods + 1))[::-1]
        shares = shares / shares.sum()
        hypothetical_infection_day = np.random.choice(
            burn_in_periods, p=shares, replace=True, size=len(scaled_infections)
        )
    else:
        hypothetical_infection_day = np.zeros(len(scaled_infections))

    spread_infections = [
        (hypothetical_infection_day == period) & scaled_infections
        for period in range(burn_in_periods)
    ]

    return spread_infections
