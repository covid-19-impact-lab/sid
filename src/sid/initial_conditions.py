"""This module contains everything related to initial conditions.

The initial conditions can be used to create a more diverse pattern of disease processes
among individuals in the beginning of the simulation. This is achieved by assuming a
growth rate for infections over a period of time and advancing individuals health
statuses in time.

"""
import copy

import numba as nb
import numpy as np
import pandas as pd
from sid.contacts import boolean_choice
from sid.update_states import update_states


def scale_and_spread_initial_infections(
    states, initial_infections, params, initial_conditions, optional_state_columns, seed
):
    """Scale up and spread initial infections."""
    scaled_infections = _scale_up_initial_infections(
        initial_infections=initial_infections,
        states=states,
        params=params,
        assort_by=initial_conditions["assort_by"],
    )

    spread_out_infections = _spread_out_initial_infections(
        scaled_infections=scaled_infections,
        burn_in_periods=initial_conditions["burn_in_periods"],
        growth_rate=initial_conditions["growth_rate"],
    )

    # Number of contacts are only available during the simulation.
    optional_columns = copy.deepcopy(optional_state_columns)
    optional_columns["contacts"] = False

    for infections in spread_out_infections:
        states = update_states(
            states=states,
            newly_infected_contacts=infections,
            newly_infected_events=infections,
            params=params,
            seed=seed,
            optional_state_columns=optional_columns,
        )

    return states


def _scale_up_initial_infections(initial_infections, states, params, assort_by):
    r"""Increase number of infections by a multiplier taken from params.

    The probability for each individual to become infectious depends on the share of
    infected people in their assort by group, :math:`\mu` and the growth factor,
    :math:`r`.

    .. math::

        p_i = \frac{\mu * (r - 1)}{1 - \mu}

    The formula ensures that relative number of cases between groups defined by the
    variables in ``assort_by`` is preserved.

    Args:
        initial_infections (numpy.ndarray): Boolean array indicating initial infections.
        states (pandas.DataFrame): The states DataFrame.
        params (pandas.DataFrame): The parameters DataFrame.
        assort_by (List[str]): A list of ``assort_by`` variables.

    Returns:
        scaled_up (pandas.Series): A boolean series with upscaled infections.

    """
    multiplier = params.loc[("known_cases_multiplier",) * 3, "value"]
    states["known_infections"] = initial_infections
    share_infections = states.groupby(assort_by)["known_infections"].transform("mean")
    states.drop(columns="known_infections", inplace=True)

    prob_numerator = share_infections * (multiplier - 1)
    prob_denominator = 1 - share_infections
    prob = prob_numerator / prob_denominator

    scaled_up_arr = _scale_up_initial_infections_numba(
        initial_infections.to_numpy(), prob.to_numpy()
    )
    scaled_up = pd.Series(scaled_up_arr, index=states.index)
    return scaled_up


def _spread_out_initial_infections(scaled_infections, burn_in_periods, growth_rate):
    """Spread out initial infections over several periods, given a growth rate."""
    scaled_infections = scaled_infections.to_numpy()
    reversed_shares = []
    end_of_period_share = 1
    for _ in range(burn_in_periods):
        start_of_period_share = end_of_period_share / growth_rate
        added = end_of_period_share - start_of_period_share
        reversed_shares.append(added)
        end_of_period_share = start_of_period_share
    shares = reversed_shares[::-1]
    shares[-1] = 1 - np.sum([shares[:-1]])

    hypothetical_infection_day = np.random.choice(
        burn_in_periods, p=shares, replace=True, size=len(scaled_infections)
    )

    spread_infections = []
    for period in range(burn_in_periods):
        hypothetially_infected_on_that_day = hypothetical_infection_day == period
        infected_at_all = scaled_infections
        spread_infections.append(hypothetially_infected_on_that_day & infected_at_all)

    return spread_infections


@nb.jit
def _scale_up_initial_infections_numba(initial_infections, probabilities):
    """Scale up initial infections.

    Args:
        initial_infections (numpy.ndarray): Boolean array indicating initial infections.
        probabilities (numpy.ndarray): Probabilities for becoming infected.

    Returns:
        scaled_infections (numpy.ndarray): Upscaled infections.

    """
    n_obs = initial_infections.shape[0]
    scaled_infections = initial_infections.copy()
    for i in range(n_obs):
        if not scaled_infections[i]:
            scaled_infections[i] = boolean_choice(probabilities[i])
    return scaled_infections
