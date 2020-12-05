"""This module contains everything related to initial conditions.

The initial conditions can be used to create a more diverse pattern of disease processes
among individuals in the beginning of the simulation. This is achieved by assuming a
growth rate for infections over a period of time and advancing individuals health
statuses in time.

"""
import math
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import numba as nb
import numpy as np
import pandas as pd
from sid.config import INITIAL_CONDITIONS
from sid.contacts import boolean_choice
from sid.update_states import update_states


def scale_and_spread_initial_infections(states, params, initial_conditions, seed):
    """Scale up and spread initial infections."""
    initial_conditions = _parse_initial_conditions(initial_conditions)
    validate_initial_conditions(initial_conditions)

    initial_infections = initial_conditions["initial_infections"]
    if not isinstance(initial_infections, pd.DataFrame):
        if isinstance(initial_infections, pd.Series):
            pass
        elif isinstance(initial_infections, (float, int)):
            initial_infections = create_initial_infections(
                initial_infections, index=states.index, seed=next(seed)
            )

        scaled_infections = _scale_up_initial_infections(
            initial_infections=initial_infections,
            states=states,
            assort_by=initial_conditions["assort_by"],
            known_cases_multiplier=initial_conditions["known_cases_multiplier"],
            seed=next(seed),
        )

        spread_out_infections = _spread_out_initial_infections(
            scaled_infections=scaled_infections,
            burn_in_periods=initial_conditions["burn_in_periods"],
            growth_rate=initial_conditions["growth_rate"],
            seed=next(seed),
        )

    else:
        spread_out_infections = initial_conditions["initial_infections"]

    for _, infections in spread_out_infections.sort_index(axis=1).items():
        states = update_states(
            states=states,
            newly_infected_contacts=infections,
            newly_infected_events=infections,
            params=params,
            seed=seed,
        )

    initial_immunity = create_initial_immunity(
        initial_conditions["initial_immunity"], states["immune"], next(seed)
    )
    states = _integrate_immune_individuals(states, initial_immunity)

    return states


def _parse_initial_conditions(ic):
    """Parse the initial conditions."""
    ic = INITIAL_CONDITIONS if ic in [None, False] else {**INITIAL_CONDITIONS, **ic}

    if isinstance(ic["assort_by"], str):
        ic["assort_by"] = [ic["assort_by"]]

    return ic


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
        seed (int): The seed.

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
        initial_infections.to_numpy(), prob.to_numpy(), seed
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
    scaled_infections: pd.Series, burn_in_periods: int, growth_rate: float, seed: int
) -> pd.DataFrame:
    """Spread out initial infections over several periods, given a growth rate.

    Args:
        scaled_infections (pandas.Series): The scaled infections.
        burn_in_periods (int): Number of burn-in periods.
        growth_rate (float): The growth rate of infections from one burn-in period to
            the next.
        seed (itertools.count): The seed counter.

    Return:
        spread_infections (pandas.DataFrame): A list of boolean arrays which indicate
            new infections for each day of the burn-in period.

    """
    np.random.seed(seed)

    if burn_in_periods > 1:
        scaled_infections = scaled_infections.to_numpy()
        if growth_rate == 1:
            shares = np.array([1] + [0] * (burn_in_periods - 1))
        else:
            shares = -np.diff(1 / growth_rate ** np.arange(burn_in_periods + 1))[::-1]
            shares = shares / shares.sum()
        hypothetical_infection_day = np.random.choice(
            burn_in_periods, p=shares, replace=True, size=len(scaled_infections)
        )
    else:
        hypothetical_infection_day = np.zeros(len(scaled_infections))

    spread_infections = pd.concat(
        [
            pd.Series((hypothetical_infection_day == period) & scaled_infections)
            for period in range(burn_in_periods)
        ],
        axis=1,
    )

    return spread_infections


def create_initial_infections(
    infections: Union[int, float],
    n_people: Optional[int] = None,
    index: Optional[pd.Index] = None,
    seed: Optional[int] = None,
) -> pd.Series:
    """Create a :class:`pandas.Series` indicating infected individuals.

    Args:
        infections (Union[int, float]): The infections can be either a
            :class:`pandas.Series` where each individual has an indicator for the
            infection status, an integer representing the number of infected people or a
            float representing the share of infected individuals.
        n_people (Optional[int]): The number of individuals.
        index (Optional[pandas.Index]): The index for the infections.
        seed (Optional[int]): A seed.

    Returns:
        infections (pandas.Series): A series indicating infected individuals.

    """
    seed = np.random.randint(0, 1_000_000) if seed is None else seed
    np.random.seed(seed)

    if n_people is None and index is None:
        raise ValueError("Either 'n_people' or 'index' has to be provided.")
    elif n_people is None and index is not None:
        n_people = len(index)
    elif index is not None and n_people != len(index):
        raise ValueError("'n_people' has to match the length of 'index'.")

    if isinstance(infections, int):
        pass
    elif isinstance(infections, float) and 0 <= infections < 1:
        infections = math.ceil(n_people * infections)
    else:
        raise ValueError("'infections' must be an int or a float between 0 and 1.")

    index = pd.RangeIndex(n_people) if index is None else index
    infected_indices = np.random.choice(n_people, size=infections, replace=False)
    infections = pd.Series(index=index, data=False)
    infections.iloc[infected_indices] = True

    return infections


def create_initial_immunity(
    immunity: Union[int, float, pd.Series, None],
    infected_or_immune: pd.Series,
    seed: Optional[int],
) -> pd.Series:
    """Create indicator for initially immune people.

    There are some special cases to handle:

    1. Infected individuals are always treated as being immune and reduce the number of
       additional immune individuals.
    2. If immunity is given as an integer or float, additional immune individuals are
       sampled randomly.
    3. If immunity is given as a series, immune and infected individuals form the total
       immune population.

    Args:
        immunity (Union[int, float, pandas.Series]): The people who are immune in the
            beginning can be specified as an integer for the number, a float between 0
            and 1 for the share, and a :class:`pandas.Series` with the same index as
            states. Note that, infected individuals are immune and included.
        infected_or_immune (pandas.Series): A series which indicates either immune or
            infected individuals.

    Returns:
        initial_immunity (pandas.Series): Indicates immune individuals.

    """
    seed = np.random.randint(0, 1_000_000) if seed is None else seed
    np.random.seed(seed)

    immunity = 0 if immunity is None else immunity

    n_people = len(infected_or_immune)

    if isinstance(immunity, float):
        immunity = math.ceil(n_people * immunity)

    initial_immunity = infected_or_immune.copy()
    if isinstance(immunity, int):
        n_immune = initial_immunity.sum()
        n_additional_immune = immunity - n_immune
        if 0 < n_additional_immune <= n_people - n_immune:
            choices = np.arange(len(initial_immunity))[~initial_immunity]
            ilocs = np.random.choice(choices, size=n_additional_immune, replace=False)
            initial_immunity.iloc[ilocs] = True

    elif isinstance(immunity, pd.Series):
        initial_immunity = initial_immunity | immunity
    else:
        raise ValueError("'initial_immunity' must be an int, float or pd.Series.")

    return initial_immunity


def validate_initial_conditions(initial_conditions: Dict[str, Any]) -> None:
    initial_infections = initial_conditions["initial_infections"]
    if not (
        isinstance(initial_infections, (pd.DataFrame, pd.Series))
        or (isinstance(initial_infections, int) and initial_infections >= 0)
        or (isinstance(initial_infections, float) and 0 <= initial_infections <= 1)
    ):
        raise ValueError(
            "'initial_infections' must be a pd.DataFrame, pd.Series, int or float "
            "between 0 and 1."
        )

    if not initial_conditions["growth_rate"] >= 1:
        raise ValueError("'growth_rate' must be greater than or equal to 1.")

    burn_in_periods = initial_conditions["burn_in_periods"]
    if not (isinstance(burn_in_periods, int) and burn_in_periods >= 1):
        raise ValueError(
            "'burn_in_periods' must be an integer which is greater than or equal to 1."
        )


def _integrate_immune_individuals(
    states: pd.DataFrame, initial_immunity: pd.Series
) -> pd.DataFrame:
    """Integrate immune individuals in states."""
    extra_immune = initial_immunity & ~states["immune"]
    states.loc[extra_immune, "immune"] = True
    states.loc[extra_immune, "ever_infected"] = True
    states.loc[extra_immune, "cd_ever_infected"] = 0
    states.loc[extra_immune, "cd_immune_false"] = states.loc[
        extra_immune, "cd_immune_false_draws"
    ]

    return states
