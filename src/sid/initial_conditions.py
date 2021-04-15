"""This module contains everything related to initial conditions.

The initial conditions govern the distribution of infections and immunity in the
beginning of a simulation and can used to create patterns which match the real data.

"""
import itertools
import math
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numba as nb
import numpy as np
import pandas as pd
from sid.config import DTYPE_VIRUS_STRAIN
from sid.contacts import boolean_choice
from sid.testing import perform_testing
from sid.time import timestamp_to_sid_period
from sid.update_states import update_derived_state_variables
from sid.update_states import update_states
from sid.vaccination import vaccinate_individuals


def sample_initial_infections(
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


def sample_initial_immunity(
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


def sample_initial_distribution_of_infections_and_immunity(
    states: pd.DataFrame,
    params: pd.DataFrame,
    initial_conditions: Dict[str, Any],
    testing_demand_models: Dict[str, Dict[str, Any]],
    testing_allocation_models: Dict[str, Dict[str, Any]],
    testing_processing_models: Dict[str, Dict[str, Any]],
    virus_strains: Dict[str, Any],
    vaccination_models: Optional[Callable],
    seed: itertools.count,
    derived_state_variables: Dict[str, str],
):
    """Sample the initial distribution of infections and immunity.

    This functions allows to

    - set the number of initial infections.
    - increase the number of infections by some factor to reduce underreporting, for
      example, due to asymptomatic cases. You can also keep shares between subgroups
      constant.
    - let infections evolve over some periods to have courses of diseases in every
      stage.
    - assume pre-existing immunity in the population.

    Args:
        states (pandas.DataFrame): The states.
        params (pandas.DataFrame): The parameters.
        initial_conditions (Dict[str, Any]): The initial conditions allow you to govern
            the distribution of infections and immunity and the heterogeneity of courses
            of disease at the start of the simulation. Use ``None`` to assume no
            heterogeneous courses of diseases and 1% infections. Otherwise,
            ``initial_conditions`` is a dictionary containing the following entries:

            - ``assort_by`` (Optional[Union[str, List[str]]]): The relative infections
              is preserved between the groups formed by ``assort_by`` variables. By
              default, no group is formed and infections spread across the whole
              population.
            - ``burn_in_periods`` (int): The number of periods over which infections are
              distributed and can progress. The default is one period.
            - ``growth_rate`` (float): The growth rate specifies the increase of
              infections from one burn-in period to the next. For example, two indicates
              doubling case numbers every period. The value must be greater than or
              equal to one. Default is one which is no distribution over time.
            - ``initial_immunity`` (Union[int, float, pandas.Series]): The n_people who
              are immune in the beginning can be specified as an integer for the number,
              a float between 0 and 1 for the share, and a :class:`pandas.Series` with
              the same index as states. Note that infected individuals are also immune.
              For a 10% pre-existing immunity with 2% currently infected people, set the
              key to 0.12. By default, only infected individuals indicated by the
              initial infections are immune.
            - ``initial_infections`` (Union[int, float, pandas.Series,
              pandas.DataFrame]): The initial infections can be given as an integer
              which is the number of randomly infected individuals, as a float for the
              share or as a :class:`pandas.Series` which indicates whether an
              individuals is infected. If initial infections are a
              :class:`pandas.DataFrame`, then, the index is the same as ``states``,
              columns are dates or periods which can be sorted, and values are infected
              individuals on that date. This step will skip upscaling and distributing
              infections over days and directly jump to the evolution of states. By
              default, 1% of individuals is infected.
            - ``known_cases_multiplier`` (int): The factor can be used to scale up the
              initial infections while keeping shares between ``assort_by`` variables
              constant. This is helpful if official numbers are underreporting the
              number of cases.
            - ``virus_shares`` (Union[dict, pandas.Series]): A mapping between the names
              of the virus strains and their share among newly infected individuals in
              each burn-in period.
        virus_strains (Dict[str, Any]): A dictionary with the keys ``"names"`` and
            ``"factors"`` holding the different contagiousness factors of multiple
            viruses.
        vaccination_models (Optional[Dict[str, Dict[str, Any]): A dictionary of models
            which allow to vaccinate individuals. The ``"model"`` key holds a function
            with arguments ``states``, ``params``, and a ``seed`` which returns boolean
            indicators for individuals who received a vaccination.
        seed (itertools.count): The seed counter.
        derived_state_variables (Dict[str, str]): A dictionary that maps
            names of state variables to pandas evaluation strings that generate derived
            state variables, i.e. state variables that can be calculated from the
            existing state variables.


    Returns:
        states (pandas.DataFrame): The states with sampled infections and immunity.

    """
    initial_infections = initial_conditions["initial_infections"]
    if isinstance(initial_infections, (int, float, pd.Series)):
        if isinstance(initial_infections, (float, int)):
            initial_infections = sample_initial_infections(
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

        spread_out_virus_strains = _sample_factorized_virus_strains_for_infections(
            spread_out_infections,
            initial_conditions["virus_shares"],
        )

    else:
        spread_out_virus_strains = initial_conditions["initial_infections"]

    # this is necessary to make derived state variables usable in testing models
    states = update_derived_state_variables(states, derived_state_variables)

    for burn_in_date in initial_conditions["burn_in_periods"]:

        states["date"] = burn_in_date
        states["period"] = timestamp_to_sid_period(burn_in_date)

        states, _, to_be_processed_tests = perform_testing(
            date=burn_in_date,
            states=states,
            params=params,
            testing_demand_models=testing_demand_models,
            testing_allocation_models=testing_allocation_models,
            testing_processing_models=testing_processing_models,
            seed=seed,
        )

        newly_vaccinated = vaccinate_individuals(
            burn_in_date, vaccination_models, states, params, seed
        )

        states = update_states(
            states=states,
            newly_infected_contacts=spread_out_virus_strains[burn_in_date],
            newly_infected_events=spread_out_virus_strains[burn_in_date],
            params=params,
            to_be_processed_tests=to_be_processed_tests,
            virus_strains=virus_strains,
            newly_vaccinated=newly_vaccinated,
            seed=seed,
            derived_state_variables=derived_state_variables,
        )

    # Remove date information because when it is available, we assume the simulation is
    # resumed.
    states = states.drop(columns=["date", "period"])

    initial_immunity = sample_initial_immunity(
        initial_conditions["initial_immunity"], states["immune"], next(seed)
    )
    states = _integrate_immune_individuals(states, initial_immunity)

    return states


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


@nb.njit  # pragma: no cover
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
    scaled_infections: pd.Series,
    burn_in_periods: List[pd.Timestamp],
    growth_rate: float,
    seed: int,
) -> pd.DataFrame:
    """Spread out initial infections over several periods, given a growth rate.

    Args:
        scaled_infections (pandas.Series): The scaled infections.
        burn_in_periods (List[pd.Timestamp]): Number of burn-in periods.
        growth_rate (float): The growth rate of infections from one burn-in period to
            the next.
        seed (itertools.count): The seed counter.

    Return:
        spread_infections (pandas.DataFrame): A list of boolean arrays which indicate
            new infections for each day of the burn-in period.

    """
    np.random.seed(seed)

    n_burn_in_periods = len(burn_in_periods)

    if n_burn_in_periods > 1:
        scaled_infections = scaled_infections.to_numpy()
        if growth_rate == 1:
            shares = np.array([1] + [0] * (n_burn_in_periods - 1))
        else:
            shares = -np.diff(1 / growth_rate ** np.arange(n_burn_in_periods + 1))[::-1]
            shares = shares / shares.sum()
        hypothetical_infection_day = np.random.choice(
            n_burn_in_periods, p=shares, replace=True, size=len(scaled_infections)
        )
    else:
        hypothetical_infection_day = np.zeros(len(scaled_infections))

    spread_infections = pd.concat(
        [
            pd.Series(
                data=(hypothetical_infection_day == period) & scaled_infections,
                name=burn_in_periods[period],
            )
            for period in range(n_burn_in_periods)
        ],
        axis=1,
    )

    return spread_infections


def _sample_factorized_virus_strains_for_infections(
    spread_out_infections: pd.DataFrame,
    virus_shares: Dict[str, Any],
) -> pd.DataFrame:
    """Convert boolean infections to factorized virus strains."""
    spread_out_virus_strains = pd.DataFrame(
        data=-1,
        index=spread_out_infections.index,
        columns=spread_out_infections.columns,
        dtype=DTYPE_VIRUS_STRAIN,
    )

    virus_strain_factors = list(range(len(virus_shares)))
    probabilities = list(virus_shares.values())

    for column in spread_out_infections.columns:
        n_infected = spread_out_infections[column].sum()
        if 1 <= n_infected:
            sampled_virus_strains = np.random.choice(
                virus_strain_factors, p=probabilities, size=n_infected
            )
            spread_out_virus_strains.loc[
                spread_out_infections[column], column
            ] = sampled_virus_strains

    return spread_out_virus_strains


def _integrate_immune_individuals(
    states: pd.DataFrame, initial_immunity: pd.Series
) -> pd.DataFrame:
    """Integrate immune individuals in states.

    Args:
        states (pandas.DataFrame): The states which already include sampled infections.
        initial_immunity (pandas.Series): A series with sampled immune individuals.

    Returns:
        states (pandas.DataFrame): The states with additional immune individuals.

    """
    extra_immune = initial_immunity & ~states["immune"]
    states.loc[extra_immune, "immune"] = True
    states.loc[extra_immune, "ever_infected"] = True
    states.loc[extra_immune, "cd_ever_infected"] = 0
    states.loc[extra_immune, "cd_immune_false"] = states.loc[
        extra_immune, "cd_immune_false_draws"
    ]

    return states
