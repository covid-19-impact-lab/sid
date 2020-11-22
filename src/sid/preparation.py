import itertools as it

import numba as nb
import numpy as np
import pandas as pd
from sid.config import BOOLEAN_STATE_COLUMNS
from sid.config import DTYPE_COUNTDOWNS
from sid.config import DTYPE_INFECTION_COUNTER
from sid.config import INITIAL_CONDITIONS
from sid.contacts import boolean_choice
from sid.countdowns import COUNTDOWNS
from sid.pathogenesis import draw_course_of_disease
from sid.shared import process_optional_state_columns
from sid.update_states import update_states
from sid.validation import validate_initial_states_and_infections
from sid.validation import validate_params


def prepare_initial_states(
    initial_states,
    initial_infections,
    params,
    initial_conditions=None,
    optional_state_columns=None,
    seed=None,
):
    """Prepare the initial states.

    This functions prepares the initial states by drawing the course of the disease
    for every individual and spreading the disease as defined by the initial
    infections and initial conditions.

    The function must be called on the user-defined states before they are passed to
    the simulation and estimation routines. If the user received a states DataFrame
    from the simulation or estimation, this step can be skipped.

    Args:
        initial_states (pandas.DataFrame): See :ref:`states`. Cannot contain the column
            "date" because it is used internally.
        initial_infections (pandas.Series): A series with the same index as
            ``states`` which indicates individuals which are infected.
        params (pandas.DataFrame): DataFrame with parameters that influence the number
            of contacts, contagiousness and dangerousness of the disease, ... .
        initial_conditions (dict): Dict containing the entries "burn_in_period" (int),
            "assort_by" (list) and "growth_rate" (float). burn_in_periods and
            growth_rate are
            needed to spread out the initial infections over a period of time.
            "assort_by" specifies the aggregation level on which
            initial infections are scaled up to account for unknown cases.
        optional_state_columns (dict): Dictionary with categories of state columns
            that can additionally be added to the states dataframe, either for use in
            contact models and policies or to be saved. Most types of columns are added
            by default, but some of them are costly to add and thus only added when
            needed. Columns that are not in the state but specified in ``saved_columns``
            will not be saved. The categories are "contacts" and "reason_for_infection".
        seed (int): The seed which controls the randomness in courses of diseases and
            how infections progress.

    Returns:
        states (pandas.DataFrame): A DataFrame which can be used for simulating the
            spread of the disease in the population.

    """
    initial_conditions = (
        INITIAL_CONDITIONS
        if initial_conditions is None
        else {**INITIAL_CONDITIONS, **initial_conditions}
    )

    initial_states = initial_states.copy(deep=True)
    initial_infections = initial_infections.copy(deep=True)
    validate_initial_states_and_infections(initial_states, initial_infections)
    validate_params(params)

    optional_state_columns = process_optional_state_columns(optional_state_columns)
    # There is not contact information, yet.
    optional_state_columns["contacts"] = False

    seed = it.count(np.random.randint(0, 1_000_000)) if seed is None else it.count(seed)

    states = initialize_state_columns(initial_states)
    states = draw_course_of_disease(states, params, seed)

    scaled_infections = _scale_up_initial_infections(
        initial_infections=initial_infections,
        states=states,
        params=params,
        assort_by=initial_conditions["assort_by"],
    )
    spreading_infections = _spread_out_initial_infections(
        scaled_infections=scaled_infections,
        burn_in_periods=initial_conditions["burn_in_periods"],
        growth_rate=initial_conditions["growth_rate"],
    )

    for infections in spreading_infections:
        states = update_states(
            states=states,
            newly_infected_contacts=infections,
            newly_infected_events=infections,
            params=params,
            seed=seed,
            optional_state_columns=optional_state_columns,
        )

    return states


def initialize_state_columns(states):
    for col in BOOLEAN_STATE_COLUMNS:
        if col not in states.columns:
            states[col] = False

    for col in COUNTDOWNS:
        if col not in states.columns:
            states[col] = -1
        states[col] = states[col].astype(DTYPE_COUNTDOWNS)

    states["n_has_infected"] = DTYPE_INFECTION_COUNTER(0)
    states["pending_test_date"] = pd.NaT

    return states


def _scale_up_initial_infections(initial_infections, states, params, assort_by):
    """Increase number of infections by a multiplier taken from params.

    The relative number of cases between groups defined by the variables in
    ``assort_by`` is preserved.

    """
    states["known_infections"] = initial_infections
    average_infections = states.groupby(assort_by)["known_infections"].transform("mean")
    states = states.drop(columns="known_infections")

    multiplier = params.loc[("known_cases_multiplier",) * 3, "value"]
    prob_numerator = average_infections * (multiplier - 1)
    prob_denominator = 1 - average_infections
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
        hypothetically_infected_on_that_day = hypothetical_infection_day == period
        infected_at_all = scaled_infections
        spread_infections.append(hypothetically_infected_on_that_day & infected_at_all)

    return spread_infections


@nb.jit
def _scale_up_initial_infections_numba(initial_infections, probabilities):
    n_obs = initial_infections.shape[0]
    res = initial_infections.copy()
    for i in range(n_obs):
        if not res[i]:
            res[i] = boolean_choice(probabilities[i])
    return res
