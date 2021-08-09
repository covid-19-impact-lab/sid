import itertools
from typing import Callable
from typing import Optional

import numpy as np
import pandas as pd


def prepare_susceptibility_factor(
    susceptibility_factor_model: Optional[Callable],
    initial_states: pd.DataFrame,
    params: pd.DataFrame,
    seed: itertools.count,
) -> np.ndarray:
    """Prepare the multiplier for infection probabilities.

    The multiplier defines individual susceptibility which can be used to let infection
    probabilities vary by age.

    If no multiplier is given, all individuals have the same susceptibility. Otherwise,
    a custom function generates multipliers for the infection probability for each
    individual.

    Args:
        susceptibility_factor_model (Optional[Callable]): The custom function
            which computes individual multipliers with states, parameters and a seed.
        initial_states (pandas.DataFrame): The initial states.
        params (pandas.DataFrame): The parameters.
        seed (itertools.count): The seed counter.

    Returns: susceptibility_factor (numpy.ndarray): An array with a
        multiplier for each individual between 0 and 1.

    """
    if susceptibility_factor_model is None:
        susceptibility_factor = np.ones(len(initial_states))
    else:
        susceptibility_factor = susceptibility_factor_model(
            initial_states, params, next(seed)
        )
        if not isinstance(susceptibility_factor, (pd.Series, np.ndarray)):
            raise ValueError(
                "'susceptibility_factor_model' must return a pd.Series or a np.ndarray."
            )
        elif len(susceptibility_factor) != len(initial_states):
            raise ValueError(
                "The 'susceptibility_factor' must be given for each individual."
            )
        elif isinstance(susceptibility_factor, pd.Series):
            susceptibility_factor = susceptibility_factor.to_numpy()

        # Make sure the highest multiplier is set to one so that random contacts only
        # need to be reduced by the infection probability of the contact model.
        susceptibility_factor = susceptibility_factor / susceptibility_factor.max()

        if (
            not (0 <= susceptibility_factor).all()
            and (susceptibility_factor <= 1).all()
        ):
            raise ValueError(
                "The susceptibility factors need to lie in the interval [0, 1]."
            )

    return susceptibility_factor


def update_susceptibility_factor(
    susceptibility_factor: np.ndarray,
    states: pd.DataFrame,
    params: pd.DataFrame,
    seed: itertools.count,
) -> np.ndarray:
    """Update susceptibility factor using state dependent information.

    Individual susceptibility is adjusted e.g. in the case of infection or vaccination.

    """
    np.random.seed(next(seed))

    low = params.loc[("susceptibility", "random_factor", "low"), "value"]
    high = params.loc[("susceptibility", "random_factor", "high"), "value"]
    random_factor = np.random.uniform(low, high, size=len(states))

    immunity_factor = 1 - states["immunity_level"]

    updated = susceptibility_factor * immunity_factor * random_factor
    updated = np.clip(updated, 0, 1)
    return updated
