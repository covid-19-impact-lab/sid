"""This module contains the code related to seasonality."""
import itertools
from typing import Callable
from typing import Optional

import numpy as np
import pandas as pd


def prepare_seasonality_factor(
    seasonality_factor_model: Optional[Callable],
    params: pd.DataFrame,
    dates: pd.DatetimeIndex,
    seed: itertools.count,
) -> np.ndarray:
    """Prepare the seasonality factor.

    Args:
        seasonality_factor_model (Optional[Callable]): The function which calculates the
            seasonality factor for each day.
        params (pd.DataFrame): The parameters.
        dates (pd.DatetimeIndex): All simulated days.
        seed (itertools.count): A seed counter

    Returns:
        seasonality_factor (np.ndarray): Factors for the infection probabilities of each
            day.

    """
    if seasonality_factor_model is None:
        seasonality_factor = pd.Series(index=dates, data=1)
    else:
        seasonality_factor = seasonality_factor_model(
            params=params, dates=dates, seed=next(seed)
        )

        if not isinstance(seasonality_factor, (pd.Series, np.ndarray)):
            raise ValueError(
                "'seasonality_factor_model' must return a pd.Series or a np.ndarray."
            )
        elif len(seasonality_factor) != len(dates):
            raise ValueError(
                "The 'seasonality_factor' must be given for each individual."
            )
        elif isinstance(seasonality_factor, pd.Series):
            seasonality_factor = seasonality_factor.to_numpy()

        # Make sure the highest multiplier is set to one so that random contacts only
        # need to be reduced by the infection probability of the contact model.
        seasonality_factor = seasonality_factor / seasonality_factor.max()

        if not (0 <= seasonality_factor).all() and (seasonality_factor <= 1).all():
            raise ValueError(
                "The seasonality factors need to lie in the interval [0, 1]."
            )

    return seasonality_factor
