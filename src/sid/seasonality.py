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

        if isinstance(seasonality_factor, pd.Series):
            seasonality_factor = seasonality_factor.reindex(dates)
        elif isinstance(seasonality_factor, np.ndarray):
            seasonality_factor = pd.Series(index=dates, data=seasonality_factor)
        else:
            raise ValueError(
                "'seasonality_factor_model' must return a pd.Series or a np.ndarray "
                "with 'dates' as index and seasonality factors as data."
            )

        # Make sure the highest multiplier is set to one so that random contacts only
        # need to be reduced by the infection probability of the contact model.
        seasonality_factor = seasonality_factor / seasonality_factor.max()

        if not seasonality_factor.between(0, 1).all():
            raise ValueError(
                "The seasonality factors need to lie in the interval [0, 1]."
            )

    return seasonality_factor
