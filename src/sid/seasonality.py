"""This module contains the code related to seasonality."""
import itertools
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd


def prepare_seasonality_factor(
    seasonality_factor_model: Optional[Callable],
    params: pd.DataFrame,
    dates: pd.DatetimeIndex,
    seed: itertools.count,
    contact_models: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    """Prepare the seasonality factor.

    Args:
        seasonality_factor_model (Optional[Callable]): The function which calculates the
            seasonality factor for each day.
        params (pd.DataFrame): The parameters.
        dates (pd.DatetimeIndex): All simulated days.
        seed (itertools.count): A seed counter
        contact_models (Dict[str, Dict[str, Any]]): See :ref:`contact_models`.

    Returns:
        seasonality_factor (np.ndarray): Factors for the infection probabilities of each
            day.

    """
    if seasonality_factor_model is None:
        factor = pd.DataFrame(index=dates, columns=contact_models, data=1)
    else:
        raw_factor = seasonality_factor_model(
            params=params, dates=dates, seed=next(seed)
        )

        if isinstance(raw_factor, pd.Series):
            raw_factor = raw_factor.reindex(dates)
            factor = pd.concat([raw_factor] * len(contact_models), axis=1)
            factor.columns = contact_models.keys()
        elif isinstance(raw_factor, pd.DataFrame):
            factor = pd.DataFrame(index=dates, columns=contact_models, data=1)
            factor.update(raw_factor)
        else:
            raise ValueError(
                "'seasonality_factor_model' must return a pd.Series or DataFrame "
                "with 'dates' as index and seasonality factors as data."
            )

    factor = factor.astype(float)

    for col in factor:
        if not factor[col].between(0, 1).all():
            raise ValueError(
                "The seasonality factors need to lie in the interval [0, 1]."
            )

    return factor
