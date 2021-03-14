"""This module contains the code for vaccinating individuals."""
import itertools
from typing import Callable
from typing import Optional

import pandas as pd
from sid.validation import validate_return_is_series_or_ndarray


def vaccinate_individuals(
    vaccination_model: Optional[Callable],
    states: pd.DataFrame,
    params: pd.DataFrame,
    seed: itertools.count,
) -> pd.Series:
    """Vaccinate individuals.

    Args:
        vaccination_model (Optional[Callable]): A function accepting ``states``,
            ``params``, and a ``seed`` which returns boolean indicators for individuals
            who received a vaccination.
        states (pandas.DataFrame): The states.
        params (pandas.DataFrame): The params.
        seed (itertools.count): The seed counter.

    """
    if vaccination_model is None:
        newly_vaccinated = None
    else:
        newly_vaccinated = vaccination_model(states, params, next(seed))
        newly_vaccinated = validate_return_is_series_or_ndarray(
            newly_vaccinated, index=states.index, when="vaccination_model"
        )

    return newly_vaccinated
