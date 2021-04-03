"""This module contains the code for vaccinating individuals."""
import itertools
from typing import Any
from typing import Dict
from typing import Optional

import pandas as pd
from sid.validation import validate_return_is_series_or_ndarray


def vaccinate_individuals(
    date: pd.Timestamp,
    vaccination_models: Optional[Dict[str, Dict[str, Any]]],
    states: pd.DataFrame,
    params: pd.DataFrame,
    seed: itertools.count,
) -> pd.Series:
    """Vaccinate individuals.

    Args:
        date (pandas.Timestamp): The current date.
        vaccination_models (Optional[Dict[str, Dict[str, Any]): A dictionary of models
            which allow to vaccinate individuals. The ``"model"`` key holds a function
            with arguments ``states``, ``params``, and a ``seed`` which returns boolean
            indicators for individuals who received a vaccination.
        states (pandas.DataFrame): The states.
        params (pandas.DataFrame): The params.
        seed (itertools.count): The seed counter.

    """
    newly_vaccinated = pd.Series(index=states.index, data=False)

    for name, model in vaccination_models.items():
        loc = model.get("loc", params.index)
        func = model["model"]

        if model["start"] <= date <= model["end"]:
            new_newly_vaccinated = func(
                receives_vaccine=newly_vaccinated.copy(deep=True),
                states=states,
                params=params.loc[loc],
                seed=next(seed),
            )

            new_newly_vaccinated = validate_return_is_series_or_ndarray(
                new_newly_vaccinated, name, "vaccination_models", states.index
            )

            newly_vaccinated.loc[new_newly_vaccinated] = True

    return newly_vaccinated
