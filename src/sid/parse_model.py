"""This module contains the code the parse input data."""
import warnings
from collections.abc import Iterable
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from sid.config import INITIAL_CONDITIONS
from sid.virus_strains import factorize_multiple_boolean_or_categorical_infections


def parse_duration(duration: Union[Dict[str, Any], None]) -> Dict[str, Any]:
    """Parse the user-defined duration.

    Args:
        duration (Union[Dict[str, Any], None]): A dictionary which contains keys and
            values suited to be passed to :func:`pandas.date_range`. Only the first
            three arguments, ``"start"``, ``"end"``, and ``"periods"``, are allowed.

    Returns:
        internal_duration (Dict[str, Any]): A dictionary containing start and end dates
            and dates for the whole period.

    Examples:
        >>> parse_duration({"start": "2020-03-01", "end": "2020-03-10"})
        {'start': Timestamp('2020-03-01 00:00:00', freq='D'), 'end': ...

    """
    if duration is None:
        duration = {"start": "2020-01-27", "periods": 10}
    else:
        allowed_args = ("start", "end", "periods")
        not_allowed_args = set(duration) - set(allowed_args)
        if not_allowed_args:
            warnings.warn(
                "Only 'start', 'end', and 'periods' are admissible keys for 'duration'."
            )
            duration = {k: v for k, v in duration.items() if k in allowed_args}

    iterable = pd.date_range(**duration)

    internal_duration = {}
    internal_duration["start"] = iterable[0]
    internal_duration["end"] = iterable[-1]
    internal_duration["dates"] = iterable

    return internal_duration


def parse_initial_conditions(
    ic: Dict[str, Any],
    start_date_simulation: pd.Timestamp,
    virus_strains: Dict[str, Any],
) -> Dict[str, Any]:
    """Parse the initial conditions."""
    ic = {**INITIAL_CONDITIONS} if ic is None else {**INITIAL_CONDITIONS, **ic}

    if isinstance(ic["assort_by"], str):
        ic["assort_by"] = [ic["assort_by"]]

    if isinstance(ic["initial_infections"], pd.DataFrame):
        try:
            ic["initial_infections"].columns = pd.to_datetime(
                ic["initial_infections"].columns
            )
        except ValueError as e:
            raise ValueError(
                "The columns of 'initial_infections' must be convertible by "
                "pd.to_datetime."
            ) from e
        else:
            ic["burn_in_periods"] = ic["initial_infections"].columns.sort_values()

        ic["initial_infections"] = factorize_multiple_boolean_or_categorical_infections(
            ic["initial_infections"], virus_strains
        )

    if isinstance(ic["burn_in_periods"], int):
        if ic["burn_in_periods"] == 0:
            raise ValueError("'burn_in_periods' must be greater or equal than 1.")
        start = start_date_simulation - pd.Timedelta(ic["burn_in_periods"], unit="d")
        ic["burn_in_periods"] = pd.date_range(start, start_date_simulation)[:-1]

    elif isinstance(ic["burn_in_periods"], Iterable):
        n_burn_in_periods = len(ic["burn_in_periods"])
        start = start_date_simulation - pd.Timedelta(n_burn_in_periods, unit="d")
        expected = pd.date_range(start, start_date_simulation)[:-1]
        if not (ic["burn_in_periods"] == expected).all():
            raise ValueError(
                f"Expected 'burn_in_periods' {expected}, but got "
                f"{ic['burn_in_periods']} instead. This might happen because the "
                "pd.Dataframe passed as 'initial_infections' does not have dates as "
                "strings or pd.Timestamps for column names."
            )
    else:
        raise ValueError(
            f"'burn_in_periods' must be an integer or an iterable which is convertible "
            f"with pd.to_datetime, but got {ic['burn_in_periods']} instead."
        )

    if ic["virus_shares"] is None:
        ic["virus_shares"] = {
            name: 1 / len(virus_strains["names"]) for name in virus_strains["names"]
        }
    elif isinstance(ic["virus_shares"], (dict, pd.Series)):
        ic["virus_share"] = {
            name: ic["virus_shares"][name] for name in virus_strains["names"]
        }
    else:
        raise ValueError("'virus_shares' must be a dict or a pd.Series.")

    return ic


def parse_virus_strains(virus_strains: Optional[List[str]], params: pd.DataFrame):
    """Parse the information of the different infectiousness for each virus strain.

    The multipliers are scaled between 0 and 1 such that random contacts only need to be
    reduced with the infection probabilities in
    :func:`sid.contacts._reduce_random_contacts_with_infection_probs`.

    Args:
        virus_strains (Optional[List[str]]): A list of names indicating the different
            virus strains used in the model. Their different infectiousness is looked up
            in the params DataFrame. By default, only one virus strain is used.
        params (pandas.DataFrame): The params DataFrame.

    Returns:
        virus_strains (Dict[str, Any]): A dictionary with two keys.

        - ``"names"`` holds the sorted names of the virus strains.
        - ``"multipliers"`` holds the multipliers for the contagiousness of the viruses
          scaled between 0 and 1.

    """
    if virus_strains is None:
        virus_strains = {"names": ["base_strain"], "multipliers": np.array([1])}

    elif isinstance(virus_strains, list):
        if len(virus_strains) == 0:
            raise ValueError("The list of 'virus_strains' cannot be empty.")

        sorted_strains = sorted(virus_strains)
        multipliers = np.array(
            [
                params.loc[("virus_strain", name, "multiplier"), "value"]
                for name in sorted_strains
            ]
        )
        multipliers = multipliers / multipliers.max()

        if any(multipliers < 0):
            raise ValueError("Multipliers of 'virus_strains' cannot be <0.")

        virus_strains = {"names": sorted_strains, "multipliers": multipliers}

    else:
        raise ValueError("'virus_strains' is not None and not a list.")

    return virus_strains


def _convert_boolean_initial_infections_to_factorized_array(s):
    arr = np.full(len(s), -1, dtype=np.int8)
    arr[s] = 0
    return arr
