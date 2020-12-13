from collections.abc import Iterable
from typing import Any
from typing import Dict

import numpy as np
import pandas as pd
from sid.config import INITIAL_CONDITIONS


def parse_duration(duration):
    """Parse the user-defined duration.

    Args:
        duration (dict): Duration is a dictionary containing kwargs for
            :func:`pandas.date_range`.

    Returns:
        new_duration (dict): A dictionary containing start and end dates and an iterable
            of the same types.

    Examples:
        >>> parse_duration({"start": "2020-03-01", "end": "2020-03-10"})
        {'start': Timestamp('2020-03-01 00:00:00', freq='D'), 'end': ...

    """
    if duration is None:
        duration = {"start": "2020-01-27", "periods": 10}

    iterable = pd.date_range(**duration)

    internal_duration = {}
    internal_duration["start"] = iterable[0]
    internal_duration["end"] = iterable[-1]
    internal_duration["dates"] = iterable

    return internal_duration


def parse_share_known_cases(share_known_cases, duration, burn_in_periods):
    """Parse the share of known cases.

    In case ``share_known_cases is None``, the multiplier is set to 0 which means no
    cases among all cases are known and receive a test.

    """
    extended_index = np.append(burn_in_periods, duration["dates"])

    if isinstance(share_known_cases, (float, int)):
        share_known_cases = pd.Series(index=extended_index, data=share_known_cases)

    elif isinstance(share_known_cases, pd.Series):
        if not duration["dates"].isin(share_known_cases.index).all():
            raise ValueError(
                "'share_known_cases' must be given for each date of the simulation "
                "period."
            )
        # Extend series with burn-in periods and if shares for burn-in periods do not
        # exist, backfill NaNs.
        share_known_cases = share_known_cases.reindex(extended_index).fillna(
            method="bfill"
        )

    elif share_known_cases is None:
        share_known_cases = pd.Series(index=extended_index, data=0)

    else:
        raise ValueError(
            f"'share_known_cases' is {type(share_known_cases)}, but must be int, float "
            "or pd.Series."
        )

    return share_known_cases


def parse_initial_conditions(
    ic: Dict[str, Any], start_date_simulation: pd.Timestamp
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

    return ic
