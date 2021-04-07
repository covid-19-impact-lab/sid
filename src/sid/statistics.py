from typing import Union

import numpy as np
import pandas as pd

__all__ = ["calculate_r_effective", "calculate_r_zero"]


def calculate_r_effective(df: pd.DataFrame, window_length: int = 7) -> pd.Series:
    """Calculate the effective reproduction number, :math:`R_e`.

    More explanation can be found in the `Wikipedia article <Wikipedia>`_.

    Note:
        The infection counter is only reset to zero once a person becomes infected again
        so abstracting from very fast reinfections its mean among those that ceased to
        be infectious in the last window_length is :math:`R_e`.

    Args:
        df (pandas.DataFrame): states DataFrame for which to calculate :math:`R_e`,
            usually the states of one day.
        window_length (int): how many days to use to identify the previously infectious
            people. The lower, the more changes in behavior can be seen, but the smaller
            the number of people on which to calculate :math:`R_e`.

    Returns:
        r_effective (pandas.Series): mean number of people infected by someone whose
            infectious spell ended in the last *window_length* days.

    .. _Wikipedia:
        https://en.wikipedia.org/wiki/Basic_reproduction_number

    """
    infectious_in_the_last_n_days = df["cd_infectious_false"].between(-window_length, 0)

    grouper = _create_time_grouper(df)
    if grouper is None:
        r_effective = df.loc[infectious_in_the_last_n_days]["n_has_infected"].mean()
    else:
        r_effective = (
            df.loc[infectious_in_the_last_n_days]
            .groupby(grouper)["n_has_infected"]
            .mean()
        )

        # The groupby-mean removed some dates without infections. Add them again.
        all_periods = np.sort(df[grouper.key].unique())
        r_effective = r_effective.reindex(index=all_periods).fillna(0)

    return r_effective


def calculate_r_zero(df: pd.DataFrame, window_length: int = 7) -> pd.Series:
    """Calculate the basic replication number :math:`R_0`.

    This is done by dividing the effective reproduction number by the share of
    susceptible people in the DataFrame. Using R_e and the share of the susceptible
    people from the very last period of the time means that heterogeneous matching and
    changes in the rate of immunity are neglected.

    More explanation can be found here: https://bit.ly/2VZOR5a.

    Args:
        df (pandas.DataFrame): states DataFrame for which to calculate :math:`R_0`,
            usually the states of one period.
        window_length (int): how many days to use to identify the previously infectious
            people. The lower, the more changes in behavior can be seen, but the smaller
            the number of people on which to calculate :math:`R_0`.

    Returns:
        r_zero (pandas.Series): The average number of people that would have been
            infected by someone whose infectious spell ended in the last *window_length*
            days if everyone had been susceptible, neglecting heterogeneous matching and
            changes in the rate of immunity.

    """
    r_effective = calculate_r_effective(df=df, window_length=window_length)

    grouper = _create_time_grouper(df)
    if grouper is None:
        share_susceptibles = 1 - df["immune"].mean()
        r_zero = r_effective / share_susceptibles
    else:
        share_susceptibles = 1 - df.groupby(grouper)["immune"].mean()
        r_zero = r_effective / share_susceptibles

    return r_zero


def _create_time_grouper(df: pd.DataFrame) -> Union[pd.Grouper, None]:
    """Create a grouper for the time dimension of the DataFrame."""
    if "date" in df.columns:
        grouper = pd.Grouper(key="date", freq="D")
    elif "period" in df.columns:
        grouper = pd.Grouper(key="period")
    else:
        grouper = None

    return grouper
