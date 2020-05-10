import numpy as np
import pandas as pd

from sid.config import DTYPE_GROUP_CODE


def factorize_assortative_variables(states, assort_by):
    """Factorize assortative variables.

    This function forms unique values by combining the different values of assortative
    variables. If there are no assortative variables, a single group is assigned to all
    states.

    The group codes are converted to a lower dtype to save memory.

    Args:
        states (pandas.DataFrame): The user-defined initial states.
        assort_by (list, optional): List of variable names. Contacts are assortative by
            these variables.

    Returns:
        group_codes (numpy.ndarray): Array containing the code for each states.
        group_codes_values (numpy.ndarray): One-dimensional array where positions
            correspond the values of assortative variables to form the group.

    """
    if assort_by:
        assort_by_series = [states[col].to_numpy() for col in assort_by]
        group_codes, group_codes_values = pd.factorize(
            pd._libs.lib.fast_zip(assort_by_series), sort=True
        )
        group_codes = group_codes.astype(DTYPE_GROUP_CODE)

    else:
        group_codes = np.zeros(len(states), dtype=np.uint8)
        group_codes_values = [(0,)]

    return group_codes, group_codes_values


def calculate_r_effective(df, window_length=7):
    """Calculate the effective reproduction number.

    source: https://bit.ly/2VZOR5a

    Args:
        df (pandas.DataFrame): states DataFrame for which to calculate R_e, usually
            the states of one period.
        window_length (int): how many periods to use to identify the previously
            infectious people. The lower, the more changes in behavior can be seen,
            but the smaller the number of people on which to calculate R_e.

    Returns:
        r_effective (float): mean number of people infected by someone whose infectious
            spell ended in the last *window_length* periods.

    """
    prev_infected = df[df["cd_infectious_false"].between(-window_length, 0)]
    # the infection counter is only reset to zero once a person becomes infected again
    # so abstracting from very fast reinfections its mean among those that
    # ceased to be infectious in the last window_length is R_e.
    r_effective = prev_infected["n_has_infected"].mean()
    return r_effective


def calculate_r_zero(df, window_length=7):
    """Calculate the basic replication number R_0.

    This is done by dividing the effective reproduction number by the share of
    susceptible people in the dataframe. Using R_e and the share of the susceptible
    people from the very last period of the time means that heterogeneous matching and
    changes in the rate of immunity are neglected.

    source: https://bit.ly/2VZOR5a

    Args:
        df (pd.DataFrame): states DataFrame for which to calculate R_0, usually
            the states of one period.
        window_length (int): how many periods to use to identify the previously
            infectious people. The lower, the more changes in behavior can be seen,
            but the smaller the number of people on which to calculate R_0.

    Returns:
        r_zero (float): mean number of people that would have been infected by someone
            whose infectious spell ended in the last *window_length* periods if everyone
            had been susceptible, neglecting heterogeneous matching and changes in the
            rate of immunity.

    """
    r_effective = calculate_r_effective(df=df, window_length=window_length)
    pct_susceptible = 1 - df["immune"].mean()
    r_zero = r_effective / pct_susceptible
    return r_zero
