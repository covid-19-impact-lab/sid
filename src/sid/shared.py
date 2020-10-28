import numpy as np
import pandas as pd
from sid.config import DTYPE_GROUP_CODE
from sid.config import INDEX_NAMES
from sid.config import ROOT_DIR


def get_epidemiological_parameters():
    return pd.read_csv(ROOT_DIR / "covid_epi_params.csv", index_col=INDEX_NAMES)


def get_date(states):
    return states.date.iloc[0]


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
        (tuple): Tuple containing

        - group_codes (numpy.ndarray): Array containing the code for each states.
        - group_codes_values (numpy.ndarray): One-dimensional array where positions
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

    More information can be found here: https://bit.ly/2VZOR5a.

    Args:
        df (pandas.DataFrame): states DataFrame for which to calculate R_e, usually
            the states of one day.
        window_length (int): how many days to use to identify the previously infectious
            people. The lower, the more changes in behavior can be seen, but the smaller
            the number of people on which to calculate R_e.

    Returns:
        r_effective (float): mean number of people infected by someone whose infectious
            spell ended in the last *window_length* days.

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
    susceptible people in the DataFrame. Using R_e and the share of the susceptible
    people from the very last period of the time means that heterogeneous matching and
    changes in the rate of immunity are neglected.

    More explanation can be found here: https://bit.ly/2VZOR5a.

    Args:
        df (pandas.DataFrame): states DataFrame for which to calculate R_0, usually the
            states of one period.
        window_length (int): how many days to use to identify the previously infectious
            people. The lower, the more changes in behavior can be seen, but the smaller
            the number of people on which to calculate R_0.

    Returns:
        r_zero (float): mean number of people that would have been infected by someone
            whose infectious spell ended in the last *window_length* days if everyone
            had been susceptible, neglecting heterogeneous matching and changes in the
            rate of immunity.

    """
    r_effective = calculate_r_effective(df=df, window_length=window_length)
    pct_susceptible = 1 - df["immune"].mean()
    r_zero = r_effective / pct_susceptible
    return r_zero


def random_choice(choices, probabilities=None, decimals=5):
    """Return elements of choices for a two-dimensional array of probabilities.

    It is assumed that probabilities are ordered (n_samples, n_choices).

    The function is taken from this `StackOverflow post
    <https://stackoverflow.com/questions/40474436>`_ as a workaround for
    :func:`numpy.random.choice` as it can only handle one-dimensional probabilities.

    Examples:
        Here is an example with non-zero probabilities.

        >>> n_samples = 100_000
        >>> n_choices = 3
        >>> p = np.array([0.15, 0.35, 0.5])
        >>> ps = np.tile(p, (n_samples, 1))
        >>> choices = random_choice(n_choices, ps)
        >>> np.round(np.bincount(choices), decimals=-3) / n_samples
        array([0.15, 0.35, 0.5 ])

        Here is an example where one choice has probability zero.

        >>> choices = np.arange(3)
        >>> p = np.array([0.4, 0, 0.6])
        >>> ps = np.tile(p, (n_samples, 1))
        >>> choices = random_choice(3, ps)
        >>> np.round(np.bincount(choices), decimals=-3) / n_samples
        array([0.4, 0. , 0.6])

    """
    if isinstance(choices, int):
        choices = np.arange(choices)
    elif isinstance(choices, (dict, list, tuple)):
        choices = np.array(list(choices))
    elif isinstance(choices, np.ndarray):
        pass
    else:
        raise TypeError(f"'choices' has invalid type {type(choices)}.")

    if probabilities is None:
        n_choices = choices.shape[-1]
        probabilities = np.ones((1, n_choices)) / n_choices
        probabilities = np.broadcast_to(probabilities, choices.shape)
    elif isinstance(probabilities, (pd.Series, pd.DataFrame)):
        probabilities = probabilities.to_numpy()
    elif isinstance(probabilities, np.ndarray):
        pass
    else:
        raise TypeError(f"'probabilities' has invalid type {type(probabilities)}.")

    cumulative_distribution = probabilities.cumsum(axis=1)
    # Probabilities often do not sum to one but 0.99999999999999999.
    cumulative_distribution[:, -1] = np.round(cumulative_distribution[:, -1], decimals)

    if not (cumulative_distribution[:, -1] == 1).all():
        raise ValueError("Probabilities do not sum to one.")

    u = np.random.rand(cumulative_distribution.shape[0], 1)

    # Note that :func:`np.argmax` returns the first index for multiple maximum values.
    indices = (u < cumulative_distribution).argmax(axis=1)

    out = np.take(choices, indices)
    if out.shape == (1,):
        out = out[0]

    return out


def validate_return_is_series_or_ndarray(x, index=None, when=None):
    if isinstance(x, (pd.Series, np.ndarray)):
        return pd.Series(data=x, index=index)
    else:
        raise ValueError(f"'{when}' must always return a pd.Series or a np.ndarray.")


def date_is_within_start_and_end_date(date, start, end):
    """Indicate whether date lies within the start and end dates.

    ``None`` is interpreted as an open boundary.

    Examples:
        >>> date_is_within_start_and_end_date("2020-01-02", "2020-01-01", "2020-01-03")
        True
        >>> date_is_within_start_and_end_date("2020-01-01", "2020-01-02", "2020-01-03")
        False
        >>> date_is_within_start_and_end_date("2020-01-01", None, "2020-01-03")
        True

    """
    is_within = True
    if start is not None and pd.Timestamp(start) > pd.Timestamp(date):
        is_within = False
    if end is not None and pd.Timestamp(end) < pd.Timestamp(date):
        is_within = False

    return is_within
