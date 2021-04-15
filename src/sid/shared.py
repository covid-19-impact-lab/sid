import numba as nb
import numpy as np
import pandas as pd
from sid.config import DTYPE_GROUP_CODE
from sid.config import INDEX_NAMES
from sid.config import ROOT_DIR


def load_epidemiological_parameters():
    """Load epidemiological_parameters."""
    return pd.read_csv(ROOT_DIR / "covid_epi_params.csv", index_col=INDEX_NAMES)


def factorize_assortative_variables(states, assort_by):
    """Factorize assortative variables.

    This function forms unique values by combining the different values of assortative
    variables. If there are no assortative variables, a single group is assigned to all
    states.

    The unique values of the ``assort_by`` variables should be sorted to maintain the
    relationship, especially if already factorized variables are passed.

    The group codes are converted to a lower dtype to save memory.

    Args:
        states (pandas.DataFrame): The user-defined initial states.
        assort_by (List[str]): List of variable names. Contacts are assortative by these
            variables.

    Returns:
        (tuple): Tuple containing

        - group_codes (:class:`numpy.ndarray`): Array containing the code for each
          states.
        - group_codes_values (:class:`numpy.ndarray`): One-dimensional array where
          positions correspond the values of assortative variables to form the group.

    """
    if len(assort_by) == 1:
        assort_by_series = states[assort_by[0]]
        try:
            integer_assort_by_series = assort_by_series.astype(int)
            assort_by_series = integer_assort_by_series.where(
                integer_assort_by_series >= 0, pd.NA
            ).astype("Int64")
        except (TypeError, ValueError):
            pass

        codes, uniques = pd.factorize(assort_by_series, sort=True)
        codes = codes.astype(DTYPE_GROUP_CODE)
    elif assort_by:
        assort_by_series = [states[col].to_numpy() for col in assort_by]
        codes, uniques = pd.factorize(
            pd._libs.lib.fast_zip(assort_by_series), sort=True
        )
        codes = codes.astype(DTYPE_GROUP_CODE)
    else:
        codes = np.zeros(len(states), dtype=DTYPE_GROUP_CODE)
        uniques = np.empty(1, dtype=object)
        uniques[0] = (0,)

    return codes, uniques


def random_choice(choices, probabilities=None, decimals=5):
    """Return elements of choices for a two-dimensional array of probabilities.

    It is assumed that probabilities are ordered (n_samples, n_choices).

    The function is taken from this `StackOverflow post
    <https://stackoverflow.com/questions/40474436>`_ as a workaround for
    :func:`numpy.random.choice` as it can only handle one-dimensional probabilities.

    Examples:
        Here is an example with non-zero probabilities.

        >>> n_samples = 1_000_000
        >>> n_choices = 3
        >>> p = np.array([0.15, 0.35, 0.5])
        >>> ps = np.tile(p, (n_samples, 1))
        >>> choices = random_choice(n_choices, ps)
        >>> np.round(np.bincount(choices) / n_samples, decimals=2)
        array([0.15, 0.35, 0.5 ])

        Here is an example where one choice has probability zero.

        >>> choices = np.arange(3)
        >>> p = np.array([0.4, 0, 0.6])
        >>> ps = np.tile(p, (n_samples, 1))
        >>> choices = random_choice(3, ps)
        >>> np.round(np.bincount(choices) / n_samples, decimals=2)
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
    choices = np.atleast_2d(choices)

    if probabilities is None:
        n_choices = choices.shape[-1]
        probabilities = np.ones((1, n_choices)) / n_choices
        probabilities = np.broadcast_to(probabilities, choices.shape)
    elif isinstance(probabilities, (pd.Series, pd.DataFrame)):
        probabilities = probabilities.to_numpy()
    elif isinstance(probabilities, (dict, list, tuple)):
        probabilities = np.array(list(probabilities))
    elif isinstance(probabilities, np.ndarray):
        pass
    else:
        raise TypeError(f"'probabilities' has invalid type {type(probabilities)}.")
    probabilities = np.atleast_2d(probabilities)

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


@nb.njit  # pragma: no cover
def boolean_choice(truth_probability):  # pragma: no cover
    """Sample boolean value with probability given for ``True``.

    Args:
        truth_probability (float): Must be between 0 and 1.

    Returns:
        bool: Boolean array.

    Example:
        >>> boolean_choice(1)
        True
        >>> boolean_choice(0)
        False

    """
    u = np.random.uniform(0, 1)
    return u <= truth_probability


def boolean_choices(truth_probabilities):
    """Sample boolean value with probabilities given for ``True``.

    Args:
        truth_probabilities (float): Must be between 0 and 1.

    Returns:
        bool: Boolean array.

    Example:
        >>> boolean_choice(np.array([1, 0]))
        array([ True, False])

    """
    u = np.random.uniform(0, 1, size=len(truth_probabilities))
    return u <= truth_probabilities


def separate_contact_model_names(contact_models):
    recurrent_models = [c for c in contact_models if contact_models[c]["is_recurrent"]]
    random_models = [c for c in contact_models if not contact_models[c]["is_recurrent"]]

    return recurrent_models, random_models


def fast_condition_evaluator(df, condition):
    if callable(condition):
        out = condition(df)
    elif isinstance(condition, str):
        out = df.eval(condition)
    else:
        raise ValueError()
    return out
