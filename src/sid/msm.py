"""Estimate models with the method of simulated moments (MSM).

The method of simulated moments is developed by [1]_, [2]_, and [3]_ and an estimation
technique where the distance between the moments of the actual data and the moments
implied by the model parameters is minimized.

References:

.. [1] McFadden, D. (1989). A method of simulated moments for estimation of discrete
       response models without numerical integration. Econometrica: Journal of the
       Econometric Society, 995-1026.
.. [2] Lee, B. S., & Ingram, B. F. (1991). Simulation estimation of time-series models.
       Journal of Econometrics, 47(2-3), 197-205.
.. [3] Duffie, D., & Singleton, K. (1993). Simulated Moments Estimation of Markov Models
       of Asset Prices. Econometrica, 61(4), 929-952.

"""
import copy
import functools

import numpy as np
import pandas as pd


def get_msm_func(
    simulate,
    calc_moments,
    empirical_moments,
    replace_nans,
    weighting_matrix=None,
):
    """Get the msm function.

    Args:
        simulate (callable): Function which accepts parameters and returns simulated
            data.
        calc_moments (callable or list): Function(s) used to calculate simulated
            moments. Must match structure of empirical moments i.e. if empirical_moments
            is a list of pandas.DataFrames, calc_moments must be a list of the same
            length containing functions that correspond to the moments in
            empirical_moments.
        empirical_moments (pandas.DataFrame or pandas.Series or dict or list): Contains
            the empirical moments calculated for the observed data. Moments should be
            saved to pandas.DataFrame or pandas.Series that can either be passed to the
            function directly or as items of a list or dictionary. Index of
            pandas.DataFrames can be of type MultiIndex, but columns cannot.
        replace_nans (callable or list): Functions(s) specifying how to handle NaNs in
            simulated_moments. Must match structure of empirical_moments. Exception: If
            only one replacement function is specified, it will be used on all sets of
            simulated moments.
        weighting_matrix (numpy.ndarray): Square matrix of dimension (NxN) with N
            denoting the number of empirical_moments. Used to weight squared moment
            errors.
        return_scalar (bool): Indicates whether to return moment error
            vector (False) or weighted square product of moment error vector (True).

    Returns:
        msm_func (callable): MSM function where all arguments except the parameter
            vector are set.

    """
    if weighting_matrix is None:
        weighting_matrix = get_diag_weighting_matrix(empirical_moments)

    if not _is_diagonal(weighting_matrix):
        raise ValueError("weighting_matrix must be diagonal.")

    empirical_moments = copy.deepcopy(empirical_moments)

    empirical_moments = _harmonize_input(empirical_moments)
    calc_moments = _harmonize_input(calc_moments)

    # If only one replacement function is given for multiple sets of moments, duplicate
    # replacement function for all sets of simulated moments.
    if callable(replace_nans):
        replace_nans = {k: replace_nans for k in empirical_moments}
    replace_nans = _harmonize_input(replace_nans)

    if 1 < len(replace_nans) < len(empirical_moments):
        raise ValueError(
            "Replacement functions can only be matched 1:1 or 1:n with sets of "
            "empirical moments."
        )

    elif len(replace_nans) > len(empirical_moments):
        raise ValueError(
            "There are more replacement functions than sets of empirical moments."
        )

    else:
        pass

    if len(calc_moments) != len(empirical_moments):
        raise ValueError(
            "Number of functions to calculate simulated moments must be equal to "
            "the number of sets of empirical moments."
        )

    msm_func = functools.partial(
        _msm,
        simulate=simulate,
        calc_moments=calc_moments,
        empirical_moments=empirical_moments,
        replace_nans=replace_nans,
        weighting_matrix=weighting_matrix,
    )

    return msm_func


def _msm(
    params,
    simulate,
    calc_moments,
    empirical_moments,
    replace_nans,
    weighting_matrix,
):
    """The MSM criterion function.

    This function will be prepared by :func:`get_msm_func` and have all its arguments
    except `params` attached to it.

    """
    empirical_moments = copy.deepcopy(empirical_moments)

    df = simulate(params)

    simulated_moments = {name: func(df) for name, func in calc_moments.items()}

    simulated_moments = {
        name: sim_mom.reindex_like(empirical_moments[name])
        for name, sim_mom in simulated_moments.items()
    }

    simulated_moments = {
        name: replace_nans[name](sim_mom) for name, sim_mom in simulated_moments.items()
    }

    flat_empirical_moments = _flatten_index(empirical_moments)
    flat_simulated_moments = _flatten_index(simulated_moments)

    # Order is important to manfred.
    moment_errors = flat_simulated_moments - flat_empirical_moments

    # Return moment errors as indexed DataFrame or calculate weighted square product of
    # moment errors depending on return_scalar.
    root_contribs = np.sqrt(np.diagonal(weighting_matrix)) * moment_errors
    value = np.sum(root_contribs ** 2)

    out = {
        "value": value,
        "root_contributions": root_contribs,
        "empirical_moments": empirical_moments,
        "simulated_moments": simulated_moments,
    }

    return out


def get_diag_weighting_matrix(empirical_moments, weights=None):
    """Create a diagonal weighting matrix from weights.

    Args:
        empirical_moments (pandas.DataFrame or pandas.Series or dict or list): Contains
            the empirical moments calculated for the observed data. Moments should be
            saved to pandas.DataFrame or pandas.Series that can either be passed to the
            function directly or as items of a list or dictionary.
        weights (pandas.DataFrame or pandas.Series or dict or list): Contains weights
            (usually variances) of empirical moments. Must match structure of
            empirical_moments i.e. if empirical_moments is a list of
            :class:`pandas.DataFrame`, weights be list of pandas.DataFrames as well
            where each DataFrame entry contains the weight for the corresponding moment
            in empirical_moments.

    Returns:
        (numpy.ndarray): Array contains a diagonal weighting matrix.

    """
    weights = copy.deepcopy(weights)
    empirical_moments = copy.deepcopy(empirical_moments)
    empirical_moments = _harmonize_input(empirical_moments)

    # Use identity matrix if no weights are specified.
    if weights is None:
        flat_weights = _flatten_index(empirical_moments)
        flat_weights[:] = 1

    # Harmonize input weights.
    else:
        weights = _harmonize_input(weights)

        # Reindex weights to ensure they are assigned to the correct moments in
        # the msm function.
        weights = {
            name: weight.reindex_like(empirical_moments[name])
            for name, weight in weights.items()
        }

        flat_weights = _flatten_index(weights)

    return np.diag(flat_weights)


def get_flat_moments(empirical_moments):
    """Compute the empirical moments flat indexes.

    Args:
        empirical_moments (pandas.DataFrame or pandas.Series or dict or list):
            Containing pandas.DataFrame or pandas.Series. Contains the empirical moments
            calculated for the observed data. Moments should be saved to
            pandas.DataFrame or pandas.Series that can either be passed to the function
            directly or as items of a list or dictionary.

    Returns:
        flat_empirical_moments (pandas.DataFrame): Vector of empirical_moments with flat
            index.

    """
    empirical_moments = copy.deepcopy(empirical_moments)
    empirical_moments = _harmonize_input(empirical_moments)
    flat_empirical_moments = _flatten_index(empirical_moments)

    return flat_empirical_moments


def _harmonize_input(data):
    """Harmonize different types of inputs by turning all inputs into dicts.

    - pandas.DataFrames/Series and callable functions will turn into a list containing a
      single item (i.e. the input).
    - Dictionaries will be sorted according to keys and then turn into a list containing
      the dictionary entries.

    """
    # Convert single DataFrames, Series or function into list containing one item.
    if isinstance(data, (pd.DataFrame, pd.Series)) or callable(data):
        data = {0: data}

    # Sort dictionary according to keys and turn into list.
    elif isinstance(data, dict):
        pass

    elif isinstance(data, (tuple, list)):
        data = {i: data_ for i, data_ in enumerate(data)}

    else:
        raise ValueError(
            "Function only accepts lists, dictionaries, functions, Series and "
            "DataFrames as inputs."
        )

    return data


def _flatten_index(data):
    """Flatten the index as a combination of the former index and the columns."""
    data_flat = []

    for name, series_or_df in data.items():
        series_or_df.index = series_or_df.index.map(str)
        # Unstack DataFrames and Series to add columns/Series name to index.
        if isinstance(series_or_df, pd.DataFrame):
            df = series_or_df.rename(columns=lambda x: f"{name}_{x}")
        # Series without a name are named using a counter to avoid duplicate indexes.
        elif isinstance(series_or_df, pd.Series):
            df = series_or_df.to_frame(name=f"{name}")
        else:
            raise NotImplementedError

        # Columns to the index.
        df = df.unstack()
        df.index = df.index.to_flat_index().str.join("_")
        data_flat.append(df)

    return pd.concat(data_flat)


def _is_diagonal(mat):
    """Check if the matrix is diagonal."""
    return not np.count_nonzero(mat - np.diag(np.diagonal(mat)))
