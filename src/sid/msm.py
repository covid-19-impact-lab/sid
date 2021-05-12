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
import functools

import numpy as np
import pandas as pd


def get_msm_func(
    simulate,
    calc_moments,
    empirical_moments,
    replace_nans,
    weighting_matrix=None,
    additional_outputs=None,
):
    """Get the msm function.

    Args:
        simulate (callable): Function which accepts parameters and returns simulated
            data.
        calc_moments (callable or dict): Function(s) used to calculate simulated
            moments. If it is a dictionary, it must have the same keys as
            empirical_moments
        empirical_moments (pandas.DataFrame or pandas.Series or dict): One pandas
            object or a dictionary of pandas objects with empirical moments.
        replace_nans (callable or list): Functions(s) specifying how to handle NaNs in
            simulated_moments. Must match structure of empirical_moments. Exception: If
            only one replacement function is specified, it will be used on all sets of
            simulated moments.
        weighting_matrix (numpy.ndarray): Square matrix of dimension (NxN) with N
            denoting the number of empirical_moments. Used to weight squared moment
            errors.
        additional_outputs (dict or None): Dictionary of functions. Each function is
            evaluated on the output of the simulate function and the result is
            saved in the output dictionary of the msm function.

    Returns:
        msm_func (callable): MSM function where all arguments except the parameter
            vector are set.

    """
    if weighting_matrix is None:
        weighting_matrix = get_diag_weighting_matrix(empirical_moments)

    if not _is_diagonal(weighting_matrix):
        raise ValueError("weighting_matrix must be diagonal.")

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

    if additional_outputs is not None:
        if not _is_dict_of_callables(additional_outputs):
            raise ValueError("additional_outputs must be a dict of callables.")
    else:
        additional_outputs = {}

    invalid_keys = {
        "value",
        "root_contributions",
        "root_contributions",
        "empirical_moments",
        "simulated_moments",
    }

    invalid_present = invalid_keys.intersection(additional_outputs)

    if invalid_present:
        raise ValueError("Invalid keys in additional_outputs: {invalid}")

    msm_func = functools.partial(
        _msm,
        simulate=simulate,
        calc_moments=calc_moments,
        empirical_moments=empirical_moments,
        replace_nans=replace_nans,
        weighting_matrix=weighting_matrix,
        additional_outputs=additional_outputs,
    )

    return msm_func


def _msm(
    params,
    simulate,
    calc_moments,
    empirical_moments,
    replace_nans,
    weighting_matrix,
    additional_outputs,
):
    """The MSM criterion function.

    This function will be prepared by :func:`get_msm_func` and have all its arguments
    except `params` attached to it.

    """
    sim_out = simulate(params)

    simulated_moments = {name: func(sim_out) for name, func in calc_moments.items()}

    simulated_moments = {
        name: sim_mom.reindex_like(empirical_moments[name])
        for name, sim_mom in simulated_moments.items()
    }

    simulated_moments = {
        name: replace_nans[name](sim_mom) for name, sim_mom in simulated_moments.items()
    }

    flat_empirical_moments = _flatten_index(empirical_moments)
    flat_simulated_moments = _flatten_index(simulated_moments)

    moment_errors = flat_simulated_moments - flat_empirical_moments

    root_contribs = np.sqrt(np.diagonal(weighting_matrix)) * moment_errors
    value = np.sum(root_contribs ** 2)

    out = {
        "value": value,
        "root_contributions": root_contribs,
        "empirical_moments": empirical_moments,
        "simulated_moments": simulated_moments,
    }

    for name, func in additional_outputs.items():
        out[name] = func(sim_out)

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
    empirical_moments = _harmonize_input(empirical_moments)

    # Use identity matrix if no weights are specified.
    if weights is None:
        flat_weights = _flatten_index(empirical_moments)
        flat_weights[:] = 1

    # Harmonize input weights.
    else:
        weights = _harmonize_input(weights)

        # Reindex weights to ensure they are assigned to the correct moments in
        # the msm function and convert scalars to pandas objects
        cleaned = {}
        for name, weight in weights.items():
            if np.isscalar(weight):
                nonscalar = empirical_moments[name].copy(deep=True)
                nonscalar[:] = weight
                cleaned[name] = nonscalar
            else:
                cleaned[name] = weight.reindex_like(empirical_moments[name])

        flat_weights = _flatten_index(cleaned)

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
    empirical_moments = _harmonize_input(empirical_moments)
    flat_empirical_moments = _flatten_index(empirical_moments)

    return flat_empirical_moments


def _harmonize_input(data):
    """Harmonize different types of inputs by turning all inputs into dicts."""
    if isinstance(data, (pd.DataFrame, pd.Series)) or callable(data):
        data = {0: data}

    elif isinstance(data, dict):
        pass

    else:
        raise ValueError(
            "Moments must be pandas objects or dictionaries of pandas objects."
        )

    return data


def _flatten_index(data):
    """Flatten the index as a combination of the former index and the columns."""
    data_flat = []

    for name, series_or_df in data.items():
        series_or_df = series_or_df.copy(deep=True)
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


def _is_dict_of_callables(x):
    return isinstance(x, dict) and all(callable(value) for value in x.values())
