"""Functions to work with transition matrices for assortative matching."""
import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List as NumbaList

from sid.shared import factorize_assortative_variables


def create_group_transition_probs(states, assort_by, params, model_name):
    """Create a transition matrix for groups.

    Args:
        states (pandas.DataFrame): see :ref:`states`
        assort_by (list): List of variables that influence matching probabilities.
        params (pandas.DataFrame): See :ref:`params`
        model_name (str): name of the contact model.

    Returns
        cum_probs (numpy.ndarray): Array of shape n_group, n_groups. cum_probs[i, j]
            is the probability that an individual from group i meets someone from group
            j or lower.

    """
    _, group_codes_values = factorize_assortative_variables(states, assort_by)
    probs = np.ones((len(group_codes_values), len(group_codes_values)))

    if assort_by:
        same_probs = []
        other_probs = []
        for var in assort_by:
            p = params.loc[("assortative_matching", model_name, var), "value"]
            n_vals = len(states[var].unique())
            same_probs.append(p)
            other_probs.append((1 - p) / (n_vals - 1))

        for i, g_from in enumerate(group_codes_values):
            for j, g_to in enumerate(group_codes_values):
                for v, (val1, val2) in enumerate(zip(g_from, g_to)):
                    if val1 == val2:
                        probs[i, j] *= same_probs[v]
                    else:
                        probs[i, j] *= other_probs[v]

    cum_probs = probs.cumsum(axis=1)

    return cum_probs


def create_transition_matrix_from_own_prob(own_prob, group_names=None):
    """Create a transition matrix.

    The matrix is calculated from the probability of staying inside
    a group, spreading the remaining probability mass uniformly across
    other groups.

    Args:
        own_prob (float or pd.Series): Probability of staying inside
            the own group, either as scalar or as pandas.Series with one
            entry per group.
        group_names (list): List group codes. Mandatory if own_group
            is a scalar.

    Returns:
        pd.DataFrame: Transition matrix as square DataFrame. The index
            and columns are the group_names.

    Example:

        >>> create_transition_matrix_from_own_prob(0.6, ["a", "b"])
             a    b
        a  0.6  0.4
        b  0.4  0.6

        >>> op = pd.Series([0.6, 0.7], index=["a", "b"])
        >>> create_transition_matrix_from_own_prob(op)
             a    b
        a  0.6  0.4
        b  0.3  0.7

    """
    if np.isscalar(own_prob) and group_names is None:
        raise ValueError("If own_prob is a scalar you must provide group_probs.")
    elif np.isscalar(own_prob):
        own_prob = pd.Series(data=own_prob, index=group_names)
    elif group_names is not None:
        own_prob = own_prob.loc[group_names]

    n_groups = len(own_prob)
    other_prob = (1 - own_prob) / (n_groups - 1)

    trans_arr = np.tile(other_prob.to_numpy().reshape(-1, 1), n_groups)
    trans_arr[np.diag_indices(n_groups)] = own_prob
    trans_df = pd.DataFrame(trans_arr, columns=own_prob.index, index=own_prob.index)
    return trans_df


def join_transition_matrices(trans_mats):
    """Join several transition matrices into one, assuming independence.

    Args:
        trans_mats (list): List of square DataFrames. The index
            and columns are the group_names.

    Returns:
        pd.DataFrame: Joined transition matrix. The index and columns
            are the cartesian product of all individual group names in
            the same order as trans_mats.

    """
    readable_index = pd.MultiIndex.from_product([tm.index for tm in trans_mats])
    indexer = np.array(
        pd.MultiIndex.from_product([range(len(tm)) for tm in trans_mats]).to_list()
    )
    trans_arrs = NumbaList()
    for tm in trans_mats:
        trans_arrs.append(tm.to_numpy())

    prob_arr = _numba_join_transition_matrices(len(readable_index), trans_arrs, indexer)

    prob_df = pd.DataFrame(prob_arr, index=readable_index, columns=readable_index)
    return prob_df


@njit
def _numba_join_transition_matrices(dim_out, trans_arrs, indexer):
    out = np.ones((dim_out, dim_out))
    for i in range(dim_out):
        for j in range(dim_out):
            for k in range(len(trans_arrs)):
                row = indexer[i, k]
                col = indexer[j, k]
                out[i, j] *= trans_arrs[k][row, col]
    return out
