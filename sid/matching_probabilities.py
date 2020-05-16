"""Functions to work with transition matrices for assortative matching."""
import string

import numpy as np
import pandas as pd

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

    if not assort_by:
        probs = np.ones((len(group_codes_values), len(group_codes_values)))

    else:
        trans_mats = []
        for var in assort_by:
            tm = _get_transition_matrix_from_params(params, states, var, model_name)
            trans_mats.append(tm)

        probs = _join_transition_matrices(trans_mats)
        probs = probs.loc[group_codes_values, group_codes_values].to_numpy()

    cum_probs = probs.cumsum(axis=1)

    return cum_probs


def _get_transition_matrix_from_params(params, states, variable, model_name):
    """Extract transition matrix for one assort_by variable from params.

    Args:
        params (pd.DataFrame): see :ref:`params`
        states (pd.DataFrame): see :ref:`states`
        variable (str): Name of the assort by variable
        model_name (str): Name of the contact model in which variable is used.

    Returns:
        pd.DataFrame: The transition matrix.

    """
    if ("assortative_matching", model_name, variable) in params.index:
        own_prob = params.loc[("assortative_matching", model_name, variable), "value"]
        group_names = states[variable].unique().tolist()
        trans_mat = _create_transition_matrix_from_own_prob(own_prob, group_names)
    else:
        loc = f"assortative_matching_{model_name}_{variable}"
        par = params.loc[loc]
        from_ = par.index.get_level_values("subcategory")
        to = par.index.get_level_values("name")
        if (from_ == to).all():
            own_prob = params.loc[loc, "value"]
            own_prob.index = own_prob.index.get_level_values("name")
            trans_mat = _create_transition_matrix_from_own_prob(own_prob)
        else:
            trans_mat = params.loc[loc, "value"].unstack()

    return trans_mat


def _create_transition_matrix_from_own_prob(own_prob, group_names=None):
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

        >>> _create_transition_matrix_from_own_prob(0.6, ["a", "b"])
             a    b
        a  0.6  0.4
        b  0.4  0.6

        >>> op = pd.Series([0.6, 0.7], index=["a", "b"])
        >>> _create_transition_matrix_from_own_prob(op)
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


def _join_transition_matrices(trans_mats):
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
    kronecker_product = _einsum_kronecker_product(*trans_mats)
    transition_matrix = pd.DataFrame(
        kronecker_product, index=readable_index, columns=readable_index
    )
    return transition_matrix


def _einsum_kronecker_product(*trans_mats):
    """Compute a Kronecker product of multiple matrices with :func:`numpy.einsum`.

    The reshape is necessary because :func:`numpy.einsum` produces a matrix with as many
    dimensions as transition probability matrices. Each dimension has as many values as
    rows or columns in the transition matrix.

    The ordering of letters in the :func:`numpy.einsum` signature for the result ensure
    that the reshape to a two-dimensional matrix does produce the correct Kronecker
    product.

    """
    n_groups = np.prod([i.shape[0] for i in trans_mats])
    signature = _generate_einsum_signature(len(trans_mats))

    kronecker_product = np.einsum(signature, *trans_mats).reshape(n_groups, n_groups)

    return kronecker_product


def _generate_einsum_signature(n_trans_prob):
    """Generate the signature for :func:`numpy.einsum` to compute a Kronecker product.

    The ordering of the letters in the result is necessary so that the reshape to a
    square matrix does not fail.

    Example:

        >>> _generate_einsum_signature(2)
        'ab, cd -> acbd'

        >>> _generate_einsum_signature(3)
        'ab, cd, ef -> acebdf'

        >>> _generate_einsum_signature(4)
        'ab, cd, ef, gh -> acegbdfh'

    """
    n_letters = n_trans_prob * 2
    letters = string.ascii_letters[:n_letters]

    inputs = [letters[i : i + 2] for i in range(0, len(letters), 2)]
    result = ["".join([duo[i] for duo in inputs]) for i in range(len(inputs[0]))]

    return ", ".join(inputs) + " -> " + "".join(result)
