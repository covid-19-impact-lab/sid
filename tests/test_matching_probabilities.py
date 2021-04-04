import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sid.config import DTYPE_GROUP_TRANSITION_PROBABILITIES
from sid.matching_probabilities import _create_transition_matrix_from_own_prob
from sid.matching_probabilities import _einsum_kronecker_product
from sid.matching_probabilities import _join_transition_matrices
from sid.matching_probabilities import create_cumulative_group_transition_probabilities
from sid.shared import factorize_assortative_variables


@pytest.mark.integration
def test_join_transition_matrices():
    t1 = _create_transition_matrix_from_own_prob(0.6, ["bla", "blubb"])
    t2 = _create_transition_matrix_from_own_prob(
        pd.Series([0.6, 0.7], index=["a", "b"])
    )
    exp_data = np.kron(t1, t2)

    ind_tups = [("bla", "a"), ("bla", "b"), ("blubb", "a"), ("blubb", "b")]

    ind = pd.MultiIndex.from_tuples(ind_tups)
    expected = pd.DataFrame(exp_data, columns=ind, index=ind)

    calculated = _join_transition_matrices([t1, t2])

    assert_frame_equal(calculated, expected)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("assort_by", "expected"),
    [
        (
            ["age_group", "region"],
            np.array(
                [
                    [0.45, 0.025, 0.025, 0.45, 0.025, 0.025],
                    [0.025, 0.45, 0.025, 0.025, 0.45, 0.025],
                    [0.025, 0.025, 0.45, 0.025, 0.025, 0.45],
                    [0.45, 0.025, 0.025, 0.45, 0.025, 0.025],
                    [0.025, 0.45, 0.025, 0.025, 0.45, 0.025],
                    [0.025, 0.025, 0.45, 0.025, 0.025, 0.45],
                ]
            ),
        ),
        ([], np.array([[1]])),
    ],
)
def test_create_cumulative_group_transition_probabilities(
    initial_states, assort_by, params, expected
):
    # append assortative matching parameters
    assort_index = pd.MultiIndex.from_tuples(
        [
            ("assortative_matching", "model1", "age_group"),
            ("assortative_matching", "model1", "region"),
        ]
    )
    assort_probs = pd.DataFrame(columns=params.columns, index=assort_index)
    assort_probs["value"] = [0.5, 0.9]
    params = params.append(assort_probs)

    _, groups = factorize_assortative_variables(initial_states, assort_by, False)

    transition_matrix = create_cumulative_group_transition_probabilities(
        states=initial_states,
        assort_by=assort_by,
        params=params,
        model_name="model1",
        groups=groups,
    )
    np.testing.assert_allclose(transition_matrix, expected.cumsum(axis=1))
    assert transition_matrix.dtype == DTYPE_GROUP_TRANSITION_PROBABILITIES


@pytest.mark.unit
def test_einsum_kronecker_product_threefold():
    # Three-fold Kronecker product.
    trans_mats = [np.random.uniform(0, 1, size=(2, 2)) for _ in range(3)]

    expected = np.kron(np.kron(*trans_mats[:2]), trans_mats[2])
    calculated = _einsum_kronecker_product(*trans_mats)

    np.testing.assert_allclose(calculated, expected, rtol=1e-06)
    assert calculated.dtype == DTYPE_GROUP_TRANSITION_PROBABILITIES


@pytest.mark.unit
def test_einsum_kronecker_product_fourfold():
    # Four-fold Kronecker product.
    trans_mats = [np.random.uniform(0, 1, size=(2, 2)) for _ in range(4)]

    expected = np.kron(np.kron(np.kron(*trans_mats[:2]), trans_mats[2]), trans_mats[3])
    calculated = _einsum_kronecker_product(*trans_mats)

    np.testing.assert_allclose(calculated, expected, rtol=1e-06)
    assert calculated.dtype == DTYPE_GROUP_TRANSITION_PROBABILITIES


@pytest.mark.unit
@pytest.mark.parametrize(
    "own_prob, group_names, expectation, expected",
    [
        (0.5, None, pytest.raises(ValueError, match="Pass either"), None),
    ],
)
def test_create_transition_matrix_own_prob(
    own_prob, group_names, expectation, expected
):
    with expectation:
        result = _create_transition_matrix_from_own_prob(own_prob, group_names)
        assert np.allclose(result, expected)
