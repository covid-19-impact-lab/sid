import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sid.matching_probabilities import _create_transition_matrix_from_own_prob
from sid.matching_probabilities import _einsum_kronecker_product
from sid.matching_probabilities import _join_transition_matrices
from sid.matching_probabilities import create_group_transition_probs


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


@pytest.mark.parametrize(
    "assort_by, expected",
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
def test_create_group_transition_probs(initial_states, assort_by, params, expected):
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

    transition_matrix = create_group_transition_probs(
        states=initial_states, assort_by=assort_by, params=params, model_name="model1"
    )
    np.testing.assert_allclose(transition_matrix, expected.cumsum(axis=1))


def test_einsum_kronecker_product():
    # Three-fold Kronecker product.
    trans_mats = [np.random.uniform(0, 1, size=(2, 2)) for _ in range(3)]

    expected = np.kron(np.kron(*trans_mats[:2]), trans_mats[2])
    calculated = _einsum_kronecker_product(*trans_mats).astype("float32")

    np.testing.assert_allclose(calculated, expected, rtol=1e-06)

    # Four-fold Kronecker product.
    trans_mats = [np.random.uniform(0, 1, size=(2, 2)) for _ in range(4)]

    expected = np.kron(np.kron(np.kron(*trans_mats[:2]), trans_mats[2]), trans_mats[3])
    calculated = _einsum_kronecker_product(*trans_mats).astype("float32")

    np.testing.assert_allclose(calculated, expected, rtol=1e-06)
