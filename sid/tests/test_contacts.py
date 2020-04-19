import numpy as np
import pytest

from sid.contacts import create_group_indexer
from sid.contacts import create_group_transition_probs


@pytest.mark.parametrize(
    "assort_by, expected",
    [
        (
            ["age_group", "region"],
            [[9, 12], [7, 10, 13], [8, 11, 14], [0, 3, 6], [1, 4], [2, 5]],
        ),
        ([], [list(range(15))]),
    ],
)
def test_create_group_indexer(initial_states, assort_by, expected):
    calculated = create_group_indexer(initial_states, assort_by)
    calculated = [arr.tolist() for arr in calculated]

    assert calculated == expected


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
    transition_matrix = create_group_transition_probs(initial_states, assort_by, params)

    np.testing.assert_allclose(transition_matrix, expected)
