import pytest

from sid.contacts import create_group_indexer


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
