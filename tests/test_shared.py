import pytest
from sid.shared import factorize_assortative_variables


@pytest.mark.unit
@pytest.mark.parametrize(
    ("assort_by", "expected"),
    [
        (
            ["age_group", "region"],
            [
                ("Over 50", "a"),
                ("Over 50", "b"),
                ("Over 50", "c"),
                ("Under 50", "a"),
                ("Under 50", "b"),
                ("Under 50", "c"),
            ],
        ),
        ([], [(0,)]),
    ],
)
def test_factorize_assortative_variables(initial_states, assort_by, expected):
    _, group_code_values = factorize_assortative_variables(
        initial_states, assort_by, False
    )

    assert set(group_code_values) == set(expected)
