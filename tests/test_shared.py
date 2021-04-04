import numpy as np
import pytest
from sid.shared import factorize_assortative_variables


NP_ARRAY_WITH_SINGLE_TUPLE = np.empty(1, dtype=object)
NP_ARRAY_WITH_SINGLE_TUPLE[0] = (0,)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("assort_by", "expected_codes", "expected_uniques"),
    [
        (["age_group"], [1] * 7 + [0] * 8, ["Over 50", "Under 50"]),
        (
            ["age_group", "region"],
            [3, 4, 5, 3, 4, 5, 3, 1, 2, 0, 1, 2, 0, 1, 2],
            np.array(
                [
                    ("Over 50", "a"),
                    ("Over 50", "b"),
                    ("Over 50", "c"),
                    ("Under 50", "a"),
                    ("Under 50", "b"),
                    ("Under 50", "c"),
                ],
                dtype="U8, U8",
            ).astype(object),
        ),
        ([], np.zeros(15), NP_ARRAY_WITH_SINGLE_TUPLE),
    ],
)
def test_factorize_assortative_variables(
    initial_states, assort_by, expected_codes, expected_uniques
):
    codes, uniques = factorize_assortative_variables(initial_states, assort_by, None)
    assert (codes == expected_codes).all()
    assert (uniques == expected_uniques).all()
