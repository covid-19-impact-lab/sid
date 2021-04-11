from contextlib import ExitStack as does_not_raise  # noqa: N813

import numpy as np
import pandas as pd
import pytest
from sid.shared import boolean_choice
from sid.shared import factorize_assortative_variables
from sid.shared import random_choice


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


@pytest.mark.unit
@pytest.mark.parametrize("x, expected", [(0, False), (1, True)])
def test_boolean_choice(x, expected):
    result = boolean_choice(x)
    assert result is expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "choices, probs, seed, expectation, expected",
    [
        (3, [0, 0, 1], None, does_not_raise(), 2),
        ([0, 1, 2], [0, 1, 0], None, does_not_raise(), 1),
        (np.arange(3), [1, 0, 0], None, does_not_raise(), 0),
        (3, None, 0, does_not_raise(), 1),
        (3, pd.Series([1, 0, 0]), None, does_not_raise(), 0),
        (3, np.array([1, 0, 0]), None, does_not_raise(), 0),
        (
            3,
            [0.2, 0.2, 0.2],
            None,
            pytest.raises(ValueError, match="Probabilities do"),
            0,
        ),
        (np.arange(6).reshape(2, 3), None, 0, does_not_raise(), [1, 2]),
    ],
)
def test_random_choice(choices, probs, seed, expectation, expected):
    with expectation:
        np.random.seed(seed)
        result = random_choice(choices, probs)
        if np.isscalar(expected):
            assert result == expected
        else:
            assert (result == expected).all()
