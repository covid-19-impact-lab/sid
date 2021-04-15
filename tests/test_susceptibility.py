import itertools
from contextlib import ExitStack as does_not_raise  # noqa: N813

import numpy as np
import pandas as pd
import pytest
from sid.susceptibility import prepare_susceptibility_factor


@pytest.mark.unit
@pytest.mark.parametrize(
    "model, states, expectation, expected",
    [
        (None, [0] * 100, does_not_raise(), np.ones(100)),
        (
            lambda *x: True,
            None,
            pytest.raises(ValueError, match="'susceptibility_factor_model"),
            None,
        ),
        (
            lambda *x: pd.Series([1]),
            [1, 1],
            pytest.raises(ValueError, match="The 'susceptibility_factor"),
            None,
        ),
        (lambda *x: pd.Series([1]), [1], does_not_raise(), [1]),
        (lambda *x: np.ones(1), [1], does_not_raise(), [1]),
        (lambda *x: np.array([1, 2]), [1, 1], does_not_raise(), [0.5, 1]),
    ],
)
def test_prepare_susceptibility_factor(model, states, expectation, expected):
    with expectation:
        result = prepare_susceptibility_factor(model, states, None, itertools.count())
        assert (result == expected).all()
