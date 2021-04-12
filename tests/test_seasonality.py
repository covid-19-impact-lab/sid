import itertools
from contextlib import ExitStack as does_not_raise  # noqa: N813

import numpy as np
import pandas as pd
import pytest
from sid.seasonality import prepare_seasonality_factor


@pytest.mark.unit
@pytest.mark.parametrize(
    "model, params, dates, expectation, expected",
    [
        pytest.param(
            None,
            None,
            pd.date_range("2020-01-01", periods=2),
            does_not_raise(),
            pd.Series(index=pd.date_range("2020-01-01", periods=2), data=1),
        ),
        pytest.param(
            lambda params, dates, seed: pd.Series(index=dates, data=[1, 2, 3]),
            None,
            pd.date_range("2020-01-01", periods=3),
            does_not_raise(),
            pd.Series(
                index=pd.date_range("2020-01-01", periods=3),
                data=np.array([1, 2, 3]) / 3,
            ),
        ),
        pytest.param(
            lambda params, dates, seed: np.ones(len(dates)),
            None,
            pd.date_range("2020-01-01", periods=2),
            does_not_raise(),
            pd.Series(index=pd.date_range("2020-01-01", periods=2), data=1.0),
        ),
        pytest.param(
            lambda params, dates, seed: None,
            None,
            pd.date_range("2020-01-01", periods=2),
            pytest.raises(ValueError, match="'seasonality_factor_model'"),
            None,
        ),
        pytest.param(
            lambda params, dates, seed: pd.Series(
                index=pd.date_range("2020-01-01", periods=2), data=[-1, 2]
            ),
            None,
            pd.date_range("2020-01-01", periods=2),
            pytest.raises(ValueError, match="The seasonality factors"),
            None,
        ),
    ],
)
def test_prepare_seasonality_factor(model, params, dates, expectation, expected):
    with expectation:
        result = prepare_seasonality_factor(model, params, dates, itertools.count())
        assert result.equals(expected)
