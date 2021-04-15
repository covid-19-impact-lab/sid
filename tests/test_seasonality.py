import itertools
from contextlib import ExitStack as does_not_raise  # noqa: N813

import numpy as np
import pandas as pd
import pytest
from resources import CONTACT_MODELS
from sid import get_simulate_func
from sid.seasonality import prepare_seasonality_factor


@pytest.mark.end_to_end
def test_simulate_a_simple_model(params, initial_states, tmp_path):
    simulate = get_simulate_func(
        params=params,
        initial_states=initial_states,
        contact_models=CONTACT_MODELS,
        seasonality_factor_model=lambda params, dates, seed: pd.Series(
            index=dates, data=1
        ),
        saved_columns={"other": ["channel_infected_by_contact"]},
        path=tmp_path,
        seed=144,
    )

    result = simulate(params)

    time_series = result["time_series"].compute()
    last_states = result["last_states"].compute()

    for df in [time_series, last_states]:
        assert isinstance(df, pd.DataFrame)
        assert set(df["channel_infected_by_contact"].cat.categories) == {
            "not_infected_by_contact",
            "standard",
        }


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
