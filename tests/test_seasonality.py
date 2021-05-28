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
    "model, params, dates, expectation, expected, contact_models",
    [
        pytest.param(
            None,
            None,
            pd.date_range("2020-01-01", periods=2),
            does_not_raise(),
            pd.DataFrame(
                index=pd.date_range("2020-01-01", periods=2),
                data=1,
                columns=["meet_two_people"],
            ).astype(float),
            {"meet_two_people": {}},
        ),
        pytest.param(
            lambda params, dates, seed: pd.Series(index=dates, data=[1 / 3, 2 / 3, 1]),
            None,
            pd.date_range("2020-01-01", periods=3),
            does_not_raise(),
            pd.DataFrame(
                index=pd.date_range("2020-01-01", periods=3),
                data=np.array([1, 2, 3]) / 3,
                columns=["meet_two_people"],
            ),
            {"meet_two_people": {}},
        ),
        pytest.param(
            lambda params, dates, seed: None,
            None,
            pd.date_range("2020-01-01", periods=2),
            pytest.raises(ValueError, match="'seasonality_factor_model'"),
            None,
            {"meet_two_people": {}},
        ),
        pytest.param(
            lambda params, dates, seed: pd.Series(
                index=pd.date_range("2020-01-01", periods=2), data=[-1, 2]
            ),
            None,
            pd.date_range("2020-01-01", periods=2),
            pytest.raises(ValueError, match="The seasonality factors"),
            None,
            {"meet_two_people": {}},
        ),
    ],
)
def test_prepare_seasonality_factor(
    model, params, dates, expectation, expected, contact_models
):
    with expectation:
        result = prepare_seasonality_factor(
            model, params, dates, itertools.count(), contact_models
        )
        assert result.equals(expected)


@pytest.mark.unit
def test_prepare_seasonality_factor_with_dataframe_return():
    def _model(params, dates, seed):
        df = pd.DataFrame(index=dates)
        df["households"] = [0.8, 0.8, 1]
        df["leisure"] = [0.5, 0.5, 1]
        return df

    result = prepare_seasonality_factor(
        seasonality_factor_model=_model,
        params=None,
        dates=pd.date_range("2021-04-01", "2021-04-03"),
        seed=itertools.count(),
        contact_models={"households": {}, "leisure": {}, "work": {}},
    )
    expected = pd.DataFrame(
        data=[[0.8, 0.5, 1.0]] * 2 + [[1, 1, 1]],
        columns=["households", "leisure", "work"],
        index=pd.date_range("2021-04-01", "2021-04-03"),
    )

    pd.testing.assert_frame_equal(result, expected)
