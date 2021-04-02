import numpy as np
import pandas as pd
import pytest
from sid.statistics import calculate_r_effective
from sid.statistics import calculate_r_zero


@pytest.fixture()
def data_for_a_single_day():
    df = pd.DataFrame(
        {
            "cd_infectious_false": [-2, -1, -1, 0] + [2, 5],
            "n_has_infected": [2, 1, 1, 1] + [0, 0],
            "immune": [True] * 4 + [False, False],
        }
    )
    return df


@pytest.mark.unit
@pytest.mark.parametrize("window, expected", [(2, 5 / 4), (1, 1)])
def test_r_effective_single_day(data_for_a_single_day, window, expected):
    result = calculate_r_effective(data_for_a_single_day, window)
    assert result == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "window, expected", [(2, (5 / 4) / (2 / 6)), (1, (3 / 3) / (1 / 3))]
)
def test_r_zero_single_day(data_for_a_single_day, window, expected):
    result = calculate_r_zero(data_for_a_single_day, window)
    assert np.isclose(result, expected)


@pytest.fixture()
def data_for_multiple_days():
    df = pd.DataFrame(
        {
            "cd_infectious_false": [-2, -1, -1, 0] + [2, 5],
            "n_has_infected": [2, 1, 1, 1] + [0, 0],
            "immune": [True] * 4 + [False, False],
            "date": pd.to_datetime(["2020-03-09"] * 3 + ["2020-03-10"] * 3),
        }
    )
    return df


@pytest.mark.unit
@pytest.mark.parametrize("window, expected", [(2, [4 / 3, 1]), (1, [1, 1])])
def test_r_effective_multiple_days(data_for_multiple_days, window, expected):
    result = calculate_r_effective(data_for_multiple_days, window)
    assert (result == expected).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "window, expected", [(2, [np.inf, 1 / (2 / 3)]), (1, [np.inf, 1 / (2 / 3)])]
)
def test_r_zero_multiple_days(data_for_multiple_days, window, expected):
    result = calculate_r_zero(data_for_multiple_days, window)
    assert np.allclose(result, expected)
