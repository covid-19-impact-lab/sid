from contextlib import ExitStack as does_not_raise  # noqa: N813

import numpy as np
import pandas as pd
import pytest
from sid.config import INITIAL_CONDITIONS
from sid.parse_model import parse_duration
from sid.parse_model import parse_initial_conditions


@pytest.mark.unit
@pytest.mark.parametrize(
    "duration, expectation, expected",
    [
        (
            {"start": "2020-01-01", "end": "2020-01-02"},
            does_not_raise(),
            {
                "start": pd.Timestamp("2020-01-01"),
                "end": pd.Timestamp("2020-01-02"),
                "dates": pd.DatetimeIndex(pd.to_datetime(["2020-01-01", "2020-01-02"])),
            },
        ),
        (
            {"start": "2020-01-01", "periods": 2},
            does_not_raise(),
            {
                "start": pd.Timestamp("2020-01-01"),
                "end": pd.Timestamp("2020-01-02"),
                "dates": pd.DatetimeIndex(pd.to_datetime(["2020-01-01", "2020-01-02"])),
            },
        ),
        (
            {"start": "2020-01-01", "periods": 2, "freq": "s"},
            pytest.warns(UserWarning, match="Only 'start', 'end', and 'periods'"),
            {
                "start": pd.Timestamp("2020-01-01"),
                "end": pd.Timestamp("2020-01-02"),
                "dates": pd.DatetimeIndex(pd.to_datetime(["2020-01-01", "2020-01-02"])),
            },
        ),
        ({"periods": 2}, pytest.raises(ValueError, match="Of the four"), None),
    ],
)
def test_parse_duration(duration, expectation, expected):
    with expectation:
        result = parse_duration(duration)
        for k in result:
            if k == "dates":
                assert np.all(result[k] == expected[k])
            else:
                assert result[k] == expected[k]


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "initial_conditions",
        "start_date_simulation",
        "virus_strains",
        "expectation",
        "expected",
    ),
    [
        (
            None,
            pd.Timestamp("2020-01-02"),
            {"names": ["base_strain"], "multipliers": np.ones(1)},
            does_not_raise(),
            {**INITIAL_CONDITIONS},
        ),
        (
            {"assort_by": ["region"]},
            pd.Timestamp("2020-01-02"),
            {"names": ["base_strain"], "multipliers": np.ones(1)},
            does_not_raise(),
            {**INITIAL_CONDITIONS, "assort_by": ["region"]},
        ),
        (
            {"assort_by": "region"},
            pd.Timestamp("2020-01-02"),
            {"names": ["base_strain"], "multipliers": np.ones(1)},
            does_not_raise(),
            {**INITIAL_CONDITIONS, "assort_by": ["region"]},
        ),
        (
            {"growth_rate": 0},
            pd.Timestamp("2020-01-02"),
            {"names": ["base_strain"], "multipliers": np.ones(1)},
            pytest.raises(ValueError, match="'growth_rate' must be greater than or"),
            None,
        ),
        (
            {"burn_in_periods": 0},
            pd.Timestamp("2020-01-02"),
            {"names": ["base_strain"], "multipliers": np.ones(1)},
            pytest.raises(ValueError, match="'burn_in_periods' must be greater or"),
            None,
        ),
        (
            {"burn_in_periods": 2.0},
            pd.Timestamp("2020-01-02"),
            {"names": ["base_strain"], "multipliers": np.ones(1)},
            pytest.raises(ValueError, match="'burn_in_periods' must be an integer"),
            None,
        ),
        (
            {"initial_infections": None},
            pd.Timestamp("2020-01-02"),
            {"names": ["base_strain"], "multipliers": np.ones(1)},
            pytest.raises(ValueError, match="'initial_infections' must be a"),
            None,
        ),
    ],
)
def test_parse_initial_conditions(
    initial_conditions, start_date_simulation, virus_strains, expectation, expected
):
    with expectation:
        result = parse_initial_conditions(
            initial_conditions, start_date_simulation, virus_strains
        )
        expected["burn_in_periods"] = pd.DatetimeIndex([pd.Timestamp("2020-01-01")])
        assert result == expected
