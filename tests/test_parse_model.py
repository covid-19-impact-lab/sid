import pytest
from sid.parse_model import parse_duration
from contextlib import ExitStack as does_not_raise  # noqa: N813
import pandas as pd
import numpy as np


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
