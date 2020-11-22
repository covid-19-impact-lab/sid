import numpy as np
import pandas as pd
import pytest
from sid.config import SID_TIME_START
from sid.time import period_to_timestamp
from sid.time import sid_period_to_timestamp
from sid.time import timestamp_to_period
from sid.time import timestamp_to_sid_period


@pytest.mark.unit
@pytest.mark.parametrize(
    "period, relative_to, expected",
    [
        (0, SID_TIME_START, SID_TIME_START),
        (1, "2020-01-01", pd.Timestamp("2020-01-02")),
        (-1, "2020-01-01", pd.Timestamp("2019-12-31")),
        ([2, 3], "2020-01-01", pd.to_datetime(["2020-01-03", "2020-01-04"])),
    ],
)
def test_period_to_timestamp(period, relative_to, expected):
    result = period_to_timestamp(period, relative_to)
    if np.isscalar(period):
        assert result == expected
    else:
        assert (result == expected).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "period, expected",
    [
        (0, SID_TIME_START),
        (1, pd.Timestamp("2019-01-02")),
        (-1, pd.Timestamp("2018-12-31")),
        ([2, 3], pd.to_datetime(["2019-01-03", "2019-01-04"])),
    ],
)
def test_sid_period_to_timestamp(period, expected):
    result = sid_period_to_timestamp(period)
    if np.isscalar(period):
        assert result == expected
    else:
        assert (result == expected).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "timestamp, relative_to, expected",
    [
        ("2019-01-01", SID_TIME_START, 0),
        ("2020-01-01", "2020-01-01", 0),
        ("2019-12-31", "2020-01-01", -1),
        (["2020-01-03", "2020-01-04"], "2020-01-01", [2, 3]),
    ],
)
def test_timestamp_to_period(timestamp, relative_to, expected):
    result = timestamp_to_period(timestamp, relative_to)
    if np.isscalar(timestamp):
        assert result == expected
    else:
        assert (result == expected).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "timestamp, expected",
    [
        (SID_TIME_START, 0),
        ("2019-01-02", 1),
        ("2018-12-31", -1),
        (["2019-01-03", "2019-01-04"], [2, 3]),
    ],
)
def test_timestamp_to_sid_period(timestamp, expected):
    result = timestamp_to_sid_period(timestamp)
    if np.isscalar(timestamp):
        assert result == expected
    else:
        assert (result == expected).all()
