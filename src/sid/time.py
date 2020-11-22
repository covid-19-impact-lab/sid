"""This module contains functions for handling sid's internal period.

sid's internal period is similar to Unix time, but the reference date is 2019-01-01
instead of 1970-01-01 and it is not measured in seconds but days. This allows to use a
int16 instead of int32 for Unix time.

The internal period is used to store dates more efficiently as a int16 instead of the
normal datetime64.

The advantage of this approach over enumerating periods passed via the ``duration``
argument of :func:`~sid.simulate.get_simulate_func` is that there is still information
on the exact dates in the states even if the ``"date"`` column is removed during
estimation to reduce memory consumption.

"""

from functools import partial

import numpy as np
import pandas as pd
from sid.config import SID_TIME_START, DTYPE_SID_PERIOD


def period_to_timestamp(period, relative_to):
    return pd.to_datetime(relative_to) + pd.to_timedelta(period, unit="d")


def timestamp_to_period(timestamp, relative_to):
    return DTYPE_SID_PERIOD(
        (pd.to_datetime(timestamp) - pd.to_datetime(relative_to)).days
    )


sid_period_to_timestamp = partial(period_to_timestamp, relative_to=SID_TIME_START)
timestamp_to_sid_period = partial(timestamp_to_period, relative_to=SID_TIME_START)
