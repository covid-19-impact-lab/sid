import functools

import numpy as np
import pandas as pd
from sid.config import SID_TIME_START


def period_to_timestamp(period, relative_to):
    return pd.to_datetime(relative_to) + pd.to_timedelta(period, unit="d")


def timestamp_to_period(timestamp, relative_to):
    return np.int16((pd.to_datetime(timestamp) - pd.to_datetime(relative_to)).days)


sid_period_to_timestamp = functools.partial(
    period_to_timestamp, relative_to=SID_TIME_START
)
timestamp_to_sid_period = functools.partial(
    timestamp_to_period, relative_to=SID_TIME_START
)
