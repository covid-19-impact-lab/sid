import pandas as pd
from sid.config import SID_TIME_START


def sid_time_to_timestamp(sid_time):
    return SID_TIME_START + pd.to_timedelta(sid_time, unit="d")
