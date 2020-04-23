import pandas as pd


def parse_duration(duration):
    """Parse the user-defined duration.

    Args:
        duration (int or dict): Duration can be an integer which will simulate data for
            `range(0, duration)` periods. It can also be a dictionary containing kwargs
            for :func:`pandas.date_range`.

    Returns:
        new_duration (dict): A dictionary containing start and end periods or dates and
            an iterable of the same types.

    Examples:
        >>> parse_duration(10)
        {'start': 0, 'end': 10, 'iterable': range(0, 10), 'column_name': 'period'}

        >>> parse_duration({"start": "2020-03-01", "end": "2020-03-10"})
        {'start': datetime.date(2020, 3, 1), 'end': datetime.date(2020, 3, 10), ...

    """
    new_duration = {}
    if isinstance(duration, int) and duration >= 0:
        new_duration["start"] = 0
        new_duration["end"] = duration
        new_duration["iterable"] = range(0, duration)
        new_duration["column_name"] = "period"

    elif isinstance(duration, dict):
        iterable = pd.date_range(**duration).date
        new_duration["start"] = iterable[0]
        new_duration["end"] = iterable[-1]
        new_duration["iterable"] = iterable
        new_duration["column_name"] = "date"

    else:
        raise ValueError(
            "duration must be a positive integer or kwargs for pd.date_range."
        )

    return new_duration


def parse_period_or_date(period_or_date):
    """Parse period or date.

    Examples:
        >>> parse_period_or_date(1)
        1

        >>> parse_period_or_date("2020-03-01")
        datetime.date(2020, 3, 1)

    """
    if isinstance(period_or_date, int):
        pass
    else:
        period_or_date = pd.to_datetime(period_or_date).date()

    return period_or_date
