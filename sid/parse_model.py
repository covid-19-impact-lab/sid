import pandas as pd


def parse_duration(duration):
    """Parse the user-defined duration.

    Args:
        duration (dict): Duration is a dictionary containing kwargs for
            :func:`pandas.date_range`.

    Returns:
        new_duration (dict): A dictionary containing start and end periods or dates and
            an iterable of the same types.

    Examples:
        >>> parse_duration({"start": "2020-03-01", "end": "2020-03-10"})
        {'start': datetime.date(2020, 3, 1), 'end': datetime.date(2020, 3, 10), ...

    """
    if duration is None:
        duration = {"start": "2020-02-01", "periods": 10}

    iterable = pd.date_range(**duration)

    internal_duration = {}
    internal_duration["start"] = iterable[0]
    internal_duration["end"] = iterable[-1]
    internal_duration["dates"] = iterable

    return internal_duration
