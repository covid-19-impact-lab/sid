import pandas as pd


def parse_duration(duration=None):
    """Parse the user-defined duration.

    Args:
        duration (dict): Duration is a dictionary containing kwargs for
            :func:`pandas.date_range`.

    Returns:
        new_duration (dict): A dictionary containing start and end periods or dates and
            an iterable of the same types.

    Examples:
        >>> parse_duration({"start": "2020-03-01", "end": "2020-03-10"})
        {'start': Timestamp('2020-03-01 00:00:00', freq='D'), 'end': ...

    """
    if duration is None:
        duration = {"start": "2020-01-27", "periods": 10}

    iterable = pd.date_range(**duration)

    internal_duration = {}
    internal_duration["start"] = iterable[0]
    internal_duration["end"] = iterable[-1]
    internal_duration["dates"] = iterable

    return internal_duration
