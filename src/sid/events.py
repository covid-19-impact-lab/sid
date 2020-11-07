import pandas as pd


def calculate_infections_by_events(states, params, events):
    """Apply events to states and return indicator for infections.

    Each event is evaluated which yields a collection of series with indicators for
    infected people. All events are merged with the logical OR.

    Args:
        states (pandas.DataFrame): See :ref:`states`.
        params (pandas.DataFrame): See :ref:`params`.
        events (dict): Dictionary of events which cause infections.

    Returns:
        newly_infected_events (pandas.Series): Series marking individuals who have been
            infected through an event. The index is the same as states, values are
            boolean. `True` marks individuals infected by an event.

    """
    infections_by_events = pd.DataFrame(index=states.index)

    for name, event in events.items():
        loc = event.get("loc", params.index)
        func = event["model"]

        infections_by_events[name] = func(states, params.loc[loc])

    newly_infected_events = infections_by_events.any(axis=1)

    return newly_infected_events
