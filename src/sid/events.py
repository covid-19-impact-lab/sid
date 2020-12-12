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
    infected_by_event = pd.Series(index=states.index, data=False)
    channel_infected_by_event = pd.Series(index=states.index, data=-1)

    for i, event in enumerate(events.values()):
        loc = event.get("loc", params.index)
        func = event["model"]

        s = func(states, params.loc[loc])
        infected_by_event = infected_by_event | s
        channel_infected_by_event.loc[s & channel_infected_by_event.eq(-1)] = i

    codes_to_event = {-1: "not_infected_by_event", **dict(enumerate(events))}
    channel_infected_by_event = pd.Series(
        pd.Categorical(channel_infected_by_event, categories=list(codes_to_event)),
        index=states.index,
    ).cat.rename_categories(codes_to_event)

    return infected_by_event, channel_infected_by_event
