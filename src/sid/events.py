"""This module contains the code to calculate infections by events."""
import pandas as pd
from sid.config import DTYPE_VIRUS_STRAIN
from sid.virus_strains import combine_first_factorized_infections
from sid.virus_strains import factorize_boolean_or_categorical_infections


def calculate_infections_by_events(states, params, events, virus_strains, seed):
    """Apply events to states and return indicator for infections.

    Each event is evaluated which yields a collection of series with indicators for
    infected people. All events are merged with the logical OR.

    Args:
        states (pandas.DataFrame): See :ref:`states`.
        params (pandas.DataFrame): See :ref:`params`.
        events (dict): Dictionary of events which cause infections.
        virus_strains (Dict[str, Any]): A dictionary with the keys ``"names"`` and
            ``"factors"`` holding the different contagiousness factors of multiple
            viruses.
        seed (itertools.count): The seed counter.

    Returns:
        newly_infected_events (pandas.Series): Series marking individuals who have been
            infected through an event. The index is the same as states, values are
            boolean. `True` marks individuals infected by an event.

    """
    infected_by_event = pd.Series(index=states.index, data=-1, dtype=DTYPE_VIRUS_STRAIN)
    channel_infected_by_event = pd.Series(index=states.index, data=-1)

    for i, event in enumerate(events.values()):
        loc = event.get("loc", params.index)
        func = event["model"]

        categorical_infections = func(states, params.loc[loc], next(seed))

        factorized_infections = factorize_boolean_or_categorical_infections(
            categorical_infections, virus_strains
        )
        infected_by_event = combine_first_factorized_infections(
            infected_by_event, factorized_infections
        )

        channel_infected_by_event.loc[
            factorized_infections >= 0 & channel_infected_by_event.eq(-1)
        ] = i

    codes_to_event = {-1: "not_infected_by_event", **dict(enumerate(events))}
    channel_infected_by_event = pd.Series(
        pd.Categorical(channel_infected_by_event, categories=list(codes_to_event)),
        index=states.index,
    ).cat.rename_categories(codes_to_event)

    infected_by_event = pd.Series(infected_by_event, index=states.index)

    return infected_by_event, channel_infected_by_event
