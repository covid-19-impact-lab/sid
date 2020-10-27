import functools

import pandas as pd

from sid.events import calculate_infections_by_events


def event_infect_n(states, params, i):  # noqa: U100
    s = pd.Series(index=states.index, data=False)
    s.iloc[i] = True

    return s


def test_no_events_combined_with_infections_by_contact(initial_states, params):
    infections_by_events = calculate_infections_by_events(initial_states, params, {})

    infections_by_contacts = pd.Series(data=False, index=initial_states.index)
    infections_by_contacts.iloc[:2] = True

    combined_infections = infections_by_contacts | infections_by_events

    assert infections_by_contacts.equals(combined_infections)


def test_calculate_infections_by_events(initial_states, params):
    events = {
        "infect_first": {"model": functools.partial(event_infect_n, i=0)},
        "infect_second": {"model": functools.partial(event_infect_n, i=1)},
    }

    infections = calculate_infections_by_events(initial_states, params, events)

    expected = pd.Series(data=False, index=initial_states.index)
    expected.iloc[:2] = True

    assert infections.equals(expected)
