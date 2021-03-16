import functools
import itertools

import numpy as np
import pandas as pd
import pytest
from sid.events import calculate_infections_by_events


def event_infect_n(states, params, seed, i):  # noqa: U100
    s = pd.Series(index=states.index, data=False)
    s.iloc[i] = True

    return s


@pytest.mark.integration
def test_no_events_combined_with_infections_by_contact(initial_states, params):
    virus_strains = {"names": ["base_strain"], "factors": np.ones(1)}

    infections_by_events, was_infected_by_event = calculate_infections_by_events(
        initial_states, params, {}, virus_strains, itertools.count()
    )

    assert (infections_by_events == -1).all()
    assert was_infected_by_event.eq("not_infected_by_event").all()


@pytest.mark.integration
def test_calculate_infections_by_events(initial_states, params):
    events = {
        "infect_first": {"model": functools.partial(event_infect_n, i=0)},
        "infect_second": {"model": functools.partial(event_infect_n, i=1)},
    }

    virus_strains = {"names": ["base_strain"], "factors": np.ones(1)}

    infections, was_infected_by_event = calculate_infections_by_events(
        initial_states, params, events, virus_strains, itertools.count()
    )

    expected = pd.Series(data=-1, index=initial_states.index, dtype="int8")
    expected.iloc[:2] = 0

    assert infections.equals(expected)
    assert (
        was_infected_by_event.cat.categories
        == [
            "not_infected_by_event",
            "infect_first",
            "infect_second",
        ]
    ).all()
