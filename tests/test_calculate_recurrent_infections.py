import numba as nb
import numpy as np
from sid.contacts import _calculate_infections_by_recurrent_contacts


def test_recurrent_contact_infects_susceptibles_and_leaves_other_group_untouched():
    recurrent_contacts = np.array([True, True, False, True]).reshape(-1, 1)
    infectious = np.array([True, False, False, False])
    immune = np.array([True, False, False, False])

    group_codes = np.array([0, 0, 0, 1]).reshape(-1, 1)

    sub_indexers = nb.typed.List()
    sub_indexers.append(np.array([0, 1, 2]))
    sub_indexers.append(np.array([3]))
    indexers = nb.typed.List()
    indexers.append(sub_indexers)

    infection_probabilities = np.array([1])
    infected = np.array([False, False, False, False])
    infection_counter = np.zeros(4, dtype=np.int_)
    infection_probability_multiplier = np.ones(len(recurrent_contacts))

    (
        infected,
        infection_counter,
        immune,
        was_infected,
    ) = _calculate_infections_by_recurrent_contacts(
        recurrent_contacts,
        infectious,
        immune,
        group_codes,
        indexers,
        infection_probabilities,
        infection_probability_multiplier,
        infected,
        infection_counter,
        0,
    )

    assert (infected == [False, True, False, False]).all()
    assert (infection_counter == [1, 0, 0, 0]).all()
    assert (immune == [True, True, False, False]).all()
    assert (was_infected == [-1, 0, -1, -1]).all()


def test_infections_occur_not_in_other_recurrent_group():
    recurrent_contacts = np.array([True, True, True, True]).reshape(-1, 1)
    infectious = np.array([True, False, False, False])
    immune = np.array([True, False, False, False])

    group_codes = np.array([0, 0, 1, 1]).reshape(-1, 1)

    sub_indexers = nb.typed.List()
    sub_indexers.append(np.array([0, 1]))
    sub_indexers.append(np.array([2, 3]))
    indexers = nb.typed.List()
    indexers.append(sub_indexers)

    infection_probabilities = np.array([1])
    infected = np.array([False, True, False, False])
    infection_counter = np.zeros(4, dtype=np.int_)
    infection_probability_multiplier = np.ones(len(recurrent_contacts))

    (
        infected,
        infection_counter,
        immune,
        was_infected,
    ) = _calculate_infections_by_recurrent_contacts(
        recurrent_contacts,
        infectious,
        immune,
        group_codes,
        indexers,
        infection_probabilities,
        infection_probability_multiplier,
        infected,
        infection_counter,
        0,
    )

    assert (infected == [False, True, False, False]).all()
    assert (infection_counter == [1, 0, 0, 0]).all()
    assert (immune == [True, True, False, False]).all()
    assert (was_infected == [-1, 0, -1, -1]).all()
