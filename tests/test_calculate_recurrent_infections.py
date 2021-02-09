import numba as nb
import numpy as np
from sid.contacts import _calculate_infections_by_recurrent_contacts


def test_recurrent_contact_does_not_meet():
    recurrent_contacts = np.array([False, True, True]).reshape(-1, 1)
    infectious = np.array([False, True, False])
    immune = np.array([False, True, False])

    group_codes = np.zeros((3, 1), dtype=np.int_)

    sub_indexers = nb.typed.List()
    sub_indexers.append(np.array([0, 1, 2]))
    indexers = nb.typed.List()
    indexers.append(sub_indexers)

    infection_probabilities = np.array([1])
    infected = np.array([False, False, False])
    infection_counter = np.zeros(3, dtype=np.int_)

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
        infected,
        infection_counter,
        0,
    )

    assert (infected == [False, False, True]).all()
    assert (infection_counter == [0, 1, 0]).all()
    assert (immune == [False, True, True]).all()
    assert (was_infected == [-1, -1, 0]).all()


def test_infections_occur_in_other_recurrent_group():
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
        infected,
        infection_counter,
        0,
    )

    assert (infected == [False, True, False, False]).all()
    assert (infection_counter == [1, 0, 0, 0]).all()
    assert (immune == [True, True, False, False]).all()
    assert (was_infected == [-1, 0, -1, -1]).all()
