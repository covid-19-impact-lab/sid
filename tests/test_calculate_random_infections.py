import numba as nb
import numpy as np
from sid.contacts import _calculate_infections_by_random_contacts


def test_random_contact_infects_susceptibles():
    """Individual infects random contacts within its and across groups."""
    random_contacts = np.array([2, 1, 1]).reshape(-1, 1)
    infectious = np.array([True, False, False])
    immune = np.array([True, False, False])

    group_codes = np.array([0, 0, 1]).reshape(-1, 1)

    sub_indexers = nb.typed.List()
    sub_indexers.append(np.array([0, 1]))
    sub_indexers.append(np.array([2]))
    indexers = nb.typed.List()
    indexers.append(sub_indexers)

    infected = np.array([False, False, False])
    infection_counter = np.zeros(3, dtype=np.int_)

    assortative_matching_cum_probs = nb.typed.List()
    assortative_matching_cum_probs.append(np.array([0.8, 0.2, 1, 0]).reshape(2, 2))

    (
        infected,
        infection_counter,
        immune,
        missed,
        was_infected,
    ) = _calculate_infections_by_random_contacts(
        random_contacts,
        infectious,
        immune,
        group_codes,
        assortative_matching_cum_probs,
        indexers,
        infected,
        infection_counter,
        0,
    )

    assert (infected == [False, True, True]).all()
    assert (infection_counter == [2, 0, 0]).all()
    assert (immune == [True, True, True]).all()
    assert (missed == 0).all()
    assert (was_infected == [-1, 0, 0]).all()


def test_random_contact_immune_and_people_without_contacts_are_not_infected():
    """Infections do not occur for immune random contacts and those without contacts."""
    random_contacts = np.array([10, 10, 0, 10, 0]).reshape(-1, 1)
    infectious = np.array([True, False, False, False, False])
    immune = np.array([True, True, False, True, False])

    group_codes = np.array([0, 0, 0, 1, 1]).reshape(-1, 1)

    sub_indexers = nb.typed.List()
    sub_indexers.append(np.array([0, 1, 2]))
    sub_indexers.append(np.array([3, 4]))
    indexers = nb.typed.List()
    indexers.append(sub_indexers)

    infected = np.array([False, False, False, False, False])
    infection_counter = np.zeros(5, dtype=np.int_)

    assortative_matching_cum_probs = nb.typed.List()
    assortative_matching_cum_probs.append(np.array([0.5, 0.5, 0.5, 0.5]).reshape(2, 2))

    (
        infected,
        infection_counter,
        immune,
        missed,
        was_infected,
    ) = _calculate_infections_by_random_contacts(
        random_contacts,
        infectious,
        immune,
        group_codes,
        assortative_matching_cum_probs,
        indexers,
        infected,
        infection_counter,
        0,
    )

    assert (infected == [False, False, False, False, False]).all()
    assert (infection_counter == [0, 0, 0, 0, 0]).all()
    assert (immune == [True, True, False, True, False]).all()
    assert (missed[[0, 1, 3], :] > 0).all()
    assert (missed[[2, 4]] == 0).all()
    assert (was_infected == [-1, -1, -1, -1, -1]).all()