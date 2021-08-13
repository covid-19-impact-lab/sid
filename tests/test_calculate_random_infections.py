import numba as nb
import numpy as np
import pytest
from sid.contacts import _calculate_infections_by_random_contacts


@pytest.mark.unit
def test_random_contact_infects_susceptibles():
    """Individual infects random contacts within its and across groups."""
    random_contacts = np.array([2, 1, 1]).reshape(-1, 1)
    infectious = np.array([True, False, False])
    immunity_level = np.array([1.0, 0.0, 0.0])

    group_codes = np.array([0, 0, 1]).reshape(-1, 1)

    sub_indexers = nb.typed.List()
    sub_indexers.append(np.array([0, 1]))
    sub_indexers.append(np.array([2]))
    indexers = nb.typed.List()
    indexers.append(sub_indexers)

    infection_counter = np.zeros(3, dtype=np.int_)
    susceptibility_factor = np.ones(len(random_contacts))

    assortative_matching_cum_probs = nb.typed.List()
    assortative_matching_cum_probs.append(np.array([0.8, 0.2, 1, 0]).reshape(2, 2))

    virus_strain = np.array([0, -1, -1])
    contagiousness_factor = np.array([1])
    persistency_factor = np.array([1])

    (
        newly_infected,
        infection_counter,
        missed,
        was_infected,
    ) = _calculate_infections_by_random_contacts(
        random_contacts,
        infectious,
        immunity_level,
        virus_strain,
        group_codes,
        assortative_matching_cum_probs,
        indexers,
        susceptibility_factor,
        contagiousness_factor,
        persistency_factor,
        infection_counter,
        0,
    )

    assert (newly_infected == [-1, 0, 0]).all()
    assert (infection_counter == [2, 0, 0]).all()
    assert (missed == 0).all()
    assert (was_infected == [-1, 0, 0]).all()


@pytest.mark.unit
def test_random_contact_immune_and_people_without_contacts_are_not_infected():
    """Infections do not occur for immune random contacts and those without contacts."""
    random_contacts = np.array([10, 10, 0, 10, 0]).reshape(-1, 1)
    infectious = np.array([True, False, False, False, False])
    immunity_level = np.array([1.0, 1.0, 0.0, 1.0, 0.0])

    group_codes = np.array([0, 0, 0, 1, 1]).reshape(-1, 1)

    sub_indexers = nb.typed.List()
    sub_indexers.append(np.array([0, 1, 2]))
    sub_indexers.append(np.array([3, 4]))
    indexers = nb.typed.List()
    indexers.append(sub_indexers)

    infection_counter = np.zeros(5, dtype=np.int_)
    susceptibility_factor = np.ones(len(random_contacts))

    assortative_matching_cum_probs = nb.typed.List()
    assortative_matching_cum_probs.append(np.array([0.5, 0.5, 0.5, 0.5]).reshape(2, 2))

    virus_strain = np.array([0, -1, -1, -1, -1])
    contagiousness_factor = np.array([1])
    persistency_factor = np.array([1])

    (
        newly_infected,
        infection_counter,
        missed,
        was_infected,
    ) = _calculate_infections_by_random_contacts(
        random_contacts,
        infectious,
        immunity_level,
        virus_strain,
        group_codes,
        assortative_matching_cum_probs,
        indexers,
        susceptibility_factor,
        contagiousness_factor,
        persistency_factor,
        infection_counter,
        0,
    )

    assert (newly_infected == [-1, -1, -1, -1, -1]).all()
    assert (infection_counter == [0, 0, 0, 0, 0]).all()
    assert (missed[[0, 1, 3], :] > 0).all()
    assert (missed[[2, 4]] == 0).all()
    assert (was_infected == [-1, -1, -1, -1, -1]).all()


@pytest.mark.unit
def test_multiple_virus_strains_spread_in_different_random_groups():
    random_contacts = np.array([1, 1, 1, 1]).reshape(-1, 1)
    infectious = np.array([True, False, True, False])
    immunity_level = np.array([1.0, 0.0, 0.0, 0.0])

    group_codes = np.array([0, 0, 1, 1]).reshape(-1, 1)

    sub_indexers = nb.typed.List()
    sub_indexers.append(np.array([0, 1]))
    sub_indexers.append(np.array([2, 3]))
    indexers = nb.typed.List()
    indexers.append(sub_indexers)

    infection_counter = np.zeros(4, dtype=np.int_)
    susceptibility_factor = np.ones(len(random_contacts))

    assortative_matching_cum_probs = nb.typed.List()
    assortative_matching_cum_probs.append(np.array([1, 0, 0, 1]).reshape(2, 2))

    virus_strain = np.array([0, -1, 1, -1])
    contagiousness_factor = np.array([1, 1])
    persistency_factor = np.array([1, 1])

    (
        newly_infected,
        infection_counter,
        missed,
        was_infected,
    ) = _calculate_infections_by_random_contacts(
        random_contacts,
        infectious,
        immunity_level,
        virus_strain,
        group_codes,
        assortative_matching_cum_probs,
        indexers,
        susceptibility_factor,
        contagiousness_factor,
        persistency_factor,
        infection_counter,
        0,
    )

    assert (newly_infected == [-1, 0, -1, 1]).all()
    assert (infection_counter == [1, 0, 1, 0]).all()
    assert (missed == 0).all()
    assert (was_infected == [-1, 0, -1, 0]).all()
