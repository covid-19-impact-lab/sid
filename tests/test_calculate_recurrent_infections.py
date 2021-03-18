import numba as nb
import numpy as np
import pytest
from sid.contacts import _calculate_infections_by_recurrent_contacts


@pytest.mark.unit
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
    infection_counter = np.zeros(4, dtype=np.int_)
    susceptibility_factor = np.ones(len(recurrent_contacts))

    virus_strain = np.array([0, -1, -1, -1])
    contagiousness_factor = np.array([1])

    (
        newly_infected,
        infection_counter,
        immune,
        was_infected,
    ) = _calculate_infections_by_recurrent_contacts(
        recurrent_contacts,
        infectious,
        immune,
        virus_strain,
        group_codes,
        indexers,
        infection_probabilities,
        susceptibility_factor,
        contagiousness_factor,
        infection_counter,
        0,
    )

    assert (newly_infected == [-1, 0, -1, -1]).all()
    assert (infection_counter == [1, 0, 0, 0]).all()
    assert (immune == [True, True, False, False]).all()
    assert (was_infected == [-1, 0, -1, -1]).all()


@pytest.mark.unit
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
    infection_counter = np.zeros(4, dtype=np.int_)
    susceptibility_factor = np.ones(len(recurrent_contacts))

    virus_strain = np.array([0, -1, -1])
    contagiousness_factor = np.array([1])

    (
        newly_infected,
        infection_counter,
        immune,
        was_infected,
    ) = _calculate_infections_by_recurrent_contacts(
        recurrent_contacts,
        infectious,
        immune,
        virus_strain,
        group_codes,
        indexers,
        infection_probabilities,
        susceptibility_factor,
        contagiousness_factor,
        infection_counter,
        0,
    )

    assert (newly_infected == [-1, 0, -1, -1]).all()
    assert (infection_counter == [1, 0, 0, 0]).all()
    assert (immune == [True, True, False, False]).all()
    assert (was_infected == [-1, 0, -1, -1]).all()


@pytest.mark.unit
def test_infections_can_be_scaled_with_multiplier():
    """Run a scenario where infections are halved by the multiplier."""
    n_individuals = 100_000
    recurrent_contacts = np.full((n_individuals, 1), True)
    infectious = np.array([True] + [False] * (n_individuals - 1))
    immune = np.array([True] + [False] * (n_individuals - 1))

    group_codes = np.zeros((n_individuals, 1), dtype=np.int_)

    sub_indexers = nb.typed.List()
    sub_indexers.append(np.arange(n_individuals, dtype=np.int_))
    indexers = nb.typed.List()
    indexers.append(sub_indexers)

    infection_probabilities = np.array([1])
    infection_counter = np.zeros(n_individuals, dtype=np.int_)
    susceptibility_factor = np.full(n_individuals, 0.5)

    virus_strain = np.array([0] + [-1] * (n_individuals - 1))
    contagiousness_factor = np.array([1])

    (
        newly_infected,
        infection_counter,
        immune,
        was_infected,
    ) = _calculate_infections_by_recurrent_contacts(
        recurrent_contacts,
        infectious,
        immune,
        virus_strain,
        group_codes,
        indexers,
        infection_probabilities,
        susceptibility_factor,
        contagiousness_factor,
        infection_counter,
        0,
    )

    assert np.isclose((newly_infected == 0).sum(), n_individuals / 2, atol=1e2)
    assert np.isclose(infection_counter[0], n_individuals / 2, atol=1e2)
    assert np.isclose(immune.sum(), n_individuals / 2, atol=1e2)


@pytest.mark.unit
def test_multiple_virus_strains_spread_in_different_recurrent_groups():
    recurrent_contacts = np.array([True, True, True, True]).reshape(-1, 1)
    infectious = np.array([True, False, True, False])
    immune = np.array([True, False, True, False])

    group_codes = np.array([0, 0, 1, 1]).reshape(-1, 1)

    sub_indexers = nb.typed.List()
    sub_indexers.append(np.array([0, 1]))
    sub_indexers.append(np.array([2, 3]))
    indexers = nb.typed.List()
    indexers.append(sub_indexers)

    infection_probabilities = np.array([1])
    infection_counter = np.zeros(4, dtype=np.int_)
    susceptibility_factor = np.ones(len(recurrent_contacts))

    virus_strain = np.array([0, -1, 1, -1])
    contagiousness_factor = np.array([1, 1])

    (
        newly_infected,
        infection_counter,
        immune,
        was_infected,
    ) = _calculate_infections_by_recurrent_contacts(
        recurrent_contacts,
        infectious,
        immune,
        virus_strain,
        group_codes,
        indexers,
        infection_probabilities,
        susceptibility_factor,
        contagiousness_factor,
        infection_counter,
        0,
    )

    assert (newly_infected == [-1, 0, -1, 1]).all()
    assert (infection_counter == [1, 0, 1, 0]).all()
    assert (immune == [True, True, True, True]).all()
    assert (was_infected == [-1, 0, -1, 0]).all()
