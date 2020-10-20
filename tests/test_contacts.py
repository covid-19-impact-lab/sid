import itertools

import numpy as np
import pandas as pd
import pytest
from numba.typed import List as NumbaList
from numpy.testing import assert_array_equal
from sid.contacts import _calculate_infections_by_contacts_numba
from sid.contacts import calculate_infections_by_contacts
from sid.contacts import create_group_indexer
from sid.contacts import get_loop_entries
from sid.contacts import reduce_contacts_with_infection_probs


@pytest.mark.parametrize(
    "assort_by, expected",
    [
        (
            ["age_group", "region"],
            [[9, 12], [7, 10, 13], [8, 11, 14], [0, 3, 6], [1, 4], [2, 5]],
        ),
        ([], [list(range(15))]),
    ],
)
def test_create_group_indexer(initial_states, assort_by, expected):
    calculated = create_group_indexer(initial_states, assort_by)
    calculated = [arr.tolist() for arr in calculated]

    assert calculated == expected


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.xfail(reason="We redefined how contacts are matched")
def test_calculate_infections_numba_with_single_group(num_regression, seed):
    (
        contacts,
        infectious,
        immune,
        group_codes,
        group_probabilities,
        indexer,
        infection_prob,
        is_recurrent,
        loop_order,
    ) = _sample_data_for_calculate_infections_numba(n_individuals=100, seed=seed)

    (
        infected,
        infection_counter,
        immune,
        missed,
    ) = _calculate_infections_by_contacts_numba(
        contacts,
        infectious,
        immune,
        group_codes,
        group_probabilities,
        indexer,
        infection_prob,
        seed,
        is_recurrent,
        loop_order,
    )

    num_regression.check(
        data_dict={
            "infected": infected.astype("float"),
            "n_has_infected": infection_counter.astype("float"),
            "immune": immune.astype("float"),
            "missed": missed[:, 0].astype("float"),
        },
    )


def _sample_data_for_calculate_infections_numba(
    n_individuals=None,
    n_contacts=None,
    infectious_share=None,
    group_shares=None,
    group_probabilities=None,
    infection_prob=None,
    seed=None,
):
    """Sample data for the calculation of new infections."""
    if seed is not None:
        np.random.seed(seed)

    if n_individuals is None:
        n_individuals = np.random.randint(5, 1_000)

    if n_contacts is None:
        contacts = np.random.randint(2, 6, size=n_individuals)
    else:
        contacts = np.full(n_individuals, n_contacts)

    if infectious_share is None:
        infectious_share = np.random.uniform(0.000001, 1)

    infectious = np.zeros(n_individuals, dtype=np.bool)
    mask = np.random.choice(n_individuals, size=int(n_individuals * infectious_share))
    infectious[mask] = True

    immune = infectious.copy()

    if group_shares is None:
        n_groups = np.random.randint(1, 4)
        group_shares = np.random.uniform(0.1, 1, size=n_groups)
    group_shares = group_shares / group_shares.sum()
    group_shares[-1] = 1 - group_shares[:-1].sum()

    n_groups = len(group_shares)
    group_codes = np.random.choice(n_groups, p=group_shares, size=n_individuals)

    if group_probabilities is None:
        group_probabilities = np.random.uniform(0.00001, 1, size=(n_groups, n_groups))
        group_probabilities = group_probabilities / group_probabilities.sum(
            axis=1, keepdims=True
        )
    group_probs_list = NumbaList()
    group_probs_list.append(group_probabilities.cumsum(axis=1))

    indexer = NumbaList()
    for group in range(n_groups):
        indexer.append(np.where(group_codes == group)[0])

    indexers_list = NumbaList()
    indexers_list.append(indexer)

    if infection_prob is None:
        ip = np.random.uniform()
        infection_prob = np.array([ip])

    is_recurrent = np.array([False])

    loop_order = np.array(list(itertools.product(range(n_individuals), range(1))))

    return (
        contacts.reshape(-1, 1),
        infectious,
        immune,
        group_codes.reshape(-1, 1),
        group_probs_list,
        indexers_list,
        infection_prob,
        is_recurrent,
        loop_order,
    )


def test_calculate_infections():
    """test with only one recurrent contact model and very few states"""
    # set up states
    states = pd.DataFrame(
        {
            "infectious": [True] + [False] * 7,
            "immune": [True] + [False] * 7,
            "group_codes_households": [0] * 4 + [1] * 4,
            "households": [0] * 4 + [1] * 4,
            "n_has_infected": 0,
        }
    )

    contacts = np.ones((len(states), 1))

    params = pd.DataFrame(
        columns=["value"],
        data=1,
        index=pd.MultiIndex.from_tuples(
            [("infection_prob", "households", "households")]
        ),
    )

    indexers = {"households": create_group_indexer(states, ["households"])}

    group_probs = {}

    (
        calc_infected,
        calc_n_has_additionally_infected,
        calc_missed_contacts,
    ) = calculate_infections_by_contacts(
        states=states,
        contacts=contacts,
        params=params,
        indexers=indexers,
        group_cdfs=group_probs,
        seed=itertools.count(),
    )

    exp_infected = pd.Series([False] + [True] * 3 + [False] * 4)
    exp_infection_counter = pd.Series([3] + [0] * 7).astype(np.int32)
    exp_immune = pd.Series([True] * 4 + [False] * 4)
    assert calc_infected.equals(exp_infected)
    assert (
        (states["n_has_infected"] + calc_n_has_additionally_infected)
        .astype(np.int32)
        .equals(exp_infection_counter)
    )
    assert (states["immune"] | calc_infected).equals(exp_immune)
    assert np.all(calc_missed_contacts == 0)


def test_get_loop_entries():
    calculated = get_loop_entries(10, 15)
    expected = np.array(list(itertools.product(range(10), range(15)))).astype(int)
    assert_array_equal(calculated, expected)


def test_reduce_contacts_with_infection_prob_one():
    choices = [0, 1, 2, 3, 4]
    weights = [0.5, 0.2, 0.1, 0.1, 0.1]
    contacts = np.random.choice(choices, p=weights, size=(100, 10)).astype(int)
    is_recurrent = np.array([True, False] * 5)
    probs = np.full(10, 1)

    reduced = reduce_contacts_with_infection_probs(contacts, is_recurrent, probs, 1234)

    assert_array_equal(reduced, contacts)


def test_reduce_contacts_with_infection_prob_zero():
    choices = [0, 1, 2, 3, 4]
    weights = [0.5, 0.2, 0.1, 0.1, 0.1]
    contacts = np.random.choice(choices, p=weights, size=(100, 10)).astype(int)
    is_recurrent = np.array([True, False] * 5)
    probs = np.full(10, 0)

    reduced = reduce_contacts_with_infection_probs(contacts, is_recurrent, probs, 1234)

    assert (reduced[:, ~is_recurrent] == 0).all()
    assert_array_equal(reduced[:, is_recurrent], contacts[:, is_recurrent])


def test_reduce_contacts_approximately():
    choices = [0, 1, 2, 3, 4]
    weights = [0.5, 0.2, 0.1, 0.1, 0.1]
    contacts = np.random.choice(choices, p=weights, size=(100_000, 10)).astype(int)
    is_recurrent = np.array([True, False] * 5)
    probs = np.arange(10) / 20

    reduced = reduce_contacts_with_infection_probs(contacts, is_recurrent, probs, 1234)

    expected_ratios = probs[~is_recurrent]

    calculated_ratios = reduced[:, ~is_recurrent].sum(axis=0) / contacts[
        :, ~is_recurrent
    ].sum(axis=0)

    diff = calculated_ratios - expected_ratios

    assert (diff <= 0.005).all()
