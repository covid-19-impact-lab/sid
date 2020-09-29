import itertools

import numpy as np
import pandas as pd
import pytest
from numba.typed import List as NumbaList

from sid.config import DTYPE_N_CONTACTS
from sid.contacts import _calculate_infections_by_contacts_numba
from sid.contacts import calculate_contacts
from sid.contacts import calculate_infections_by_contacts
from sid.contacts import create_group_indexer


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
    states = pd.DataFrame()
    states["infectious"] = [True] + [False] * 7
    states["immune"] = states["infectious"]
    states["group_codes_households"] = [0] * 4 + [1] * 4
    states["households"] = [0] * 4 + [1] * 4
    states["n_has_infected"] = 0
    states["n_has_infected"] = states["n_has_infected"].astype(int)

    contacts = np.ones((len(states), 1))

    params = pd.DataFrame(
        columns=["value"],
        data=1.0,
        index=pd.MultiIndex.from_tuples(
            [("infection_prob", "households", "households")]
        ),
    )

    indexers = {"households": create_group_indexer(states, ["households"])}

    group_probs = {}

    calc_infected, calc_states = calculate_infections_by_contacts(
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
    assert calc_states["n_has_infected"].astype(np.int32).equals(exp_infection_counter)
    assert calc_states["immune"].equals(exp_immune)


# =====================================================================================
# calculate_contacts
# =====================================================================================


@pytest.fixture
def states_all_alive(initial_states):
    states = initial_states[:8].copy()
    states["dead"] = False
    states["needs_icu"] = False
    return states


@pytest.fixture
def contact_models():
    def meet_one(states, params):
        return pd.Series(1, index=states.index)

    def first_half_meet(states, params):
        n_contacts = pd.Series(0, index=states.index)
        first_half = round(len(states) / 2)
        n_contacts[:first_half] = 1
        return n_contacts

    contact_models = {
        "meet_one": {
            "model": meet_one,
            "is_recurrent": False,
        },
        "first_half_meet": {
            "model": first_half_meet,
            "is_recurrent": False,
        },
    }
    return contact_models


def test_calculate_contacts_no_policy(states_all_alive, contact_models):
    contact_policies = {}
    date = pd.Timestamp("2020-09-29")
    params = pd.DataFrame()
    first_half = round(len(states_all_alive) / 2)
    expected = np.array(
        [[1, i < first_half] for i in range(len(states_all_alive))],
        dtype=DTYPE_N_CONTACTS,
    )
    res = calculate_contacts(
        contact_models=contact_models,
        contact_policies=contact_policies,
        states=states_all_alive,
        params=params,
        date=date,
    )
    np.testing.assert_array_equal(expected, res)


def test_calculate_contacts_policy_inactive(states_all_alive, contact_models):
    contact_policies = {
        "first_half_meet": {
            "start": "2020-08-01",
            "end": "2020-08-30",
            "is_active": lambda x: True,
            "multiplier": 0,
        },
    }
    date = pd.Timestamp("2020-09-29")
    params = pd.DataFrame()
    first_half = round(len(states_all_alive) / 2)
    expected = np.tile([1, 0], (len(states_all_alive), 1)).astype(DTYPE_N_CONTACTS)
    expected[:first_half, 1] = 0
    res = calculate_contacts(
        contact_models=contact_models,
        contact_policies=contact_policies,
        states=states_all_alive,
        params=params,
        date=date,
    )
    np.testing.assert_array_equal(expected, res)


def test_calculate_contacts_policy_active(states_all_alive, contact_models):
    contact_policies = {
        "first_half_meet": {
            "start": "2020-09-01",
            "end": "2020-09-30",
            "is_active": lambda states: True,
            "multiplier": 0,
        },
    }
    date = pd.Timestamp("2020-09-29")
    params = pd.DataFrame()
    expected = np.tile([1, 0], (len(states_all_alive), 1)).astype(DTYPE_N_CONTACTS)
    res = calculate_contacts(
        contact_models=contact_models,
        contact_policies=contact_policies,
        states=states_all_alive,
        params=params,
        date=date,
    )
    np.testing.assert_array_equal(expected, res)


@pytest.fixture
def states_with_dead(states_all_alive):
    states_with_dead = states_all_alive.copy()
    states_with_dead.loc[:2, "dead"] = [True, False, True]
    states_with_dead.loc[5:, "needs_icu"] = [True, False, True]
    return states_with_dead


def test_calculate_contacts_with_dead(states_with_dead, contact_models):
    contact_policies = {}
    date = pd.Timestamp("2020-09-29")
    params = pd.DataFrame()
    expected = np.array(
        [
            [0, 0],
            [1, 1],
            [0, 0],
            [1, 1],
            [1, 0],
            [0, 0],
            [1, 0],
            [0, 0],
        ],
        dtype=DTYPE_N_CONTACTS,
    )
    res = calculate_contacts(
        contact_models=contact_models,
        contact_policies=contact_policies,
        states=states_with_dead,
        params=params,
        date=date,
    )
    np.testing.assert_array_equal(expected, res)
