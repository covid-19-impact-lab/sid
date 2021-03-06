import itertools

import numpy as np
import pandas as pd
import pytest
from numba import njit
from numba.typed import List as NumbaList
from numpy.testing import assert_array_equal
from sid.config import DTYPE_N_CONTACTS
from sid.contacts import _calculate_infections_by_contacts_numba
from sid.contacts import _reduce_contacts_with_infection_probs
from sid.contacts import calculate_contacts
from sid.contacts import calculate_infections_by_contacts
from sid.contacts import create_group_indexer


@pytest.mark.unit
@pytest.mark.parametrize(
    ("assort_by", "expected"),
    [
        (
            ["age_group", "region"],
            [[9, 12], [7, 10, 13], [8, 11, 14], [0, 3, 6], [1, 4], [2, 5]],
        ),
        ([], [list(range(15))]),
    ],
)
def test_create_group_indexer(initial_states, assort_by, expected):
    calculated = create_group_indexer(initial_states, assort_by, is_recurrent=False)
    calculated = [arr.tolist() for arr in calculated]

    assert calculated == expected


@pytest.mark.unit
def test_create_group_indexer_recurrent():
    df = pd.DataFrame()
    df["some_id"] = pd.Series([1, -1, 2, 2, 1]).astype("category")
    calculated = create_group_indexer(df, ["some_id"], True)
    calculated = [arr.tolist() for arr in calculated]
    assert calculated == [[0, 4], [2, 3]]


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(10))
def test_calculate_infections_numba_with_single_group(num_regression, seed):
    """If you need to regenerate the test data, use ``pytest --force-regen``."""
    (
        contacts,
        infectious,
        immune,
        group_codes,
        group_probabilities,
        indexer,
        infection_prob,
        is_recurrent,
    ) = _sample_data_for_calculate_infections_numba(n_individuals=100, seed=seed)

    (
        infected,
        infection_counter,
        immune,
        missed,
        was_infected_by,
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

    return (
        contacts.reshape(-1, 1),
        infectious,
        immune,
        group_codes.reshape(-1, 1),
        group_probs_list,
        indexers_list,
        infection_prob,
        is_recurrent,
    )


@pytest.fixture()
def setup_households_w_one_infection():
    states = pd.DataFrame(
        {
            "infectious": [True] + [False] * 7,
            "immune": [True] + [False] * 7,
            "group_codes_households": [0] * 4 + [1] * 4,
            "households": [0] * 4 + [1] * 4,
            "group_codes_non_rec": [0] * 4 + [1] * 4,
            "n_has_infected": 0,
        }
    )

    contacts = np.ones((len(states), 1), dtype=int)

    params = pd.DataFrame(
        columns=["value"],
        data=1,
        index=pd.MultiIndex.from_tuples(
            [("infection_prob", "households", "households")]
        ),
    )

    indexers = {
        "households": create_group_indexer(states, ["households"], is_recurrent=False)
    }

    group_probs = {}

    group_codes_names = {"households": "group_codes_households"}

    return states, contacts, params, indexers, group_probs, group_codes_names


@pytest.mark.integration
def test_calculate_infections_only_recurrent_all_participate(
    setup_households_w_one_infection,
):
    (
        states,
        contacts,
        params,
        indexers,
        group_probs,
        group_codes_names,
    ) = setup_households_w_one_infection

    (
        calc_infected,
        calc_n_has_additionally_infected,
        calc_missed_contacts,
        was_infected_by,
    ) = calculate_infections_by_contacts(
        states=states,
        contacts=contacts,
        params=params,
        indexers=indexers,
        group_cdfs=group_probs,
        code_to_contact_model={0: "the_model"},
        group_codes_names=group_codes_names,
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


@pytest.mark.integration
def test_calculate_infections_only_recurrent_sick_skips(
    setup_households_w_one_infection,
):
    (
        states,
        contacts,
        params,
        indexers,
        group_probs,
        group_codes_names,
    ) = setup_households_w_one_infection

    contacts[0] = 0

    (
        calc_infected,
        calc_n_has_additionally_infected,
        calc_missed_contacts,
        was_infected_by,
    ) = calculate_infections_by_contacts(
        states=states,
        contacts=contacts,
        params=params,
        indexers=indexers,
        group_cdfs=group_probs,
        code_to_contact_model={0: "the_model"},
        group_codes_names=group_codes_names,
        seed=itertools.count(),
    )

    exp_infected = pd.Series([False] * 8)
    exp_infection_counter = pd.Series([0] * 8).astype(np.int32)

    assert calc_infected.equals(exp_infected)
    assert calc_n_has_additionally_infected.astype(np.int32).equals(
        exp_infection_counter
    )


@pytest.mark.integration
def test_calculate_infections_only_recurrent_one_skips(
    setup_households_w_one_infection,
):
    (
        states,
        contacts,
        params,
        indexers,
        group_probs,
        group_codes_names,
    ) = setup_households_w_one_infection

    # 2nd person does not participate in household meeting
    contacts[1] = 0

    (
        calc_infected,
        calc_n_has_additionally_infected,
        calc_missed_contacts,
        was_infected_by,
    ) = calculate_infections_by_contacts(
        states=states,
        contacts=contacts,
        params=params,
        indexers=indexers,
        group_cdfs=group_probs,
        code_to_contact_model={0: "the_model"},
        group_codes_names=group_codes_names,
        seed=itertools.count(),
    )

    exp_infected = pd.Series([False, False] + [True] * 2 + [False] * 4)
    exp_infection_counter = pd.Series([2] + [0] * 7).astype(np.int32)
    assert calc_infected.equals(exp_infected)
    assert calc_n_has_additionally_infected.astype(np.int32).equals(
        exp_infection_counter
    )


@pytest.mark.integration
def test_calculate_infections_only_recurrent_one_immune(
    setup_households_w_one_infection,
):
    (
        states,
        contacts,
        params,
        indexers,
        group_probs,
        group_codes_names,
    ) = setup_households_w_one_infection

    states.loc[1, "immune"] = True

    (
        calc_infected,
        calc_n_has_additionally_infected,
        calc_missed_contacts,
        was_infected_by,
    ) = calculate_infections_by_contacts(
        states=states,
        contacts=contacts,
        params=params,
        indexers=indexers,
        group_cdfs=group_probs,
        code_to_contact_model={0: "the_model"},
        group_codes_names=group_codes_names,
        seed=itertools.count(),
    )

    exp_infected = pd.Series([False, False] + [True] * 2 + [False] * 4)
    exp_infection_counter = pd.Series([2] + [0] * 7).astype(np.int32)
    assert calc_infected.equals(exp_infected)
    assert calc_n_has_additionally_infected.astype(np.int32).equals(
        exp_infection_counter
    )


def set_deterministic_context(m):
    """Replace all randomness in the model with specified orders."""

    @njit
    def fake_choose_other_group(a, cdf):
        """Deterministically switch between groups."""
        return a[0]

    m.setattr("sid.contacts.choose_other_group", fake_choose_other_group)

    @njit
    def fake_choose_j(a, weights):
        """Deterministically switch between groups."""
        return a[1]

    m.setattr("sid.contacts.choose_other_individual", fake_choose_j)


@pytest.mark.integration
def test_calculate_infections_only_non_recurrent(
    setup_households_w_one_infection, monkeypatch
):
    states, contacts, *_ = setup_households_w_one_infection

    contacts[0] = 1

    params = pd.DataFrame(
        columns=["value"],
        data=1.0,
        index=pd.MultiIndex.from_tuples([("infection_prob", "non_rec", "non_rec")]),
    )
    indexers = {
        "non_rec": create_group_indexer(
            states, ["group_codes_non_rec"], is_recurrent=False
        )
    }
    group_probs = {"non_rec": np.array([[0.8, 1], [0.2, 1]])}

    with monkeypatch.context() as m:
        set_deterministic_context(m)
        (
            calc_infected,
            calc_n_has_additionally_infected,
            calc_missed_contacts,
            was_infected_by,
        ) = calculate_infections_by_contacts(
            states=states,
            contacts=contacts,
            params=params,
            indexers=indexers,
            group_cdfs=group_probs,
            code_to_contact_model={0: "the_model"},
            group_codes_names={"non_rec": "group_codes_non_rec"},
            seed=itertools.count(),
        )

    exp_infected = pd.Series([False, True, False, False, False, False, False, False])
    exp_infection_counter = pd.Series([1] + [0] * 7).astype(np.int32)
    assert calc_infected.equals(exp_infected)
    assert calc_n_has_additionally_infected.astype(np.int32).equals(
        exp_infection_counter
    )


# =====================================================================================
# calculate_contacts
# =====================================================================================


def shut_down_model(states, contacts, seed):  # noqa: U100
    """Set all contacts to zero independent of incoming contacts."""
    return pd.Series(0, index=states.index)


@pytest.fixture()
def states_all_alive(initial_states):
    states = initial_states[:8].copy()
    states["dead"] = False
    states["needs_icu"] = False
    return states


@pytest.fixture()
def contact_models():
    def meet_one(states, params, seed):
        return pd.Series(1, index=states.index)

    def first_half_meet(states, params, seed):
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


@pytest.mark.integration
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
        seed=itertools.count(),
    )
    np.testing.assert_array_equal(expected, res)


@pytest.mark.integration
def test_calculate_contacts_policy_inactive(states_all_alive, contact_models):
    contact_policies = {
        "noone_meets": {
            "affected_contact_model": "first_half_meet",
            "start": pd.Timestamp("2020-08-01"),
            "end": pd.Timestamp("2020-08-30"),
            "is_active": lambda x: True,
            "policy": shut_down_model,
        },
    }
    date = pd.Timestamp("2020-09-29")
    params = pd.DataFrame()
    first_half = round(len(states_all_alive) / 2)
    expected = np.tile([1, 0], (len(states_all_alive), 1)).astype(DTYPE_N_CONTACTS)
    expected[:first_half, 1] = 1
    res = calculate_contacts(
        contact_models=contact_models,
        contact_policies=contact_policies,
        states=states_all_alive,
        params=params,
        date=date,
        seed=itertools.count(),
    )
    np.testing.assert_array_equal(expected, res)


@pytest.mark.integration
def test_calculate_contacts_policy_active(states_all_alive, contact_models):
    contact_policies = {
        "noone_meets": {
            "affected_contact_model": "first_half_meet",
            "start": pd.Timestamp("2020-09-01"),
            "end": pd.Timestamp("2020-09-30"),
            "is_active": lambda states: True,
            "policy": shut_down_model,
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
        seed=itertools.count(),
    )
    np.testing.assert_array_equal(expected, res)


@pytest.mark.integration
def test_calculate_contacts_policy_inactive_through_function(
    states_all_alive, contact_models
):
    contact_policies = {
        "noone_meets": {
            "affected_contact_model": "first_half_meet",
            "start": pd.Timestamp("2020-09-01"),
            "end": pd.Timestamp("2020-09-30"),
            "is_active": lambda states: False,
            "policy": shut_down_model,
        },
    }
    date = pd.Timestamp("2020-09-29")
    params = pd.DataFrame()
    expected = np.tile([1, 0], (len(states_all_alive), 1)).astype(DTYPE_N_CONTACTS)
    first_half = round(len(states_all_alive) / 2)
    expected[:first_half, 1] = 1
    res = calculate_contacts(
        contact_models=contact_models,
        contact_policies=contact_policies,
        states=states_all_alive,
        params=params,
        date=date,
        seed=itertools.count(),
    )
    np.testing.assert_array_equal(expected, res)


@pytest.mark.integration
def test_calculate_contacts_policy_active_policy_func(states_all_alive, contact_models):
    def reduce_to_1st_quarter(states, contacts, seed):
        contacts = contacts.copy()
        contacts[: int(len(contacts) / 4)] = 0
        return contacts

    contact_policies = {
        "noone_meets": {
            "affected_contact_model": "first_half_meet",
            "start": pd.Timestamp("2020-09-01"),
            "end": pd.Timestamp("2020-09-30"),
            "policy": reduce_to_1st_quarter,
            "is_active": lambda states: True,
        },
    }
    date = pd.Timestamp("2020-09-29")
    params = pd.DataFrame()
    expected = np.tile([1, 0], (len(states_all_alive), 1)).astype(DTYPE_N_CONTACTS)
    expected[2:4, 1] = 1
    res = calculate_contacts(
        contact_models=contact_models,
        contact_policies=contact_policies,
        states=states_all_alive,
        params=params,
        date=date,
        seed=itertools.count(),
    )
    np.testing.assert_array_equal(expected, res)


@pytest.fixture()
def states_with_dead(states_all_alive):
    states_with_dead = states_all_alive.copy()
    states_with_dead.loc[:2, "dead"] = [True, False, True]
    states_with_dead.loc[5:, "needs_icu"] = [True, False, True]
    return states_with_dead


@pytest.mark.integration
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
        seed=itertools.count(),
    )
    np.testing.assert_array_equal(expected, res)


@pytest.mark.unit
def test_reduce_contacts_with_infection_prob_one():
    choices = [0, 1, 2, 3, 4]
    weights = [0.5, 0.2, 0.1, 0.1, 0.1]
    contacts = np.random.choice(choices, p=weights, size=(100, 10)).astype(int)
    is_recurrent = np.array([True, False] * 5)
    probs = np.full(10, 1)

    reduced = _reduce_contacts_with_infection_probs(contacts, is_recurrent, probs, 1234)

    assert_array_equal(reduced, contacts)


@pytest.mark.unit
def test_reduce_contacts_with_infection_prob_zero():
    choices = [0, 1, 2, 3, 4]
    weights = [0.5, 0.2, 0.1, 0.1, 0.1]
    contacts = np.random.choice(choices, p=weights, size=(100, 10)).astype(int)
    is_recurrent = np.array([True, False] * 5)
    probs = np.full(10, 0)

    reduced = _reduce_contacts_with_infection_probs(contacts, is_recurrent, probs, 1234)

    assert (reduced[:, ~is_recurrent] == 0).all()
    assert_array_equal(reduced[:, is_recurrent], contacts[:, is_recurrent])


@pytest.mark.unit
def test_reduce_contacts_approximately():
    choices = [0, 1, 2, 3, 4]
    weights = [0.5, 0.2, 0.1, 0.1, 0.1]
    contacts = np.random.choice(choices, p=weights, size=(100_000, 10)).astype(int)
    is_recurrent = np.array([True, False] * 5)
    probs = np.arange(10) / 20

    reduced = _reduce_contacts_with_infection_probs(contacts, is_recurrent, probs, 1234)

    expected_ratios = probs[~is_recurrent]

    calculated_ratios = reduced[:, ~is_recurrent].sum(axis=0) / contacts[
        :, ~is_recurrent
    ].sum(axis=0)

    diff = calculated_ratios - expected_ratios

    assert (diff <= 0.005).all()
