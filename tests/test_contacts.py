import itertools

import numba as nb
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sid.config import DTYPE_N_CONTACTS
from sid.contacts import _reduce_random_contacts_with_infection_probs
from sid.contacts import calculate_contacts
from sid.contacts import calculate_infections_by_contacts
from sid.contacts import create_group_indexer


@pytest.mark.unit
@pytest.mark.parametrize(
    "states, group_code_name, expected",
    [
        (
            pd.DataFrame({"a": [1] * 7 + [0] * 8}),
            "a",
            [list(range(7, 15)), list(range(7))],
        ),
        (
            pd.DataFrame({"a": pd.Series([0, 1, 2, 3, 0, 1, 2, 3]).astype("category")}),
            "a",
            [[0, 4], [1, 5], [2, 6], [3, 7]],
        ),
        (
            pd.DataFrame({"a": pd.Series([0, 1, 2, 3, 0, 1, 2, -1])}),
            "a",
            [[0, 4], [1, 5], [2, 6], [3]],
        ),
    ],
)
def test_create_group_indexer(states, group_code_name, expected):
    result = create_group_indexer(states, group_code_name)
    result = [r.tolist() for r in result]
    assert result == expected


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
            "virus_strain": pd.Series(["base_strain"] + [pd.NA] * 7, dtype="category"),
        }
    )

    contacts = np.ones((len(states), 1), dtype=bool)

    params = pd.DataFrame(
        columns=["value"],
        data=1,
        index=pd.MultiIndex.from_tuples(
            [("infection_prob", "households", "households")]
        ),
    )

    indexers = {"recurrent": nb.typed.List()}
    indexers["recurrent"].append(create_group_indexer(states, ["households"]))

    assortative_matching_cum_probs = nb.typed.List()
    assortative_matching_cum_probs.append(np.zeros((0, 0)))

    group_codes_info = {"households": {"name": "group_codes_households"}}

    susceptibility_factor = np.ones(len(states))

    virus_strains = {"names": ["base_strain"], "factors": np.ones(1)}

    return (
        states,
        contacts,
        params,
        indexers,
        assortative_matching_cum_probs,
        group_codes_info,
        susceptibility_factor,
        virus_strains,
    )


@pytest.mark.integration
def test_calculate_infections_only_recurrent_all_participate(
    setup_households_w_one_infection,
):
    (
        states,
        recurrent_contacts,
        params,
        indexers,
        assortative_matching_cum_probs,
        group_codes_info,
        susceptibility_factor,
        virus_strains,
    ) = setup_households_w_one_infection

    (
        calc_infected,
        calc_n_has_additionally_infected,
        calc_missed_contacts,
        was_infected_by,
    ) = calculate_infections_by_contacts(
        states=states,
        recurrent_contacts=recurrent_contacts,
        random_contacts=None,
        params=params,
        indexers=indexers,
        assortative_matching_cum_probs=assortative_matching_cum_probs,
        contact_models={"households": {"is_recurrent": True}},
        group_codes_info=group_codes_info,
        susceptibility_factor=susceptibility_factor,
        virus_strains=virus_strains,
        seed=itertools.count(),
    )

    exp_infected = pd.Series([-1] + [0] * 3 + [-1] * 4, dtype="int8")
    exp_infection_counter = pd.Series([3] + [0] * 7, dtype="int32")
    exp_immune = pd.Series([True] * 4 + [False] * 4)
    assert calc_infected.equals(exp_infected)
    assert (
        (states["n_has_infected"] + calc_n_has_additionally_infected)
        .astype(np.int32)
        .equals(exp_infection_counter)
    )
    assert (states["immune"] | (calc_infected == 0)).equals(exp_immune)
    assert calc_missed_contacts is None


@pytest.mark.integration
def test_calculate_infections_only_recurrent_sick_skips(
    setup_households_w_one_infection,
):
    (
        states,
        recurrent_contacts,
        params,
        indexers,
        assortative_matching_cum_probs,
        group_codes_info,
        susceptibility_factor,
        virus_strains,
    ) = setup_households_w_one_infection

    recurrent_contacts[0] = 0

    (
        calc_infected,
        calc_n_has_additionally_infected,
        calc_missed_contacts,
        was_infected_by,
    ) = calculate_infections_by_contacts(
        states=states,
        recurrent_contacts=recurrent_contacts,
        random_contacts=None,
        params=params,
        indexers=indexers,
        assortative_matching_cum_probs=assortative_matching_cum_probs,
        contact_models={"households": {"is_recurrent": True}},
        group_codes_info=group_codes_info,
        susceptibility_factor=susceptibility_factor,
        virus_strains=virus_strains,
        seed=itertools.count(),
    )

    exp_infected = pd.Series([-1] * 8, dtype="int8")
    exp_infection_counter = pd.Series([0] * 8, dtype="int32")

    assert calc_infected.equals(exp_infected)
    assert calc_n_has_additionally_infected.astype(np.int32).equals(
        exp_infection_counter
    )
    assert calc_missed_contacts is None


@pytest.mark.integration
def test_calculate_infections_only_recurrent_one_skips(
    setup_households_w_one_infection,
):
    (
        states,
        recurrent_contacts,
        params,
        indexers,
        assortative_matching_cum_probs,
        group_codes_info,
        susceptibility_factor,
        virus_strains,
    ) = setup_households_w_one_infection

    # 2nd person does not participate in household meeting
    recurrent_contacts[1] = 0

    (
        calc_infected,
        calc_n_has_additionally_infected,
        calc_missed_contacts,
        was_infected_by,
    ) = calculate_infections_by_contacts(
        states=states,
        recurrent_contacts=recurrent_contacts,
        random_contacts=None,
        params=params,
        indexers=indexers,
        assortative_matching_cum_probs=assortative_matching_cum_probs,
        contact_models={"households": {"is_recurrent": True}},
        group_codes_info=group_codes_info,
        susceptibility_factor=susceptibility_factor,
        virus_strains=virus_strains,
        seed=itertools.count(),
    )

    exp_infected = pd.Series([-1, -1] + [0] * 2 + [-1] * 4, dtype="int8")
    exp_infection_counter = pd.Series([2] + [0] * 7, dtype="int32")

    assert calc_infected.equals(exp_infected)
    assert calc_n_has_additionally_infected.astype(np.int32).equals(
        exp_infection_counter
    )
    assert calc_missed_contacts is None


@pytest.mark.integration
def test_calculate_infections_only_recurrent_one_immune(
    setup_households_w_one_infection,
):
    (
        states,
        recurrent_contacts,
        params,
        indexers,
        assortative_matching_cum_probs,
        group_codes_info,
        susceptibility_factor,
        virus_strains,
    ) = setup_households_w_one_infection

    states.loc[1, "immune"] = True

    (
        calc_infected,
        calc_n_has_additionally_infected,
        calc_missed_contacts,
        was_infected_by,
    ) = calculate_infections_by_contacts(
        states=states,
        recurrent_contacts=recurrent_contacts,
        random_contacts=None,
        params=params,
        indexers=indexers,
        assortative_matching_cum_probs=assortative_matching_cum_probs,
        contact_models={"households": {"is_recurrent": True}},
        group_codes_info=group_codes_info,
        susceptibility_factor=susceptibility_factor,
        virus_strains=virus_strains,
        seed=itertools.count(),
    )

    exp_infected = pd.Series([-1, -1] + [0] * 2 + [-1] * 4, dtype="int8")
    exp_infection_counter = pd.Series([2] + [0] * 7, dtype="int32")
    assert calc_infected.equals(exp_infected)
    assert calc_n_has_additionally_infected.astype(np.int32).equals(
        exp_infection_counter
    )
    assert calc_missed_contacts is None


@pytest.mark.integration
def test_calculate_infections_only_non_recurrent(setup_households_w_one_infection):
    (
        states,
        random_contacts,
        *_,
        susceptibility_factor,
        virus_strains,
    ) = setup_households_w_one_infection

    random_contacts[0] = 1

    params = pd.DataFrame(
        columns=["value"],
        data=1,
        index=pd.MultiIndex.from_tuples([("infection_prob", "non_rec", "non_rec")]),
    )
    indexers = {"random": nb.typed.List()}
    indexers["random"].append(create_group_indexer(states, ["group_codes_non_rec"]))
    assortative_matching_cum_probs = nb.typed.List()
    assortative_matching_cum_probs.append(np.array([[0.8, 1], [0.2, 1]]))

    (
        calc_infected,
        calc_n_has_additionally_infected,
        calc_missed_contacts,
        was_infected_by,
    ) = calculate_infections_by_contacts(
        states=states,
        recurrent_contacts=None,
        random_contacts=random_contacts,
        params=params,
        indexers=indexers,
        assortative_matching_cum_probs=assortative_matching_cum_probs,
        contact_models={"non_rec": {"is_recurrent": False}},
        group_codes_info={"non_rec": {"name": "group_codes_non_rec"}},
        susceptibility_factor=susceptibility_factor,
        virus_strains=virus_strains,
        seed=itertools.count(),
    )

    exp_infected = pd.Series([-1, -1, 0, -1, -1, -1, -1, -1], dtype="int8")
    exp_infection_counter = pd.Series([1] + [0] * 7, dtype="int32")
    assert calc_infected.equals(exp_infected)
    assert calc_n_has_additionally_infected.astype(np.int32).equals(
        exp_infection_counter
    )
    assert not np.any(calc_missed_contacts)


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
    recurrent_contacts, random_contacts = calculate_contacts(
        contact_models=contact_models,
        contact_policies=contact_policies,
        states=states_all_alive,
        params=params,
        date=date,
        seed=itertools.count(),
    )

    assert recurrent_contacts is None
    assert (random_contacts == expected).all()


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
    recurrent_contacts, random_contacts = calculate_contacts(
        contact_models=contact_models,
        contact_policies=contact_policies,
        states=states_all_alive,
        params=params,
        date=date,
        seed=itertools.count(),
    )

    assert recurrent_contacts is None
    assert (random_contacts == expected).all()


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
    recurrent_contacts, random_contacts = calculate_contacts(
        contact_models=contact_models,
        contact_policies=contact_policies,
        states=states_all_alive,
        params=params,
        date=date,
        seed=itertools.count(),
    )

    assert recurrent_contacts is None
    assert (random_contacts == expected).all()


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
    recurrent_contacts, random_contacts = calculate_contacts(
        contact_models=contact_models,
        contact_policies=contact_policies,
        states=states_all_alive,
        params=params,
        date=date,
        seed=itertools.count(),
    )

    assert recurrent_contacts is None
    assert (random_contacts == expected).all()


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
    recurrent_contacts, random_contacts = calculate_contacts(
        contact_models=contact_models,
        contact_policies=contact_policies,
        states=states_all_alive,
        params=params,
        date=date,
        seed=itertools.count(),
    )

    assert recurrent_contacts is None
    assert (random_contacts == expected).all()


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
    recurrent_contacts, random_contacts = calculate_contacts(
        contact_models=contact_models,
        contact_policies=contact_policies,
        states=states_with_dead,
        params=params,
        date=date,
        seed=itertools.count(),
    )

    assert recurrent_contacts is None
    assert (random_contacts == expected).all()


@pytest.mark.unit
def test_reduce_contacts_with_infection_prob_one():
    choices = [0, 1, 2, 3, 4]
    weights = [0.5, 0.2, 0.1, 0.1, 0.1]
    contacts = np.random.choice(choices, p=weights, size=(100, 5)).astype(int)
    probs = np.full(5, 1)

    reduced = _reduce_random_contacts_with_infection_probs(contacts, probs, 1234)

    assert_array_equal(reduced, contacts)


@pytest.mark.unit
def test_reduce_contacts_with_infection_prob_zero():
    choices = [0, 1, 2, 3, 4]
    weights = [0.5, 0.2, 0.1, 0.1, 0.1]
    contacts = np.random.choice(choices, p=weights, size=(100, 5)).astype(int)
    probs = np.full(5, 0)

    reduced = _reduce_random_contacts_with_infection_probs(contacts, probs, 1234)

    assert (reduced == 0).all()


@pytest.mark.unit
def test_reduce_contacts_approximately():
    choices = [0, 1, 2, 3, 4]
    weights = [0.5, 0.2, 0.1, 0.1, 0.1]
    contacts = np.random.choice(choices, p=weights, size=(100_000, 5)).astype(int)
    probs = np.arange(0, 10, 2) / 20

    reduced = _reduce_random_contacts_with_infection_probs(contacts, probs, 1234)

    expected_ratios = probs

    calculated_ratios = reduced.sum(axis=0) / contacts.sum(axis=0)

    diff = calculated_ratios - expected_ratios

    assert (diff <= 0.005).all()
