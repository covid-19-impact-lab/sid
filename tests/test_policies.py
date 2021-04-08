import itertools

import numpy as np
import pandas as pd
import pytest
from sid.config import DTYPE_N_CONTACTS
from sid.contacts import _reduce_random_contacts_with_infection_probs
from sid.contacts import calculate_contacts
from sid.contacts import post_process_contacts
from sid.policies import apply_contact_policies


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
        states=states_all_alive,
        params=params,
        seed=itertools.count(),
    )

    recurrent_contacts, random_contacts = apply_contact_policies(
        contact_policies=contact_policies,
        recurrent_contacts=recurrent_contacts,
        random_contacts=random_contacts,
        states=states_all_alive,
        date=date,
        seed=itertools.count(),
    )

    recurrent_contacts, random_contacts = post_process_contacts(
        recurrent_contacts, random_contacts, states_all_alive, contact_models
    )

    assert recurrent_contacts is None
    assert (random_contacts.to_numpy() == expected).all()


@pytest.mark.integration
def test_calculate_contacts_policy_inactive(states_all_alive, contact_models):
    contact_policies = {
        "noone_meets": {
            "affected_contact_model": "first_half_meet",
            "start": pd.Timestamp("2020-08-01"),
            "end": pd.Timestamp("2020-08-30"),
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
        states=states_all_alive,
        params=params,
        seed=itertools.count(),
    )

    recurrent_contacts, random_contacts = apply_contact_policies(
        contact_policies=contact_policies,
        recurrent_contacts=recurrent_contacts,
        random_contacts=random_contacts,
        states=states_all_alive,
        date=date,
        seed=itertools.count(),
    )

    recurrent_contacts, random_contacts = post_process_contacts(
        recurrent_contacts, random_contacts, states_all_alive, contact_models
    )

    assert recurrent_contacts is None
    assert (random_contacts.to_numpy() == expected).all()


@pytest.mark.integration
def test_calculate_contacts_policy_active(states_all_alive, contact_models):
    contact_policies = {
        "noone_meets": {
            "affected_contact_model": "first_half_meet",
            "start": pd.Timestamp("2020-09-01"),
            "end": pd.Timestamp("2020-09-30"),
            "policy": shut_down_model,
        },
    }
    date = pd.Timestamp("2020-09-29")
    params = pd.DataFrame()
    expected = np.tile([1, 0], (len(states_all_alive), 1)).astype(DTYPE_N_CONTACTS)
    recurrent_contacts, random_contacts = calculate_contacts(
        contact_models=contact_models,
        states=states_all_alive,
        params=params,
        seed=itertools.count(),
    )

    recurrent_contacts, random_contacts = apply_contact_policies(
        contact_policies=contact_policies,
        recurrent_contacts=recurrent_contacts,
        random_contacts=random_contacts,
        states=states_all_alive,
        date=date,
        seed=itertools.count(),
    )

    recurrent_contacts, random_contacts = post_process_contacts(
        recurrent_contacts, random_contacts, states_all_alive, contact_models
    )

    assert recurrent_contacts is None
    assert (random_contacts.to_numpy() == expected).all()


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
        },
    }
    date = pd.Timestamp("2020-09-29")
    params = pd.DataFrame()
    expected = np.tile([1, 0], (len(states_all_alive), 1)).astype(DTYPE_N_CONTACTS)
    expected[2:4, 1] = 1
    recurrent_contacts, random_contacts = calculate_contacts(
        contact_models=contact_models,
        states=states_all_alive,
        params=params,
        seed=itertools.count(),
    )

    recurrent_contacts, random_contacts = apply_contact_policies(
        contact_policies=contact_policies,
        recurrent_contacts=recurrent_contacts,
        random_contacts=random_contacts,
        states=states_all_alive,
        date=date,
        seed=itertools.count(),
    )

    recurrent_contacts, random_contacts = post_process_contacts(
        recurrent_contacts, random_contacts, states_all_alive, contact_models
    )

    assert recurrent_contacts is None
    assert (random_contacts.to_numpy() == expected).all()


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
        states=states_with_dead,
        params=params,
        seed=itertools.count(),
    )

    recurrent_contacts, random_contacts = apply_contact_policies(
        contact_policies=contact_policies,
        recurrent_contacts=recurrent_contacts,
        random_contacts=random_contacts,
        states=states_with_dead,
        date=date,
        seed=itertools.count(),
    )

    recurrent_contacts, random_contacts = post_process_contacts(
        recurrent_contacts, random_contacts, states_with_dead, contact_models
    )

    assert recurrent_contacts is None
    assert (random_contacts.to_numpy() == expected).all()


@pytest.mark.unit
def test_reduce_contacts_with_infection_prob_one():
    choices = [0, 1, 2, 3, 4]
    weights = [0.5, 0.2, 0.1, 0.1, 0.1]
    contacts = np.random.choice(choices, p=weights, size=(100, 5)).astype(int)
    probs = np.full(5, 1)

    reduced = _reduce_random_contacts_with_infection_probs(contacts, probs, 1234)

    assert np.allclose(reduced, contacts)


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
