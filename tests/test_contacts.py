import itertools

import numba as nb
import numpy as np
import pandas as pd
import pytest
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
def households_w_one_infected():
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

    virus_strains = {"names": ["base_strain"], "factors": np.ones(1)}

    return {
        "states": states,
        "recurrent_contacts": np.ones((len(states), 1), dtype=bool),
        "random_contacts": None,
        "params": params,
        "indexers": indexers,
        "assortative_matching_cum_probs": assortative_matching_cum_probs,
        "group_codes_info": group_codes_info,
        "susceptibility_factor": np.ones(len(states)),
        "virus_strains": virus_strains,
        "seasonality_factor": 1,
    }


@pytest.mark.integration
def test_calculate_infections_only_recurrent_all_participate(
    households_w_one_infected,
):
    (
        calc_infected,
        calc_n_has_additionally_infected,
        calc_missed_contacts,
        was_infected_by,
    ) = calculate_infections_by_contacts(
        **households_w_one_infected,
        contact_models={"households": {"is_recurrent": True}},
        seed=itertools.count(),
    )

    states = households_w_one_infected["states"]
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
    households_w_one_infected,
):
    households_w_one_infected["recurrent_contacts"][0] = 0

    (
        calc_infected,
        calc_n_has_additionally_infected,
        calc_missed_contacts,
        was_infected_by,
    ) = calculate_infections_by_contacts(
        **households_w_one_infected,
        contact_models={"households": {"is_recurrent": True}},
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
    households_w_one_infected,
):
    # 2nd person does not participate in household meeting
    households_w_one_infected["recurrent_contacts"][1] = 0

    (
        calc_infected,
        calc_n_has_additionally_infected,
        calc_missed_contacts,
        was_infected_by,
    ) = calculate_infections_by_contacts(
        **households_w_one_infected,
        contact_models={"households": {"is_recurrent": True}},
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
    households_w_one_infected,
):
    households_w_one_infected["states"].loc[1, "immune"] = True

    (
        calc_infected,
        calc_n_has_additionally_infected,
        calc_missed_contacts,
        was_infected_by,
    ) = calculate_infections_by_contacts(
        **households_w_one_infected,
        contact_models={"households": {"is_recurrent": True}},
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
def test_calculate_infections_only_non_recurrent(households_w_one_infected):
    random_contacts = households_w_one_infected.pop("recurrent_contacts")
    random_contacts[0] = 1

    params = pd.DataFrame(
        columns=["value"],
        data=1,
        index=pd.MultiIndex.from_tuples([("infection_prob", "non_rec", "non_rec")]),
    )
    states = households_w_one_infected["states"]
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
        states=households_w_one_infected["states"],
        random_contacts=random_contacts,
        recurrent_contacts=None,
        params=params,
        indexers=indexers,
        assortative_matching_cum_probs=assortative_matching_cum_probs,
        contact_models={"non_rec": {"is_recurrent": False}},
        group_codes_info={"non_rec": {"name": "group_codes_non_rec"}},
        susceptibility_factor=households_w_one_infected["susceptibility_factor"],
        virus_strains=households_w_one_infected["virus_strains"],
        seasonality_factor=households_w_one_infected["seasonality_factor"],
        seed=itertools.count(),
    )

    exp_infected = pd.Series([-1, -1, 0, -1, -1, -1, -1, -1], dtype="int8")
    exp_infection_counter = pd.Series([1] + [0] * 7, dtype="int32")
    assert calc_infected.equals(exp_infected)
    assert calc_n_has_additionally_infected.astype(np.int32).equals(
        exp_infection_counter
    )
    assert not np.any(calc_missed_contacts)
