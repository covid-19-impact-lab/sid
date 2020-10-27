import pandas as pd
import pytest
from pytest import approx

from sid.shared import calculate_r_effective
from sid.shared import calculate_r_zero
from sid.shared import factorize_assortative_variables


@pytest.mark.parametrize(
    "assort_by, expected",
    [
        (
            ["age_group", "region"],
            [
                ("Over 50", "a"),
                ("Over 50", "b"),
                ("Over 50", "c"),
                ("Under 50", "a"),
                ("Under 50", "b"),
                ("Under 50", "c"),
            ],
        ),
        ([], [(0,)]),
    ],
)
def test_factorize_assortative_variables(initial_states, assort_by, expected):
    _, group_code_values = factorize_assortative_variables(initial_states, assort_by)

    assert set(group_code_values) == set(expected)


@pytest.fixture
def data_for_replication_numbers():
    df = pd.DataFrame(
        {
            "cd_infectious_false": [-2, -1, -1, 0] + [2, 5],
            "n_has_infected": [2, 1, 1, 1] + [0, 0],
            "immune": [True] * 4 + [False, False],
        }
    )
    return df


def test_r_effective_large(data_for_replication_numbers):
    expected = 5 / 4
    res = calculate_r_effective(data_for_replication_numbers, 2)
    assert res == expected


def test_r_effective_small(data_for_replication_numbers):
    expected = 3 / 3
    res = calculate_r_effective(data_for_replication_numbers, 1)
    assert res == expected


def test_r_zero_few_susceptible(data_for_replication_numbers):
    res = calculate_r_zero(data_for_replication_numbers, 2)
    expected = (5 / 4) / (2 / 6)
    assert res == approx(expected)


def test_r_zero_all_susceptible(data_for_replication_numbers):
    df = data_for_replication_numbers
    df["immune"] = False
    expected = (3 / 3) / (6 / 6)
    res = calculate_r_zero(data_for_replication_numbers, 1)
    assert res == expected
