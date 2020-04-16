from pathlib import Path

import pandas as pd
import pytest

from sid.contacts import _get_group_list
from sid.contacts import create_group_indexer
from sid.contacts import get_group_to_code


@pytest.fixture
def initial_states():
    p = Path(__file__).resolve().parent / "test_states.csv"
    return pd.read_csv(p)


def test_get_group_list(initial_states):
    assort_by = ["age_group", "region"]
    calculated = _get_group_list(initial_states, assort_by)
    expected = [
        ("Over 50", "a"),
        ("Over 50", "b"),
        ("Over 50", "c"),
        ("Under 50", "a"),
        ("Under 50", "b"),
        ("Under 50", "c"),
    ]
    assert calculated == expected


def test_group_to_code(initial_states):
    assort_by = ["age_group", "region"]
    calculated = get_group_to_code(initial_states, assort_by)
    expected = {
        "('Over 50', 'a')": 0,
        "('Over 50', 'b')": 1,
        "('Over 50', 'c')": 2,
        "('Under 50', 'a')": 3,
        "('Under 50', 'b')": 4,
        "('Under 50', 'c')": 5,
    }
    assert calculated == expected


def test_create_group_indexer(initial_states):
    assort_by = ["age_group", "region"]
    calculated = create_group_indexer(initial_states, assort_by)
    calculated = [arr.tolist() for arr in calculated]
    expected = [[9, 12], [7, 10, 13], [8, 11, 14], [0, 3, 6], [1, 4], [2, 5]]
    assert calculated == expected
