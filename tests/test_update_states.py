import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from sid.config import INDEX_NAMES
from sid.update_states import _compute_new_tests_with_share_known_cases
from sid.update_states import _kill_people_over_icu_limit


@pytest.mark.unit
def test_kill_people_over_icu_limit_not_binding():
    states = pd.DataFrame({"needs_icu": [False] * 5 + [True] * 5, "cd_dead_true": -1})
    params = pd.DataFrame(
        {
            "category": ["health_system"],
            "subcategory": ["icu_limit_relative"],
            "name": ["icu_limit_relative"],
            "value": [50_000],
        }
    ).set_index(INDEX_NAMES)

    result = _kill_people_over_icu_limit(states, params, 0)
    assert result["cd_dead_true"].eq(-1).all()


@pytest.mark.unit
def test_kill_people_over_icu_limit_binding():
    states = pd.DataFrame({"needs_icu": [False] * 4 + [True] * 6, "cd_dead_true": -1})
    params = pd.DataFrame(
        {
            "category": ["health_system"],
            "subcategory": ["icu_limit_relative"],
            "name": ["icu_limit_relative"],
            "value": [50_000],
        }
    ).set_index(INDEX_NAMES)

    result = _kill_people_over_icu_limit(states, params, 0)
    assert (result["cd_dead_true"].value_counts() == [9, 1]).all()


@pytest.mark.unit
@given(ratio_newly_infected=st.floats(0, 1), share_known_cases=st.floats(0, 1))
def test_compute_new_tests_with_share_known_cases(
    ratio_newly_infected, share_known_cases
):
    n_people = 100_000
    states = pd.DataFrame(
        {
            "newly_infected": [True] * int(ratio_newly_infected * n_people)
            + [False] * int((1 - ratio_newly_infected) * n_people)
        }
    )

    result = _compute_new_tests_with_share_known_cases(states, share_known_cases)

    assert np.isclose(
        result.mean(), ratio_newly_infected * share_known_cases, atol=0.01
    )
