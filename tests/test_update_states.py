import pandas as pd
import pytest
from sid.config import INDEX_NAMES
from sid.update_states import _kill_people_over_icu_limit
from sid.update_states import _update_info_on_new_tests
from sid.update_states import _update_info_on_new_vaccinations


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
@pytest.mark.parametrize("n_dead", range(6))
def test_kill_people_over_icu_limit_binding(n_dead):
    states = pd.DataFrame(
        {
            "needs_icu": [False] * (5 - n_dead) + [True] * (5 + n_dead),
            "cd_dead_true": -1,
        }
    )
    params = pd.DataFrame(
        {
            "category": ["health_system"],
            "subcategory": ["icu_limit_relative"],
            "name": ["icu_limit_relative"],
            "value": [50_000],
        }
    ).set_index(INDEX_NAMES)

    result = _kill_people_over_icu_limit(states, params, 0)
    expected = [10 - n_dead, n_dead] if n_dead != 0 else [10]
    assert (result["cd_dead_true"].value_counts() == expected).all()


@pytest.mark.unit
def test_update_info_on_new_tests():
    """Test that info on tests is correctly update.

    The tests assume three people: 1. A generic case, 2. someone who will receive a
    test, 3. someone who receives a positive test result, 4. someone who receives a
    negative test result.

    """
    states = pd.DataFrame(
        {
            "pending_test_date": pd.to_datetime([None, "2020-01-01", None, None]),
            "cd_received_test_result_true": [-1, -1, 0, 0],
            "cd_received_test_result_true_draws": [3, 3, 3, 3],
            "received_test_result": [False, False, True, True],
            "new_known_case": False,
            "immune": [False, False, True, False],
            "knows_immune": False,
            "cd_knows_immune_false": -1,
            "cd_immune_false": [-1, -1, 100, -1],
            "infectious": [False, False, True, False],
            "knows_infectious": False,
            "cd_knows_infectious_false": -1,
            "cd_infectious_false": [-1, -1, 5, -1],
        }
    )
    to_be_processed_tests = pd.Series([False, True, False, False])

    result = _update_info_on_new_tests(states, to_be_processed_tests)

    expected = pd.DataFrame(
        {
            "pending_test_date": pd.to_datetime([None, None, None, None]),
            "cd_received_test_result_true": [-1, 3, 0, 0],
            "cd_received_test_result_true_draws": [3, 3, 3, 3],
            "received_test_result": [False, False, False, False],
            "new_known_case": [False, False, True, False],
            "immune": [False, False, True, False],
            "knows_immune": [False, False, True, False],
            "cd_knows_immune_false": [-1, -1, 100, -1],
            "cd_immune_false": [-1, -1, 100, -1],
            "infectious": [False, False, True, False],
            "knows_infectious": [False, False, True, False],
            "cd_knows_infectious_false": [-1, -1, 5, -1],
            "cd_infectious_false": [-1, -1, 5, -1],
        }
    )

    assert result.equals(expected)


@pytest.mark.unit
def test_update_info_on_new_vaccinations():
    states = pd.DataFrame(
        {
            "received_vaccine": [False, False, False],
            "cd_is_vaccinated_true": [0, -1, -1],
            "cd_is_vaccinated_true_draws": [0, -1, 40],
        }
    )
    newly_vaccinated = pd.Series([False, False, True])

    result = _update_info_on_new_vaccinations(states, newly_vaccinated)

    expected = pd.DataFrame(
        {
            "received_vaccine": [False, False, True],
            "cd_is_vaccinated_true": [0, -1, 40],
            "cd_is_vaccinated_true_draws": [0, -1, 40],
        }
    )

    assert result.equals(expected)
