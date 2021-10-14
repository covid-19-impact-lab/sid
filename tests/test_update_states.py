import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal
from sid.config import INDEX_NAMES
from sid.update_states import _kill_people_over_icu_limit
from sid.update_states import _update_immunity_level
from sid.update_states import _update_info_on_new_tests
from sid.update_states import _update_info_on_new_vaccinations
from sid.update_states import compute_waning_immunity
from sid.update_states import update_derived_state_variables


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
            "immunity": [0.0, 0.0, 1.0, 0.0],
            "knows_immune": False,
            "symptomatic": [False, False, False, False],
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
            "immunity": [0.0, 0.0, 1.0, 0.0],
            "knows_immune": [False, False, True, False],
            "symptomatic": [False, False, False, False],
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
            "newly_vaccinated": [False, False, False, False],
            "ever_vaccinated": [False, False, False, True],
            "cd_ever_vaccinated": [-9999, -9999, -9999, -10],
        }
    )
    newly_vaccinated = pd.Series([False, False, True, False])

    result = _update_info_on_new_vaccinations(states, newly_vaccinated)

    expected = pd.DataFrame(
        {
            "newly_vaccinated": [False, False, True, False],
            "ever_vaccinated": [False, False, True, True],
            "cd_ever_vaccinated": [-9999, -9999, 0, -10],
        }
    )

    assert result.equals(expected)


@pytest.mark.unit
def test_update_derived_state_variables():
    states = pd.DataFrame()
    states["a"] = np.arange(5)
    derived_state_variables = {"b": "a <= 3"}
    calculated = update_derived_state_variables(states, derived_state_variables)["b"]
    expected = pd.Series([True, True, True, True, False], name="b")
    assert_series_equal(calculated, expected)


@pytest.fixture()
def waning_immunity_fixture():
    """Waning immunity fixture.

    We test 6 cases (assuming that time_to_reach_maximum is 7 for infection and 28 for
    vaccination):

    (-9999): Needs to be set to zero.
    (-9998): Needs to be set to zero, because linear function will be negative.
    (0): Needs to be zero.
    (-6): Is the increasing part for both infection and vaccination.
    (-8): This is in the decreasing (increasing) part for infection (vaccination).
    (-29): In this part both infection and vaccination should be decreasing.

    """
    days_since_event_cd = pd.Series([-9999, -9998, 0, -6, -8, -29])

    states = pd.DataFrame(
        {
            "cd_ever_infected": days_since_event_cd,
            "cd_ever_vaccinated": days_since_event_cd,
        }
    )

    # need to perform next lines since ``compute_waning_immunity`` expects this task to
    # be done by ``_udpate_immunity_level``.
    days_since_event = -days_since_event_cd
    days_since_event[days_since_event >= 9999] = 0

    # values below were calucated by hand
    expected_immunity_infection = pd.Series([0, 0, 0, 0.62344, 0.9899, 0.9878])
    expected_immunity_vaccination = pd.Series(
        [0, 0, 0, 0.00787172, 0.018658891, 0.7998]
    )

    expected_immunity = np.maximum(
        expected_immunity_infection, expected_immunity_vaccination
    )
    expected_states = states.assign(immunity=expected_immunity)

    return {
        "states": states,
        "days_since_event": days_since_event,
        "expected_immunity_infection": expected_immunity_infection,
        "expected_immunity_vaccination": expected_immunity_vaccination,
        "expected_states": expected_states,
    }


@pytest.mark.unit
def test_update_immunity_level(params, waning_immunity_fixture):
    states = waning_immunity_fixture["states"]
    expected_states = waning_immunity_fixture["expected_states"]
    calculated = _update_immunity_level(states, params)
    assert_frame_equal(calculated, expected_states, check_dtype=False)


@pytest.mark.unit
@pytest.mark.parametrize("event", ["infection", "vaccination"])
def testcompute_waning_immunity(params, event, waning_immunity_fixture):
    days_since_event = waning_immunity_fixture["days_since_event"]
    expected = waning_immunity_fixture[f"expected_immunity_{event}"]
    calculated = compute_waning_immunity(params, days_since_event, event)
    assert_series_equal(calculated, expected, check_dtype=False)
