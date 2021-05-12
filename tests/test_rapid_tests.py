import itertools

import numpy as np
import pandas as pd
import pytest
from resources import CONTACT_MODELS
from sid import get_simulate_func
from sid.rapid_tests import _compute_who_receives_rapid_tests
from sid.rapid_tests import _create_sensitivity
from sid.rapid_tests import _sample_test_outcome
from sid.rapid_tests import _update_states_with_rapid_tests_outcomes


@pytest.mark.end_to_end
def test_simulate_rapid_tests(params, initial_states, tmp_path):
    rapid_test_models = {
        "rapid_tests": {
            "model": lambda receives_rapid_test, states, params, contacts, seed: (
                pd.Series(index=states.index, data=True)
            )
        }
    }

    params.loc[("infection_prob", "standard", "standard"), "value"] = 0.5

    simulate = get_simulate_func(
        params=params,
        initial_states=initial_states,
        contact_models=CONTACT_MODELS,
        path=tmp_path,
        rapid_test_models=rapid_test_models,
        seed=144,
    )

    result = simulate(params)

    time_series = result["time_series"].compute()
    last_states = result["last_states"].compute()

    for df in [time_series, last_states]:
        assert isinstance(df, pd.DataFrame)

    assert last_states["is_tested_positive_by_rapid_test"].any()
    assert (last_states["cd_received_rapid_test"] == -1).all()


@pytest.mark.end_to_end
def test_simulate_rapid_tests_with_reaction_models(params, initial_states, tmp_path):
    rapid_test_models = {
        "rapid_tests": {
            "model": lambda receives_rapid_test, states, params, contacts, seed: (
                pd.Series(index=states.index, data=True)
            )
        }
    }
    rapid_test_reaction_models = {
        "shutdown_standard": {
            "model": lambda contacts, states, params, seed: contacts["standard"]
            .where(~states["is_tested_positive_by_rapid_test"], 0)
            .to_frame()
        }
    }

    params.loc[("infection_prob", "standard", "standard"), "value"] = 0.5

    simulate = get_simulate_func(
        params=params,
        initial_states=initial_states,
        contact_models=CONTACT_MODELS,
        path=tmp_path,
        rapid_test_models=rapid_test_models,
        rapid_test_reaction_models=rapid_test_reaction_models,
        saved_columns={"contacts": True},
        seed=144,
    )

    result = simulate(params)

    time_series = result["time_series"].compute()
    last_states = result["last_states"].compute()

    for df in [time_series, last_states]:
        assert isinstance(df, pd.DataFrame)

    assert last_states["is_tested_positive_by_rapid_test"].any()
    assert (last_states["cd_received_rapid_test"] == -1).all()
    assert (
        time_series.loc[
            time_series["is_tested_positive_by_rapid_test"], "n_contacts_standard"
        ]
        .eq(0)
        .all()
    )


@pytest.mark.unit
def test_compute_who_reveives_rapid_tests(initial_states, params):
    date = pd.Timestamp("2021-03-24")

    rapid_test_models = {
        "model": {
            "start": pd.Timestamp("2021-03-23"),
            "end": pd.Timestamp("2021-03-25"),
            "model": lambda receives_rapid_test, states, params, contacts, seed: (
                pd.Series(index=states.index, data=True)
            ),
        }
    }

    contacts = pd.Series(index=initial_states.index, data=0)

    receives_rapid_test = _compute_who_receives_rapid_tests(
        date, initial_states, params, rapid_test_models, contacts, itertools.count(0)
    )

    assert receives_rapid_test.all()


@pytest.mark.unit
def test_compute_who_reveives_rapid_tests_raises_error(initial_states, params):
    date = pd.Timestamp("2021-03-24")

    rapid_test_models = {
        "model": {
            "start": pd.Timestamp("2021-03-23"),
            "end": pd.Timestamp("2021-03-25"),
            "model": lambda receives_rapid_test, states, params, contacts, seed: 1,
        }
    }

    contacts = pd.Series(index=initial_states.index, data=0)

    with pytest.raises(ValueError, match="The model 'model' of 'rapid_test_models'"):
        _compute_who_receives_rapid_tests(
            date,
            initial_states,
            params,
            rapid_test_models,
            contacts,
            itertools.count(0),
        )


@pytest.mark.unit
def test_sample_test_outcome_with_sensitivity(params):
    n_individuals = 1_000_000

    receives_rapid_test = np.ones(n_individuals).astype(bool)
    states = pd.DataFrame({"infectious": receives_rapid_test})

    is_tested_positive = _sample_test_outcome(
        states, receives_rapid_test, params, itertools.count()
    )

    sensitivity = params.loc[("rapid_test", "sensitivity", "sensitivity"), "value"]
    assert np.isclose(is_tested_positive.mean(), sensitivity, atol=1e-3)


@pytest.mark.unit
def test_sample_test_outcome_with_sensitivity_time_dependent(params):
    n_individuals = 10_000_000

    cd_infectious_true = np.random.choice([1, 0, -1, -2], n_individuals, True)
    states = pd.DataFrame(
        {
            "infectious": cd_infectious_true <= 0,
            "cd_infectious_true": cd_infectious_true,
        }
    )
    receives_rapid_test = np.ones(n_individuals).astype(bool)
    receives_rapid_test[:1000] = False

    params = params.drop([("rapid_test", "sensitivity")])
    params.loc[("rapid_test", "sensitivity", 0)] = 0.5
    params.loc[("rapid_test", "sensitivity", -1)] = 0.9

    is_tested_positive = _sample_test_outcome(
        states, receives_rapid_test, params, itertools.count()
    )
    assert not is_tested_positive[:1000].any()
    expected_share_false_positive = (
        1 - params.loc[("rapid_test", "specificity", "specificity"), "value"]
    )
    res_share_false_positive = is_tested_positive[
        states["cd_infectious_true"] > 0 & receives_rapid_test
    ].mean()
    assert np.isclose(
        expected_share_false_positive, res_share_false_positive, atol=1e-3
    )

    first_day_share_positive = is_tested_positive[
        states["cd_infectious_true"] == 0 & receives_rapid_test
    ].mean()
    assert np.isclose(
        first_day_share_positive,
        0.5,
        atol=1e-3,
    )
    later_share_positive = is_tested_positive[
        states["cd_infectious_true"] < 0 & receives_rapid_test
    ].mean()
    assert np.isclose(
        later_share_positive,
        0.9,
        atol=1e-3,
    )


@pytest.mark.unit
def test_sample_test_outcome_with_specificity(params):
    n_individuals = 1_000_000

    receives_rapid_test = np.ones(n_individuals).astype(bool)
    states = pd.DataFrame({"infectious": np.zeros(n_individuals).astype(bool)})

    is_tested_positive = _sample_test_outcome(
        states, receives_rapid_test, params, itertools.count()
    )

    specificity = params.loc[("rapid_test", "specificity", "specificity"), "value"]
    assert np.isclose(is_tested_positive.mean(), 1 - specificity, atol=1e-3)


@pytest.mark.unit
def test_update_states_with_rapid_tests_outcomes():
    columns = ["cd_received_rapid_test", "is_tested_positive_by_rapid_test"]

    states = pd.DataFrame(
        [
            (-9, False),  # Person receiving no rapid test.
            (-9, False),  # Person receiving a negative rapid test.
            (-9, False),  # Person receiving a positive rapid test.
        ],
        columns=columns,
    )
    receives_rapid_test = pd.Series([False, True, True])
    is_tested_positive = pd.Series([False, False, True])

    result = _update_states_with_rapid_tests_outcomes(
        states, receives_rapid_test, is_tested_positive
    )

    expected = pd.DataFrame([(-9, False), (0, False), (0, True)], columns=columns)

    assert result.equals(expected)


@pytest.mark.unit
def test_create_sensitivity_time_dependent():
    states = pd.DataFrame(
        [
            [False, 2],  # not infectious yet
            [True, 0],  # first day of infectiousness
            [True, -1],  # 2nd day of infectiousness
            [True, -2],  # not getting tested
            [True, -4],  # after latest specified infectiousness
            [False, -5],  # not infectious anymore
        ],
        columns=["infectious", "cd_infectious_true"],
    )
    params = pd.Series([0.5, 0.7, 0.9], index=[0, -1, -2])
    receives_test_and_is_infectious = pd.Series([False, True, True, False, True, False])
    result = _create_sensitivity(states, params, receives_test_and_is_infectious)
    expected = pd.Series([0.5, 0.7, 0.9], index=[1, 2, 4])
    assert result.equals(expected)


@pytest.mark.unit
def test_create_sensitivity():
    states = pd.DataFrame(
        [
            [False, 2],  # not infectious yet
            [True, 0],  # first day of infectiousness
            [True, -1],  # 2nd day of infectiousness
            [True, -2],  # not getting tested
            [True, -4],  # after latest specified infectiousness
            [False, -5],  # not infectious anymore
        ],
        columns=["infectious", "cd_infectious_true"],
    )
    params = pd.Series([0.5], index=["sensitivity"])
    receives_test_and_is_infectious = pd.Series([False, True, True, False, True, False])
    result = _create_sensitivity(states, params, receives_test_and_is_infectious)
    assert (result == 0.5).all()
