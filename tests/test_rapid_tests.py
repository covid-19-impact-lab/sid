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


@pytest.fixture()
def rapid_test_states():
    np.random.seed(84845)
    group_size = 250_000

    uninfected = pd.DataFrame()
    uninfected["cd_infectious_true"] = np.random.choice(
        [-12, -15, -18], size=group_size, replace=True
    )
    uninfected["infectious"] = False

    pre_infectious = pd.DataFrame()
    pre_infectious["cd_infectious_true"] = np.random.choice([2, 1, 3], size=group_size)
    pre_infectious["infectious"] = False

    start_infectious = pd.DataFrame()
    start_infectious["cd_infectious_true"] = [0] * group_size
    start_infectious["infectious"] = True

    while_infectious = pd.DataFrame()
    while_infectious["cd_infectious_true"] = np.random.choice([-1, -3], size=group_size)
    while_infectious["infectious"] = True

    after_infectious = pd.DataFrame()
    after_infectious["cd_infectious_true"] = np.random.choice([-4, -8], size=group_size)
    after_infectious["infectious"] = False

    states = pd.concat(
        [
            uninfected,
            pre_infectious,
            start_infectious,
            while_infectious,
            after_infectious,
        ],
        axis=0,
    )
    return states


@pytest.mark.unit
def test_sample_test_outcome(rapid_test_states, params):
    states = rapid_test_states
    receives_rapid_test = np.random.choice(
        [True, False], size=len(states), p=[0.8, 0.2]
    )

    is_tested_positive = _sample_test_outcome(
        states=states,
        receives_rapid_test=receives_rapid_test,
        params=params,
        seed=itertools.count(),
    )

    # not to be tested
    assert not is_tested_positive[~receives_rapid_test].any()

    # uninfected
    tested_uninfected = receives_rapid_test & (states["cd_infectious_true"] < -10)
    uninfected_share_positive = is_tested_positive[tested_uninfected].mean()
    specificity = params.loc[("rapid_test", "specificity", "specificity"), "value"]
    assert np.isclose(1 - uninfected_share_positive, specificity, atol=1e-2)

    # preinfectious
    sensitivity = params.loc[("rapid_test", "sensitivity", "pre-infectious"), "value"]
    tested_preinfectious = receives_rapid_test & (states["cd_infectious_true"] > 0)
    preinfectious_share_positive = is_tested_positive[tested_preinfectious].mean()
    assert np.isclose(sensitivity, preinfectious_share_positive, atol=1e-2)

    # first day of infectiousness
    sensitivity = params.loc[("rapid_test", "sensitivity", "start_infectious"), "value"]
    tested_start_infectious = receives_rapid_test & (states["cd_infectious_true"] == 0)
    start_infectious_share_positive = is_tested_positive[tested_start_infectious].mean()
    assert np.isclose(sensitivity, start_infectious_share_positive, atol=1e-2)

    # while infectious
    sensitivity = params.loc[("rapid_test", "sensitivity", "while_infectious"), "value"]
    tested_while_infectious = receives_rapid_test & (
        states["infectious"] & (states["cd_infectious_true"] < 0)
    )
    while_infectious_share_positive = is_tested_positive[tested_while_infectious].mean()
    assert np.isclose(sensitivity, while_infectious_share_positive, atol=1e-2)

    # after infectious
    sensitivity = params.loc[("rapid_test", "sensitivity", "after_infectious"), "value"]
    tested_after_infectious = (
        receives_rapid_test
        & ~states["infectious"]
        & (states["cd_infectious_true"] < 0)
        & (states["cd_infectious_true"] > -10)
    )

    after_infectious_share_positive = is_tested_positive[tested_after_infectious].mean()
    assert np.isclose(sensitivity, after_infectious_share_positive, atol=1e-2)


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
def test_create_sensitivity():
    states = pd.DataFrame(
        columns=["infectious", "cd_infectious_true"],
        data=[
            [False, 2],  # not infectious yet
            [True, 0],  # first day of infectiousness
            [True, -1],  # 2nd day of infectiousness
            [False, -5],  # not infectious anymore
        ],
    )
    sensitivity_params = pd.Series(
        [0.35, 0.88, 0.95, 0.5],
        index=[
            "pre-infectious",
            "start_infectious",
            "while_infectious",
            "after_infectious",
        ],
    )
    result = _create_sensitivity(states, sensitivity_params)
    expected = pd.Series([0.35, 0.88, 0.95, 0.5, 0.0])
    result.equals(expected)


@pytest.mark.unit
def test_create_sensitivity_raises_nan_error():
    states = pd.DataFrame(
        columns=["infectious", "cd_infectious_true"],
        data=[
            [False, 2],  # not infectious yet
            [True, 0],  # first day of infectiousness
            [True, -1],  # 2nd day of infectiousness
            [False, -5],  # not infectious anymore
            [False, -12],  # recovered
        ],
    )
    sensitivity_params = pd.Series(
        [0.35, 0.88, 0.95, 0.5],
        index=[
            "pre-infectious",
            "start_infectious",
            "while_infectious",
            "after_infectious",
        ],
    )
    with pytest.raises(ValueError, match="NaN left in the"):
        _create_sensitivity(states, sensitivity_params)
