import itertools

import numpy as np
import pandas as pd
import pytest
from resources import CONTACT_MODELS
from sid import get_simulate_func
from sid.rapid_tests import _compute_who_receives_rapid_tests
from sid.rapid_tests import _sample_test_outcome


@pytest.mark.end_to_end
def test_simulate_rapid_tests(params, initial_states, tmp_path):
    rapid_test_models = {
        "rapid_tests": {
            "model": lambda receives_rapid_test, states, params, seed: pd.Series(
                index=states.index, data=True
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


@pytest.mark.unit
def test_compute_who_reveives_rapid_tests(initial_states, params):
    date = pd.Timestamp("2021-03-24")

    rapid_test_models = {
        "model": {
            "start": pd.Timestamp("2021-03-23"),
            "end": pd.Timestamp("2021-03-25"),
            "model": lambda receives_rapid_test, states, params, seed: pd.Series(
                index=states.index, data=True
            ),
        }
    }

    receives_rapid_test = _compute_who_receives_rapid_tests(
        date, initial_states, params, rapid_test_models, itertools.count(0)
    )

    assert receives_rapid_test.all()


@pytest.mark.unit
def test_compute_who_reveives_rapid_tests_raises_error(initial_states, params):
    date = pd.Timestamp("2021-03-24")

    rapid_test_models = {
        "model": {
            "start": pd.Timestamp("2021-03-23"),
            "end": pd.Timestamp("2021-03-25"),
            "model": lambda receives_rapid_test, states, params, seed: 1,
        }
    }

    with pytest.raises(ValueError, match="model, a rapid_test_model,"):
        _compute_who_receives_rapid_tests(
            date, initial_states, params, rapid_test_models, itertools.count(0)
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
def test_sample_test_outcome_with_specificity(params):
    n_individuals = 1_000_000

    receives_rapid_test = np.ones(n_individuals).astype(bool)
    states = pd.DataFrame({"infectious": np.zeros(n_individuals).astype(bool)})

    is_tested_positive = _sample_test_outcome(
        states, receives_rapid_test, params, itertools.count()
    )

    specificity = params.loc[("rapid_test", "specificity", "specificity"), "value"]
    assert np.isclose(is_tested_positive.mean(), 1 - specificity, atol=1e-3)
