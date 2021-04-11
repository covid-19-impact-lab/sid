import itertools
from contextlib import ExitStack as does_not_raise  # noqa: N813

import numpy as np
import pandas as pd
import pytest
from sid.config import DEFAULT_VIRUS_STRAINS
from sid.initial_conditions import _scale_up_initial_infections
from sid.initial_conditions import _scale_up_initial_infections_numba
from sid.initial_conditions import _spread_out_initial_infections
from sid.initial_conditions import (
    sample_initial_distribution_of_infections_and_immunity,
)
from sid.initial_conditions import sample_initial_immunity
from sid.initial_conditions import sample_initial_infections
from sid.parse_model import parse_initial_conditions
from sid.pathogenesis import draw_course_of_disease
from sid.simulate import _add_default_duration_to_models
from sid.simulate import _process_initial_states


@pytest.mark.unit
def test_scale_up_initial_infections_without_assort_by():
    states = pd.DataFrame(index=pd.RangeIndex(100_000))

    initial_infections = sample_initial_infections(0.1, 100_000)

    scaled_up_infections = _scale_up_initial_infections(
        initial_infections, states, None, 1.3, 0
    )

    assert np.isclose(scaled_up_infections.mean(), 0.13, atol=0.001)


@pytest.mark.unit
def test_scale_up_initial_infections_with_assort_by_equal_groups():
    states = pd.DataFrame(
        {"region": np.random.choice(["North", "East", "South", "West"], size=100_000)}
    )

    initial_infections = sample_initial_infections(0.1, 100_000)

    scaled_up_infections = _scale_up_initial_infections(
        initial_infections, states, None, 1.3, 0
    )

    assert np.isclose(scaled_up_infections.mean(), 0.13, atol=0.001)
    assert np.allclose(
        scaled_up_infections.groupby(states["region"]).mean(), 0.13, atol=0.01
    )


@pytest.mark.unit
def test_scale_up_initial_infections_with_assort_by_unequal_groups():
    probs = np.array([2, 1, 2, 1]) / 6
    regions = np.random.choice(["N", "E", "S", "W"], p=probs, size=100_000)
    states = pd.DataFrame({"region": regions})

    initial_infections = sample_initial_infections(0.1, 100_000)

    scaled_up_infections = _scale_up_initial_infections(
        initial_infections, states, ["region"], 1.3, 0
    )

    assert np.isclose(scaled_up_infections.mean(), 0.13, atol=0.001)
    assert np.allclose(
        scaled_up_infections.groupby(states["region"]).mean(), 0.13, atol=0.01
    )

    infections_per_region = scaled_up_infections.groupby(states["region"]).sum()
    share_per_region = (infections_per_region / infections_per_region.sum()).reindex(
        index=["N", "E", "S", "W"]
    )
    assert np.allclose(share_per_region, probs, 0.07)


@pytest.mark.unit
def test_scale_up_initial_infections_numba():
    initial_infections = np.zeros(100_000).astype(bool)
    probabilities = np.full(100_000, 0.15)

    result = _scale_up_initial_infections_numba(initial_infections, probabilities, 0)

    assert np.isclose(np.mean(result), 0.15, atol=0.01)


@pytest.mark.unit
def test_spread_out_initial_infections():
    infections = sample_initial_infections(0.2, 100_000)

    spread_infections = _spread_out_initial_infections(infections, np.arange(4), 2, 0)

    infections_per_day = spread_infections.sum()
    shares_per_day = infections_per_day / infections_per_day.sum()
    assert np.allclose(shares_per_day, np.array([1, 2, 4, 8]) / 15, atol=0.01)


@pytest.mark.unit
def test_spread_out_initial_infections_no_growth():
    infections = sample_initial_infections(0.2, 100_000)

    spread_infections = _spread_out_initial_infections(infections, np.arange(4), 1, 0)

    infections_per_day = spread_infections.sum()
    shares_per_day = infections_per_day / infections_per_day.sum()
    assert np.allclose(shares_per_day, np.array([1, 0, 0, 0]), atol=0.01)


@pytest.mark.unit
@pytest.mark.parametrize(
    "infections, n_people, index, seed, expectation, expected",
    [
        ([], 1, None, 0, pytest.raises(ValueError, match="'infections' must"), None),
        (None, None, None, -1, pytest.raises(ValueError, match="Seed must be"), None),
        (0.2, None, None, 0, pytest.raises(ValueError, match="Either 'n_people"), None),
        (0.2, 5, pd.RangeIndex(4), 0, pytest.raises(ValueError, match="'n_peop"), None),
        (0.2, 100_000, None, 0, does_not_raise(), lambda x: np.isclose(x.mean(), 0.2)),
        (20000, 100000, None, 0, does_not_raise(), lambda x: np.isclose(x.mean(), 0.2)),
    ],
)
def test_create_initial_infections(
    infections, n_people, index, seed, expectation, expected
):
    with expectation:
        out = sample_initial_infections(infections, n_people, index, seed)
        assert expected(out)


@pytest.mark.unit
@pytest.mark.parametrize(
    "initial_conditions, seed, expectation, expected",
    [
        (
            {"initial_infections": 0.9999999},
            itertools.count(),
            does_not_raise(),
            pd.Series([True] * 15),
        ),
        (
            {"initial_infections": 15, "burn_in_periods": 2},
            itertools.count(),
            does_not_raise(),
            pd.Series([True] * 15),
        ),
        (
            {"initial_infections": pd.Series([True] * 15)},
            itertools.count(),
            does_not_raise(),
            pd.Series([True] * 15),
        ),
        (
            {
                "initial_infections": pd.DataFrame(
                    {0: [True] * 8 + [False] * 7, 1: [False] * 8 + [True] * 7}
                )
            },
            itertools.count(),
            pytest.raises(ValueError, match="Expected 'burn_in_periods'"),
            pd.Series([True] * 15),
        ),
        (
            {
                "initial_infections": pd.DataFrame(
                    {
                        "2019-12-30": [True] * 8 + [False] * 7,
                        "2019-12-31": [False] * 8 + [True] * 7,
                    }
                )
            },
            itertools.count(),
            does_not_raise(),
            pd.Series([True] * 15),
        ),
    ],
)
def test_scale_and_spread_initial_infections(
    initial_states, params, initial_conditions, seed, expectation, expected
):
    with expectation:
        initial_states = _process_initial_states(
            initial_states, {"a": []}, DEFAULT_VIRUS_STRAINS
        )
        initial_states = draw_course_of_disease(initial_states, params, 0)
        initial_conditions = parse_initial_conditions(
            initial_conditions, pd.Timestamp("2020-01-01"), DEFAULT_VIRUS_STRAINS
        )

        result = sample_initial_distribution_of_infections_and_immunity(
            initial_states,
            params,
            initial_conditions,
            {},
            {},
            {},
            DEFAULT_VIRUS_STRAINS,
            {},
            seed,
        )
        assert result["ever_infected"].equals(expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    "immunity, infected_or_immune, expectation, expected",
    [
        (
            7,
            pd.Series([True] * 5 + [False] * 5),
            does_not_raise(),
            lambda x: np.isclose(x.mean(), 0.7),
        ),
        (
            0.7,
            pd.Series([True] * 5 + [False] * 5),
            does_not_raise(),
            lambda x: np.isclose(x.mean(), 0.7),
        ),
        (
            0,
            pd.Series([True] * 5 + [False] * 5),
            does_not_raise(),
            lambda x: np.isclose(x.mean(), 0.5),
        ),
        (
            None,
            pd.Series([True] * 5 + [False] * 5),
            does_not_raise(),
            lambda x: np.isclose(x.mean(), 0.5),
        ),
        (
            pd.DataFrame(),
            [],
            pytest.raises(ValueError, match="'initial_immunity' must be"),
            None,
        ),
    ],
)
def test_create_initial_immunity(immunity, infected_or_immune, expectation, expected):
    with expectation:
        out = sample_initial_immunity(immunity, infected_or_immune, 0)
        assert expected(out)


@pytest.mark.integration
def test_scale_and_spread_initial_infections_w_testing_models(initial_states, params):
    """Testing models can be used to replicate the share_known_cases.

    This test assumes that only halve of all infections are known.

    """

    def demand_model(states, params, seed):
        return states["newly_infected"].copy()

    def allocation_model(n_allocated_tests, demands_test, states, params, seed):
        n_allocated_tests = int(demands_test.sum() / 2)
        allocated_tests = pd.Series(False, index=states.index)
        if n_allocated_tests > 0:
            receives_test = states[demands_test].sample(n=n_allocated_tests).index
            allocated_tests.loc[receives_test] = True

        return allocated_tests

    def processing_model(n_to_be_processed_tests, states, params, seed):
        return states["pending_test"].copy()

    initial_states = pd.concat([initial_states for _ in range(10_000)]).reset_index()

    params.loc[("testing", "allocation", "rel_available_tests"), "value"] = 45_000
    params.loc[("testing", "processing", "rel_available_capacity"), "value"] = 45_000

    testing_demand_models = _add_default_duration_to_models(
        {"dz": {"model": demand_model}}, {"start": "2020-01-01", "end": "2020-01-06"}
    )
    testing_allocation_models = _add_default_duration_to_models(
        {"dz": {"model": allocation_model}},
        {"start": "2020-01-01", "end": "2020-01-06"},
    )
    testing_processing_models = _add_default_duration_to_models(
        {"dz": {"model": processing_model}},
        {"start": "2020-01-01", "end": "2020-01-06"},
    )

    initial_conditions = {
        "burn_in_periods": 3,
        "growth_rate": 2,
        "initial_infections": 70_000,
    }

    initial_states = _process_initial_states(
        initial_states, {"a": []}, DEFAULT_VIRUS_STRAINS
    )
    initial_states = draw_course_of_disease(initial_states, params, 0)
    initial_conditions = parse_initial_conditions(
        initial_conditions, pd.Timestamp("2020-01-04"), DEFAULT_VIRUS_STRAINS
    )

    df = sample_initial_distribution_of_infections_and_immunity(
        states=initial_states,
        params=params,
        initial_conditions=initial_conditions,
        testing_demand_models=testing_demand_models,
        testing_allocation_models=testing_allocation_models,
        testing_processing_models=testing_processing_models,
        virus_strains=DEFAULT_VIRUS_STRAINS,
        vaccination_models={},
        seed=itertools.count(),
    )

    assert df["ever_infected"].sum() == 70_000
    assert np.allclose(
        df["cd_immune_false"].value_counts(normalize=True),
        np.array([8, 4, 2, 1]) / 15,
        atol=1e-3,
    )
    # Shows that tests can only be assigned with a one day lag.
    assert np.allclose(
        df["cd_received_test_result_true"].value_counts(normalize=True),
        np.array([27, 2, 1]) / 30,
        atol=1e-3,
    )
