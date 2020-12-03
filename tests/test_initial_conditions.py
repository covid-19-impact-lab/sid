import itertools
from contextlib import ExitStack as does_not_raise  # noqa: N813

import numpy as np
import pandas as pd
import pytest
from sid.config import INITIAL_CONDITIONS
from sid.initial_conditions import _parse_initial_conditions
from sid.initial_conditions import _scale_up_initial_infections
from sid.initial_conditions import _scale_up_initial_infections_numba
from sid.initial_conditions import _spread_out_initial_infections
from sid.initial_conditions import create_initial_infections
from sid.initial_conditions import scale_and_spread_initial_infections
from sid.pathogenesis import draw_course_of_disease
from sid.simulate import _process_initial_states


@pytest.mark.unit
@pytest.mark.parametrize(
    ("initial_conditions", "expected"),
    [
        (None, INITIAL_CONDITIONS),
        (
            {"assort_by": ["region"]},
            {**INITIAL_CONDITIONS, **{"assort_by": ["region"]}},
        ),
        (
            {"assort_by": "region"},
            {**INITIAL_CONDITIONS, **{"assort_by": ["region"]}},
        ),
    ],
)
def test_parse_initial_conditions(initial_conditions, expected):
    result = _parse_initial_conditions(initial_conditions)
    assert result == expected


@pytest.mark.unit
def test_scale_up_initial_infections_without_assort_by():
    states = pd.DataFrame(index=pd.RangeIndex(100_000))

    initial_infections = create_initial_infections(0.1, 100_000)

    scaled_up_infections = _scale_up_initial_infections(
        initial_infections, states, None, 1.3, 0
    )

    assert np.isclose(scaled_up_infections.mean(), 0.13, atol=0.001)


@pytest.mark.unit
def test_scale_up_initial_infections_with_assort_by_equal_groups():
    states = pd.DataFrame(
        {"region": np.random.choice(["North", "East", "South", "West"], size=100_000)}
    )

    initial_infections = create_initial_infections(0.1, 100_000)

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

    initial_infections = create_initial_infections(0.1, 100_000)

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
    infections = create_initial_infections(0.2, 100_000)

    spread_infections = _spread_out_initial_infections(infections, 4, 2, 0)

    infections_per_day = spread_infections.sum()
    shares_per_day = infections_per_day / infections_per_day.sum()
    assert np.allclose(shares_per_day, np.array([1, 2, 4, 8]) / 15, atol=0.01)


@pytest.mark.unit
def test_spread_out_initial_infections_no_growth():
    infections = create_initial_infections(0.2, 100_000)

    spread_infections = _spread_out_initial_infections(infections, 4, 1, 0)

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
        out = create_initial_infections(infections, n_people, index, seed)

        if callable(expected):
            assert expected(out)
        else:
            assert out == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "initial_conditions, seed, expectation, expected",
    [
        (
            {"initial_infections": None},
            itertools.count(),
            pytest.raises(ValueError, match="'initial_infections' must"),
            None,
        ),
        (
            {"initial_infections": 1},
            itertools.count(-1),
            pytest.raises(ValueError, match="Seed"),
            None,
        ),
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
            {"growth_rate": 0},
            itertools.count(),
            pytest.raises(ValueError, match="'growth_rate' must be greater than or"),
            None,
        ),
        (
            {"burn_in_periods": 0},
            itertools.count(),
            pytest.raises(ValueError, match="'burn_in_periods' must be an integer"),
            None,
        ),
        (
            {"burn_in_periods": 2.0},
            itertools.count(),
            pytest.raises(ValueError, match="'burn_in_periods' must be an integer"),
            None,
        ),
        (
            {
                "initial_infections": pd.DataFrame(
                    {0: [True] * 8 + [False] * 7, 1: [False] * 8 + [True] * 7}
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
        initial_states = _process_initial_states(initial_states, {"a": []})
        initial_states = draw_course_of_disease(initial_states, params, 0)

        result = scale_and_spread_initial_infections(
            initial_states, params, initial_conditions, seed
        )
        assert result["ever_infected"].equals(expected)
