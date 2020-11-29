import pytest
from sid.initial_conditions import (
    _parse_initial_conditions,
    _scale_up_initial_infections,
    _scale_up_initial_infections_numba,
    _spread_out_initial_infections,
)
from sid.config import INITIAL_CONDITIONS, INDEX_NAMES
import pandas as pd
import numpy as np
import itertools


def _create_initial_infections(n_people, n_infections):
    infected_indices = np.random.choice(100_000, size=10_000, replace=False)
    initial_infections = pd.Series(index=pd.RangeIndex(100_000), data=False)
    initial_infections.iloc[infected_indices] = True
    return initial_infections


@pytest.mark.unit
@pytest.mark.parametrize(
    "initial_conditions, expected",
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

    initial_infections = _create_initial_infections(100_000, 10_000)

    scaled_up_infections = _scale_up_initial_infections(
        initial_infections, states, None, 1.3, itertools.count()
    )

    assert np.isclose(scaled_up_infections.mean(), 0.13, atol=0.001)


@pytest.mark.unit
def test_scale_up_initial_infections_with_assort_by_equal_groups():
    states = pd.DataFrame(
        {"region": np.random.choice(["North", "East", "South", "West"], size=100_000)}
    )

    initial_infections = _create_initial_infections(100_000, 10_000)

    scaled_up_infections = _scale_up_initial_infections(
        initial_infections, states, None, 1.3, itertools.count()
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

    initial_infections = _create_initial_infections(100_000, 10_000)

    scaled_up_infections = _scale_up_initial_infections(
        initial_infections, states, ["region"], 1.3, itertools.count()
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
    infections = _create_initial_infections(100_000, 20_000)

    spread_infections = _spread_out_initial_infections(
        infections, 4, 2, itertools.count()
    )

    infections_per_day = np.sum(spread_infections, axis=1)
    shares_per_day = infections_per_day / infections_per_day.sum()
    assert np.allclose(shares_per_day, np.array([1, 2, 4, 8]) / 15, atol=0.01)
