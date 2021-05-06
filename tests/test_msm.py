from contextlib import ExitStack as does_not_raise  # noqa: N813

import numpy as np
import pandas as pd
import pytest
from sid.msm import _flatten_index
from sid.msm import _harmonize_input
from sid.msm import _is_diagonal
from sid.msm import get_diag_weighting_matrix
from sid.msm import get_flat_moments
from sid.msm import get_msm_func


def dummy_simulate(_params):  # noqa: U101
    return pd.Series([1, 2])


def dummy_calc_moments(df):
    return df


@pytest.mark.end_to_end
def test_estimation_with_msm():
    s = pd.Series([1, 2])
    msm_func = get_msm_func(dummy_simulate, dummy_calc_moments, s, lambda x: x)

    result = msm_func(None)
    expected = {
        "value": 0,
        "root_contributions": pd.Series([0.0, 0.0], ["0_0", "0_1"]),
        "empirical_moments": {0: s},
        "simulated_moments": {0: s},
    }

    for k, v in result.items():
        if k == "value":
            assert v == expected[k]
        else:
            if isinstance(v, dict):
                for kk, vv in v.items():
                    vv.equals(expected[k][kk])
            else:
                v.equals(expected[k])


@pytest.mark.integration
@pytest.mark.parametrize(
    "empirical_moments, weights, expected",
    [({"a": pd.Series([1]), "b": pd.Series([2])}, None, np.eye(2))],
)
def test_get_diag_weighting_matrix(empirical_moments, weights, expected):
    result = get_diag_weighting_matrix(empirical_moments, weights)
    assert np.all(result == expected)


def test_get_diag_weighting_matrix_with_scalar_weights():
    emp_moms = {0: pd.Series([1, 2]), 1: pd.Series([2, 3, 4])}
    weights = {0: 0.3, 1: 0.7}
    result = get_diag_weighting_matrix(emp_moms, weights)
    expected = np.diag([0.3] * 2 + [0.7] * 3)
    assert np.all(result == expected)


@pytest.mark.integration
@pytest.mark.parametrize(
    "moments, expected",
    [
        ({0: pd.Series([1]), 1: pd.Series([2])}, pd.Series([1, 2], ["0_0", "1_0"])),
        (
            {"a": pd.DataFrame([[1, 2]], columns=["b", "c"])},
            pd.Series([1, 2], ["a_b_0", "a_c_0"]),
        ),
    ],
)
def test_get_flat_moments(moments, expected):
    result = get_flat_moments(moments)
    assert result.equals(expected)


def _func():  # pragma: no cover
    pass


@pytest.mark.unit
@pytest.mark.parametrize(
    "data, expectation, expected",
    [
        (pd.Series([1]), does_not_raise(), {0: pd.Series([1])}),
        (pd.DataFrame([[1]]), does_not_raise(), {0: pd.DataFrame([[1]])}),
        (_func, does_not_raise(), {0: _func}),
        ({1: 2}, does_not_raise(), {1: 2}),
        ({1, 2}, pytest.raises(ValueError, match="Moments must be"), None),
    ],
)
def test_harmonize_input(data, expectation, expected):
    with expectation:
        result = _harmonize_input(data)
        for k, v in result.items():
            if isinstance(v, (pd.Series, pd.DataFrame)):
                assert v.equals(expected[k])
            else:
                assert result == expected


@pytest.mark.unit
def test_flatten_index():
    data = {
        "a": pd.Series(data=[1]),
        "b": pd.Series(index=["b"], data=[2]),
        "c": pd.DataFrame({"c": [3]}),
        "d": pd.Series(data=[4], name="e", index=[4]),
    }
    result = _flatten_index(data)
    expected = pd.Series(index=["a_0", "b_b", "c_c_0", "d_4"], data=[1, 2, 3, 4])
    assert result.equals(expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    "mat, expected", [(np.arange(4).reshape(2, 2), False), (np.eye(2), True)]
)
def test_is_diagonal(mat, expected):
    result = _is_diagonal(mat)
    assert result == expected
