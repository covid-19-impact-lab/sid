import math
from contextlib import ExitStack as does_not_raise  # noqa: N813

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_categorical_dtype
from resources import CONTACT_MODELS
from sid.config import INDEX_NAMES
from sid.simulate import _add_default_duration_to_models
from sid.simulate import _create_group_codes_and_info
from sid.simulate import _create_group_codes_names
from sid.simulate import _prepare_susceptibility_factor
from sid.simulate import _process_assort_bys
from sid.simulate import _process_initial_states
from sid.simulate import get_simulate_func
from sid.validation import validate_params


@pytest.mark.end_to_end
def test_simulate_a_simple_model(params, initial_states, tmp_path):
    simulate = get_simulate_func(
        params=params,
        initial_states=initial_states,
        contact_models=CONTACT_MODELS,
        saved_columns={"other": ["channel_infected_by_contact"]},
        path=tmp_path,
        seed=144,
    )

    result = simulate(params)

    time_series = result["time_series"].compute()
    last_states = result["last_states"].compute()

    for df in [time_series, last_states]:
        assert isinstance(df, pd.DataFrame)
        assert set(df["channel_infected_by_contact"].cat.categories) == {
            "not_infected_by_contact",
            "standard",
        }


@pytest.mark.end_to_end
def test_resume_a_simulation(params, initial_states, tmp_path):
    simulate = get_simulate_func(
        params=params,
        initial_states=initial_states,
        contact_models=CONTACT_MODELS,
        saved_columns={"other": ["channel_infected_by_contact"]},
        path=tmp_path,
        seed=144,
    )

    result = simulate(params)

    time_series = result["time_series"].compute()
    last_states = result["last_states"].compute()

    for df in [time_series, last_states]:
        assert isinstance(df, pd.DataFrame)
        assert set(df["channel_infected_by_contact"].cat.categories) == {
            "not_infected_by_contact",
            "standard",
        }

    resumed_simulate = get_simulate_func(
        params=params,
        initial_states=last_states,
        contact_models=CONTACT_MODELS,
        saved_columns={"other": ["channel_infected_by_contact"]},
        duration={"start": "2020-02-06", "periods": 5},
        path=tmp_path,
        seed=144,
    )

    resumed_result = resumed_simulate(params)

    resumed_time_series = resumed_result["time_series"].compute()
    resumed_last_states = resumed_result["last_states"].compute()

    for df in [resumed_time_series, resumed_last_states]:
        assert isinstance(df, pd.DataFrame)
        assert set(df["channel_infected_by_contact"].cat.categories) == {
            "not_infected_by_contact",
            "standard",
        }


@pytest.mark.unit
def test_check_assort_by_are_categoricals(initial_states):
    assort_bys = _process_assort_bys(CONTACT_MODELS)

    virus_strains = {"names": ["base_strain"], "factors": np.ones(1)}

    _ = _process_initial_states(initial_states, assort_bys, virus_strains)

    initial_states = initial_states.astype(str)
    processed = _process_initial_states(initial_states, assort_bys, virus_strains)
    for var in ["age_group", "region"]:
        assert is_categorical_dtype(processed[var].dtype)


@pytest.mark.unit
def test_prepare_params(params):
    index = pd.MultiIndex.from_tuples([("a", "b", np.nan)], names=INDEX_NAMES)
    s = pd.DataFrame(index=index, data={"value": 0, "note": None, "source": None})
    params = params.copy().append(s)

    with pytest.raises(ValueError, match="No NaNs allowed in the params index."):
        validate_params(params)


@pytest.mark.unit
@pytest.mark.parametrize("input_", [pd.Series(dtype="object"), (), 1, [], {}, set()])
def test_prepare_params_with_wrong_inputs(input_):
    with pytest.raises(ValueError, match="params must be a DataFrame."):
        validate_params(input_)


@pytest.mark.unit
def test_prepare_params_not_three_dimensional_multi_index(params):
    params = params.copy().reset_index(drop=True)

    with pytest.raises(ValueError, match="params must have the index levels"):
        validate_params(params)


@pytest.mark.unit
def test_prepare_params_no_duplicates_in_index(params):
    params = params.copy().append(params)

    with pytest.raises(ValueError, match="No duplicates in the params index allowed."):
        validate_params(params)


@pytest.mark.unit
def test_prepare_params_value_with_nan(params):
    params = params.copy()
    params["value"].iloc[0] = np.nan

    with pytest.raises(ValueError, match="The 'value' column of params must not"):
        validate_params(params)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dicts, duration, expectation, expected",
    [
        (
            {"a": {}},
            {"start": "2020-01-01", "end": "2020-01-02"},
            does_not_raise(),
            {
                "a": {
                    "start": pd.Timestamp("2020-01-01"),
                    "end": pd.Timestamp("2020-01-02"),
                }
            },
        ),
        (
            {"a": {"start": None}},
            {"start": "2020-01-01", "end": "2020-01-02"},
            pytest.raises(ValueError, match="The start date"),
            None,
        ),
        (
            {"a": {"end": None}},
            {"start": "2020-01-01", "end": "2020-01-02"},
            pytest.raises(ValueError, match="The end date"),
            None,
        ),
        (
            {"a": {"end": "2019-12-31"}},
            {"start": "2020-01-01", "end": "2020-01-02"},
            pytest.raises(ValueError, match="The end date of model"),
            None,
        ),
    ],
)
def test_add_default_duration_to_models(dicts, duration, expectation, expected):
    with expectation:
        result = _add_default_duration_to_models(dicts, duration)
        assert result == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "contact_models, assort_bys, expectation, expected",
    [
        ({"a": {}}, {"a": ["id"]}, does_not_raise(), {"a": "group_codes_a"}),
        ({"b": {"is_factorized": True}}, {"b": ["id"]}, does_not_raise(), {"b": "id"}),
        (
            {"c": {"is_factorized": False}},
            {"c": ["id_1", "id_2"]},
            does_not_raise(),
            {"c": "group_codes_c"},
        ),
        (
            {"d": {"is_factorized": True}},
            {"d": ["id_1", "id_2"]},
            pytest.raises(ValueError, match="'is_factorized'"),
            None,
        ),
    ],
)
def test_create_group_codes_names(contact_models, assort_bys, expectation, expected):
    with expectation:
        result = _create_group_codes_names(contact_models, assort_bys)
        assert result == expected


@pytest.mark.integration
@pytest.mark.parametrize(
    "states, assort_bys, contact_models, expected_states, expected_group_codes_info",
    [
        pytest.param(
            pd.DataFrame({"a": [2, 4, 1, 3, -1]}, dtype="category"),
            {"cm": ["a"]},
            {"cm": {"assort_by": ["a"], "is_recurrent": False}},
            pd.DataFrame(
                {
                    "a": pd.Series([2, 4, 1, 3, -1]).astype("category"),
                    "group_codes_cm": np.int32([1, 3, 0, 2, -1]),
                }
            ),
            {"cm": {"name": "group_codes_cm", "groups": [1, 2, 3, 4]}},
            id="test with random model",
        ),
        pytest.param(
            pd.DataFrame({"a": [2, 4, 1, 3, -1]}, dtype="category"),
            {"cm": ["a"]},
            {"cm": {"assort_by": ["a"], "is_recurrent": True}},
            pd.DataFrame(
                {
                    "a": pd.Series([2, 4, 1, 3, -1]).astype("category"),
                    "group_codes_cm": np.int32([1, 3, 0, 2, -1]),
                }
            ),
            {"cm": {"name": "group_codes_cm", "groups": [1, 2, 3, 4]}},
            id="test with recurrent model",
        ),
        pytest.param(
            pd.DataFrame(
                {"a": [2, 4, 1, 3, -1], "b": [1, 1, 2, 2, 1]}, dtype="category"
            ),
            {"cm": ["a", "b"]},
            {"cm": {"assort_by": ["a", "b"], "is_recurrent": False}},
            pd.DataFrame(
                {
                    "a": pd.Series([2, 4, 1, 3, -1]).astype("category"),
                    "b": pd.Series([1, 1, 2, 2, 1]).astype("category"),
                    "group_codes_cm": np.int32([2, 4, 1, 3, 0]),
                }
            ),
            {
                "cm": {
                    "name": "group_codes_cm",
                    "groups": [(-1, 1), (1, 2), (2, 1), (3, 2), (4, 1)],
                }
            },
            id="test random model with multiple variables.",
        ),
        pytest.param(
            pd.DataFrame({"a": [2, 4, 1, 3, -1]}),
            {"cm": ["a"]},
            {"cm": {"assort_by": ["a"], "is_recurrent": True, "is_factorized": True}},
            pd.DataFrame({"a": np.int32([2, 4, 1, 3, -1])}),
            {"cm": {"name": "a", "groups": [1, 2, 3, 4]}},
            id="test recurrent model with already factorized variable",
        ),
    ],
)
def test_create_group_codes_and_info(
    states, assort_bys, contact_models, expected_states, expected_group_codes_info
):
    states, group_codes_info = _create_group_codes_and_info(
        states, assort_bys, contact_models
    )
    assert states.equals(expected_states)
    for cm in group_codes_info:
        group_codes_info[cm]["groups"] = group_codes_info[cm]["groups"].tolist()
    assert group_codes_info == expected_group_codes_info


@pytest.mark.end_to_end
def test_skipping_factorization_of_assort_by_variable_works(
    tmp_path, initial_states, params
):
    """Test that it is possible to skip the factorization of assort_by variables."""
    contact_models = {
        "households": {
            "model": lambda states, params, seed: states["hh_id"] != -1,
            "is_recurrent": True,
            "assort_by": "hh_id",
            "is_factorized": True,
        }
    }

    initial_states["hh_id"] = pd.Series(
        np.repeat(np.arange(-1, math.ceil(len(initial_states) / 2)), 2)[
            : len(initial_states)
        ],
        dtype="category",
    )

    params.loc[("infection_prob", "households", "households"), "value"] = 0.1

    simulate = get_simulate_func(
        params=params,
        initial_states=initial_states,
        contact_models=contact_models,
        duration={"start": "2020-01-01", "periods": 2},
        saved_columns={"group_codes": True},
        path=tmp_path,
        seed=144,
    )

    result = simulate(params)

    time_series = result["time_series"].compute()
    last_states = result["last_states"].compute()

    assert "group_codes_households" not in time_series
    assert "group_codes_households" not in last_states


@pytest.mark.unit
@pytest.mark.parametrize(
    "model, states, expectation, expected",
    [
        (None, [0] * 100, does_not_raise(), np.ones(100)),
        (
            lambda *x: True,
            None,
            pytest.raises(ValueError, match="'susceptibility_factor_model"),
            None,
        ),
        (
            lambda *x: pd.Series([1]),
            [1, 1],
            pytest.raises(ValueError, match="The 'susceptibility_factor"),
            None,
        ),
        (lambda *x: pd.Series([1]), [1], does_not_raise(), [1]),
        (lambda *x: np.ones(1), [1], does_not_raise(), [1]),
        (lambda *x: np.array([1, 2]), [1, 1], does_not_raise(), [0.5, 1]),
    ],
)
def test_prepare_susceptibility_factor(model, states, expectation, expected):
    with expectation:
        result = _prepare_susceptibility_factor(model, states, None, 0)
        assert (result == expected).all()
