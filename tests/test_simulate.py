from contextlib import ExitStack as does_not_raise  # noqa: N813

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_categorical_dtype
from resources import CONTACT_MODELS
from sid.config import INDEX_NAMES
from sid.simulate import _add_default_duration_to_models
from sid.simulate import _create_group_codes_names
from sid.simulate import _process_assort_bys
from sid.simulate import _process_initial_states
from sid.simulate import get_simulate_func
from sid.validation import validate_params


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


def test_check_assort_by_are_categoricals(initial_states):
    assort_bys = _process_assort_bys(CONTACT_MODELS)
    group_codes_names = _create_group_codes_names(CONTACT_MODELS, assort_bys)

    _ = _process_initial_states(
        initial_states, assort_bys, group_codes_names, CONTACT_MODELS
    )

    initial_states = initial_states.astype(str)
    processed = _process_initial_states(
        initial_states, assort_bys, group_codes_names, CONTACT_MODELS
    )
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
