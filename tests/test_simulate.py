import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_categorical_dtype
from sid.config import INDEX_NAMES
from sid.preparation import prepare_initial_states
from sid.preparation import validate_params
from sid.simulate import _process_assort_bys
from sid.simulate import _process_initial_states
from sid.simulate import get_simulate_func


def meet_two(states, params):  # noqa: U100
    return pd.Series(index=states.index, data=2)


CONTACT_MODELS = {
    "standard": {
        "model": meet_two,
        "assort_by": ["age_group", "region"],
        "is_recurrent": False,
    }
}


def test_simple_run(params, initial_states, tmp_path):
    initial_infections = pd.Series(index=initial_states.index, data=False)
    initial_infections.iloc[0] = True

    states = prepare_initial_states(initial_states, initial_infections, params)

    simulate = get_simulate_func(
        params,
        states,
        CONTACT_MODELS,
        path=tmp_path,
    )

    df = simulate(params)

    df = df.compute()

    assert isinstance(df, pd.DataFrame)


def test_check_assort_by_are_categoricals(initial_states):
    assort_bys = _process_assort_bys(CONTACT_MODELS)

    _ = _process_initial_states(initial_states, assort_bys)

    initial_states = initial_states.astype(str)
    processed = _process_initial_states(initial_states, assort_bys)
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
