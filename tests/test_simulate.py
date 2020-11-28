import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_categorical_dtype
from sid.config import INDEX_NAMES
from sid.simulate import _prepare_params
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


def test_simulate_a_simple_model(params, initial_states, tmp_path):
    initial_infections = pd.Series(index=initial_states.index, data=False)
    initial_infections.iloc[:2] = True

    simulate = get_simulate_func(
        params,
        initial_states,
        initial_infections,
        CONTACT_MODELS,
        saved_columns={"other": ["was_infected_by_contact"]},
        path=tmp_path,
    )

    df = simulate(params)

    df = df.compute()

    assert isinstance(df, pd.DataFrame)
    assert set(df["was_infected_by_contact"].cat.categories) == set(
        ["not_infected_by_contact", "standard"]
    )


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
        _prepare_params(params)


@pytest.mark.unit
@pytest.mark.parametrize("input_", [pd.Series(dtype="object"), (), 1, [], {}, set()])
def test_prepare_params_with_wrong_inputs(input_):
    with pytest.raises(ValueError, match="params must be a DataFrame."):
        _prepare_params(input_)


@pytest.mark.unit
def test_prepare_params_not_three_dimensional_multi_index(params):
    params = params.copy().reset_index(drop=True)

    with pytest.raises(ValueError, match="params must have the index levels"):
        _prepare_params(params)


@pytest.mark.unit
def test_prepare_params_no_duplicates_in_index(params):
    params = params.copy().append(params)

    with pytest.raises(ValueError, match="No duplicates in the params index allowed."):
        _prepare_params(params)


@pytest.mark.unit
def test_prepare_params_value_with_nan(params):
    params = params.copy()
    params["value"].iloc[0] = np.nan

    with pytest.raises(ValueError, match="The 'value' column of params must not"):
        _prepare_params(params)
