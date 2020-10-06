import numpy as np
import pandas as pd
import pytest
from sid.config import INDEX_NAMES
from sid.simulate import _prepare_params
from sid.simulate import _process_assort_bys
from sid.simulate import _process_initial_states
from sid.simulate import simulate


def meet_two(states, params):  # noqa: U100
    return pd.Series(index=states.index, data=2)


CONTACT_MODELS = {
    "standard": {
        "model": meet_two,
        "assort_by": ["age_group", "region"],
        "is_recurrent": False,
    }
}


def test_simple_run(params, initial_states, tmpdir):
    initial_infections = pd.Series(index=initial_states.index, data=False)
    initial_infections.iloc[0] = True

    df = simulate(
        params,
        initial_states,
        initial_infections,
        CONTACT_MODELS,
        path=tmpdir,
    )
    df = df.compute()

    assert isinstance(df, pd.DataFrame)


def test_check_assort_by_are_categoricals(initial_states):
    assort_bys = _process_assort_bys(CONTACT_MODELS)

    _ = _process_initial_states(initial_states, assort_bys)

    initial_states = initial_states.astype(str)
    with pytest.raises(TypeError):
        _process_initial_states(initial_states, assort_bys)


@pytest.mark.unit
def test_prepare_params(params):
    index = pd.MultiIndex.from_tuples([("a", "b", np.nan)], names=INDEX_NAMES)
    s = pd.DataFrame(index=index, data={"value": 0, "note": None, "source": None})
    params = params.copy().append(s)

    with pytest.raises(ValueError):
        _prepare_params(params)


@pytest.mark.unit
@pytest.mark.parametrize("input_", [pd.Series(dtype="object"), (), 1, [], {}, set()])
def test_prepare_params_with_wrong_inputs(input_):
    with pytest.raises(ValueError):
        _prepare_params(input_)


@pytest.mark.unit
def test_prepare_params_not_three_dimensional_multi_index(params):
    params = params.copy().reset_index(drop=True)

    with pytest.raises(ValueError):
        _prepare_params(params)


@pytest.mark.unit
def test_prepare_params_no_duplicates_in_index(params):
    params = params.copy().append(params)

    with pytest.raises(ValueError):
        _prepare_params(params)
