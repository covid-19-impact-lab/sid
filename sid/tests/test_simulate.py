import pandas as pd
import pytest

from sid.simulate import _process_assort_bys
from sid.simulate import _process_initial_states
from sid.simulate import simulate


def meet_two(states, params, period):  # noqa: U100
    return pd.Series(index=states.index, data=2)


CONTACT_MODELS = {"standard": {"model": meet_two, "assort_by": ["age_group", "region"]}}


def test_simple_run(params, initial_states, tmpdir):
    initial_infections = pd.Series(index=initial_states.index, data=False)
    initial_infections.iloc[0] = True

    df = simulate(
        params, initial_states, initial_infections, CONTACT_MODELS, path=tmpdir,
    )
    df = df.compute()

    assert isinstance(df, pd.DataFrame)


def test_check_assort_by_are_categoricals(initial_states):
    assort_bys = _process_assort_bys(CONTACT_MODELS)

    _ = _process_initial_states(initial_states, assort_bys)

    initial_states = initial_states.astype(str)
    with pytest.raises(TypeError):
        _process_initial_states(initial_states, assort_bys)
