import itertools
from contextlib import ExitStack as does_not_raise  # noqa: N813

import pandas as pd
import pytest
from sid.vaccination import vaccinate_individuals


@pytest.mark.integration
@pytest.mark.parametrize(
    "vaccination_model, expectation, expected",
    [
        (None, does_not_raise(), None),
        (
            lambda states, params, seed: pd.Series(index=states.index, data=True),
            does_not_raise(),
            pd.Series([True] * 15),
        ),
        (
            lambda states, params, seed: None,
            pytest.raises(ValueError, match="'vaccination_model' must always return"),
            None,
        ),
    ],
)
def test_vaccinate_individuals(
    vaccination_model, initial_states, params, expectation, expected
):
    with expectation:
        result = vaccinate_individuals(
            vaccination_model, initial_states, params, itertools.count()
        )
        if expected is None:
            assert result is expected
        else:
            assert result.equals(expected)
