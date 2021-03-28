import itertools
from contextlib import ExitStack as does_not_raise  # noqa: N813

import pandas as pd
import pytest
from sid.vaccination import vaccinate_individuals


@pytest.mark.integration
@pytest.mark.parametrize(
    "vaccination_models, expectation, expected",
    [
        ({}, does_not_raise(), pd.Series([False] * 15)),
        (
            {
                "vaccine": {
                    "model": lambda receives_vaccine, states, params, seed: pd.Series(
                        index=states.index, data=True
                    ),
                    "start": pd.Timestamp("2020-03-01"),
                    "end": pd.Timestamp("2020-03-04"),
                }
            },
            does_not_raise(),
            pd.Series([True] * 15),
        ),
        (
            {
                "vaccine": {
                    "model": lambda receives_vaccine, states, params, seed: None,
                    "start": pd.Timestamp("2020-03-01"),
                    "end": pd.Timestamp("2020-03-04"),
                }
            },
            pytest.raises(ValueError, match="The model 'vaccine' of 'vaccination_mode"),
            None,
        ),
    ],
)
def test_vaccinate_individuals(
    vaccination_models, initial_states, params, expectation, expected
):
    with expectation:
        result = vaccinate_individuals(
            pd.Timestamp("2020-03-03"),
            vaccination_models,
            initial_states,
            params,
            itertools.count(),
        )

        assert result.equals(expected)
