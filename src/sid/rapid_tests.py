import itertools
from typing import Callable
from typing import Optional

import pandas as pd
from sid.validation import validate_return_is_series_or_ndarray


def perform_rapid_tests(
    date: pd.Timestamp,
    states: pd.DataFrame,
    params: pd.DataFrame,
    rapid_tests_models: Optional[Callable],
    seed: itertools.count,
) -> pd.DataFrame:
    if rapid_tests_models:
        receives_rapid_test = _compute_who_receives_rapid_tests(
            date, states, params, rapid_tests_models, seed
        )

        is_tested_positive = _sample_test_outcome(
            states, receives_rapid_test, params, seed
        )

        states = _update_states_with_rapid_tests_outcomes(
            states, receives_rapid_test, is_tested_positive
        )

    return states


def _compute_who_receives_rapid_tests(date, states, params, rapid_tests_models, seed):
    receives_rapid_test = pd.Series(index=states.index, data=False)

    for model in rapid_tests_models.values():
        loc = model.get("loc", params.index)
        func = model["model"]

        if model["start"] <= date <= model["end"]:
            new_receives_rapid_test = func(
                receives_rapid_test=receives_rapid_test.copy(deep=True),
                states=states,
                params=params.loc[loc],
                seed=next(seed),
            )

            new_receives_rapid_test = validate_return_is_series_or_ndarray(
                new_receives_rapid_test, states.index, "rapid_tests_model"
            )

            receives_rapid_test.loc[new_receives_rapid_test] = True

    return receives_rapid_test


def _sample_test_outcome(states, receives_rapid_test, params, seed):  # noqa: U100
    return states


def _update_states_with_rapid_tests_outcomes(
    states, receives_rapid_test, is_tested_positive
):
    states[receives_rapid_test, "cd_received_rapid_test"] = 0
    states[is_tested_positive, "is_tested_positive_by_rapid_test"] = True

    return states
