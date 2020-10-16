from contextlib import ExitStack as does_not_warn_or_raise  # noqa: N813

import numpy as np
import pandas as pd
import pytest
from sid.testing_processing import process_tests


@pytest.mark.parametrize(
    "excess, expectation",
    [(True, pytest.warns(UserWarning)), (False, does_not_warn_or_raise())],
)
def test_issue_warning_if_processed_tests_exceed_available_tests(
    initial_states, params, excess, expectation
):
    testing_processing_models = {
        "all": {"model": lambda *x: pd.Series(data=True, index=initial_states.index)}
    }
    initial_states["pending_test"] = True

    params.loc[("testing", "processing", "available_capacity"), "value"] = (
        len(initial_states) - 1 if excess else len(initial_states)
    )

    with expectation:
        to_be_processed_tests = process_tests(
            initial_states, testing_processing_models, params
        )

    assert to_be_processed_tests.all()


@pytest.mark.parametrize(
    "return_, expectation",
    [
        (pd.Series(data=[True] * 15), does_not_warn_or_raise()),
        (np.full(15, True), does_not_warn_or_raise()),
        (1, pytest.raises(ValueError)),
        ([True] * 15, pytest.raises(ValueError)),
    ],
)
def test_raise_error_if_processed_tests_have_invalid_return(
    initial_states, params, return_, expectation
):
    testing_processing_models = {"all": {"model": lambda *x: return_}}
    params.loc[("testing", "processing", "available_capacity"), "value"] = len(
        initial_states
    )

    with expectation:
        to_be_processed_tests = process_tests(
            initial_states, testing_processing_models, params
        )
        assert to_be_processed_tests.all()
