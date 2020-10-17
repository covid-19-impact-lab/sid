from contextlib import ExitStack as does_not_warn_or_raise  # noqa: N813

import numpy as np
import pandas as pd
import pytest
from sid.testing_allocation import allocate_tests


@pytest.mark.integration
@pytest.mark.parametrize(
    "excess, expectation",
    [(True, pytest.warns(UserWarning)), (False, does_not_warn_or_raise())],
)
def test_issue_warning_if_allocated_tests_exceed_available_tests(
    initial_states, params, excess, expectation
):
    testing_allocation_models = {
        "all": {"model": lambda *x: pd.Series(data=True, index=initial_states.index)}
    }
    demands_test = pd.Series(index=initial_states.index, data=True)
    params.loc[("testing", "allocation", "available_tests"), "value"] = (
        len(initial_states) - 1 if excess else len(initial_states)
    )

    with expectation:
        allocated_tests = allocate_tests(
            initial_states, testing_allocation_models, demands_test, params
        )

    assert allocated_tests.all()


@pytest.mark.integration
@pytest.mark.parametrize(
    "return_, expectation",
    [
        (pd.Series(data=np.full(15, True)), does_not_warn_or_raise()),
        (np.full(15, True), does_not_warn_or_raise()),
        (1, pytest.raises(ValueError)),
        ([True] * 15, pytest.raises(ValueError)),
    ],
)
def test_raise_error_if_allocated_tests_have_invalid_return(
    initial_states, params, return_, expectation
):
    testing_allocation_models = {"all": {"model": lambda *x: return_}}
    demands_test = pd.Series(index=initial_states.index, data=True)
    params.loc[("testing", "allocation", "available_tests"), "value"] = len(
        initial_states
    )

    with expectation:
        allocated_tests = allocate_tests(
            initial_states, testing_allocation_models, demands_test, params
        )
        assert allocated_tests.all()
