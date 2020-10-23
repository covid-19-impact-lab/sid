import itertools
from contextlib import ExitStack as does_not_warn_or_raise  # noqa: N813

import numpy as np
import pandas as pd
import pytest
from sid.testing_allocation import allocate_tests


@pytest.mark.integration
def test_allocation_w_multiple_models(initial_states, params):
    iterator = itertools.count(0)

    def allocation_model(n_allocated_tests, demands_test, states, params):
        iteration = next(iterator)

        if iteration == 0:
            assert demands_test.all()
            assert n_allocated_tests == 0

            out = np.where(states["age_group"] == "Over 50", True, False)

        elif iteration == 1:
            n_over_50 = len(states.query("age_group == 'Over 50'"))
            assert n_allocated_tests == n_over_50
            assert demands_test.sum() == len(demands_test) - n_over_50

            out = np.where(demands_test & states["region"].eq("a"), True, False)

        elif iteration == 2:
            n_over_50 = states["age_group"].eq("Over 50").sum()
            region_a_under_50 = (
                states["age_group"].eq("Under 50") & states["region"].eq("a")
            ).sum()
            assert n_allocated_tests == n_over_50 + region_a_under_50
            assert (
                demands_test.sum() == len(demands_test) - n_over_50 - region_a_under_50
            )
            out = pd.Series(index=states.index, data=False)

        return out

    testing_allocation_models = {
        "iteration_1": {"model": allocation_model},
        "iteration_2": {"model": allocation_model},
        "iteration_3": {"model": allocation_model},
    }
    demands_test = pd.Series(index=initial_states.index, data=True)
    params.loc[("testing", "allocation", "available_tests"), "value"] = len(
        initial_states
    )

    allocated_tests = allocate_tests(
        initial_states, testing_allocation_models, demands_test, params, "2020-01-01"
    )

    assert isinstance(allocated_tests, pd.Series)


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
            initial_states,
            testing_allocation_models,
            demands_test,
            params,
            "2020-01-01",
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
            initial_states,
            testing_allocation_models,
            demands_test,
            params,
            "2020-01-01",
        )
        assert allocated_tests.all()
