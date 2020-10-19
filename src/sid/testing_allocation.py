import warnings

import pandas as pd
from sid.shared import validate_return_is_series_or_ndarray


def allocate_tests(states, testing_allocation_models, demands_test, params):
    """Allocate tests to people who demand one.

    The function iterates over all test allocation models and each model is able to see
    how many tests have already been allotted by the previous models.

    Args:
        states (pandas.DataFrame): The states of all individuals.
        testing_demand_models (dict): A dictionary containing the demand models for
            testing.
        demands_test (pandas.Series): A boolean series indicating which person demands a
              test.
        params (pandas.DataFrame): The parameter DataFrame.

    Returns:
        allocated_tests (pandas.Series): A boolean series indicating which individuals
            received a test.

    """
    allocated_tests = pd.Series(index=states.index, data=False)

    for model in testing_allocation_models.values():
        loc = model.get("loc", params.index)
        func = model["model"]

        allocated_tests = func(allocated_tests, demands_test, states, params.loc[loc])

    allocated_tests = validate_return_is_series_or_ndarray(
        allocated_tests, states.index, "testing_allocation_models"
    )

    n_available_tests = params.loc[
        ("testing", "allocation", "available_tests"), "value"
    ]

    if n_available_tests < allocated_tests.sum():
        warnings.warn(
            "The test allocation models distributed more tests than available."
        )

    return allocated_tests


def update_pending_tests(states, allocated_tests):
    """Update information regarding pending tests."""
    states.loc[allocated_tests, "pending_test"] = True
    states.loc[allocated_tests, "pending_test_date"] = states.loc[
        allocated_tests, "date"
    ]
    states.loc[allocated_tests, "pending_test_period"] = states.loc[
        allocated_tests, "period"
    ]
    return states
