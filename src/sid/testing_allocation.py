import warnings

import pandas as pd
from sid.config import RELATIVE_POPULATION_PARAMETER
from sid.shared import date_is_within_start_and_end_date
from sid.shared import validate_return_is_series_or_ndarray


def allocate_tests(states, testing_allocation_models, demands_test, params, date):
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
        date (pandas.Timestamp): Current date.

    Returns:
        all_allocated_tests (pandas.Series): A boolean series indicating which
            individuals received a test.

    """
    all_allocated_tests = pd.Series(index=states.index, data=False)
    current_demands_test = demands_test.copy()

    for model in testing_allocation_models.values():
        loc = model.get("loc", params.index)
        func = model["model"]

        if date_is_within_start_and_end_date(
            date, model.get("start"), model.get("end")
        ):
            allocated_tests = func(
                all_allocated_tests.sum(), current_demands_test, states, params.loc[loc]
            )
            allocated_tests = validate_return_is_series_or_ndarray(
                allocated_tests, states.index, "testing_allocation_models"
            )

            if not current_demands_test[allocated_tests].all():
                warnings.warn(
                    "A test was allocated to an individual which did not request one."
                )

            # Set the demand of individuals who received a test to ``False``.
            current_demands_test.loc[allocated_tests] = False
            # Update series with all allocated tests.
            all_allocated_tests.loc[allocated_tests] = True

    n_available_tests = int(
        params.loc[("testing", "allocation", "rel_available_tests"), "value"]
        * len(states)
        * RELATIVE_POPULATION_PARAMETER
    )

    if n_available_tests < all_allocated_tests.sum():
        warnings.warn(
            "The test allocation models distributed more tests than available."
        )

    return all_allocated_tests


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
