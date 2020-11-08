import warnings

import pandas as pd
from sid.config import RELATIVE_POPULATION_PARAMETER
from sid.shared import date_is_within_start_and_end_date
from sid.shared import validate_return_is_series_or_ndarray


def process_tests(states, testing_processing_models, params, date):
    """Process tests which have been taken by individuals and are pending.

    In ``states`` there is a column called ``"pending_test"`` which is ``True`` for
    individuals which took as test which has not been processed, yet. Processing means
    that a countdown starts at which end the individual receives the test result.

    Args:
        states (pandas.DataFrame): The states of all individuals.
        testing_processing_models (dict): A dictionary containing the demand models for
            testing.
        params (pandas.DataFrame): The parameter DataFrame.
        date (pandas.Timestamp): Current date.

    Returns:
        all_to_be_processed_tests (pandas.Series): A boolean series indicating which
            tests have been chose for processing.

    """
    all_to_be_processed_tests = pd.Series(index=states.index, data=False)

    for model in testing_processing_models.values():
        loc = model.get("loc", params.index)
        func = model["model"]

        if date_is_within_start_and_end_date(
            date, model.get("start"), model.get("end")
        ):
            to_be_processed_tests = func(
                all_to_be_processed_tests.sum(), states, params.loc[loc]
            )

            to_be_processed_tests = validate_return_is_series_or_ndarray(
                to_be_processed_tests, states.index, "testing_allocation_models"
            )

            if not states["pending_test"][to_be_processed_tests].all():
                warnings.warn(
                    "A test was processed, but the individual had no pending test."
                )

            # Set the pending test of individuals who received a test to ``False``.
            states.loc[to_be_processed_tests, "pending_test"] = False
            # Update series with all to_be_processed tests.
            all_to_be_processed_tests.loc[to_be_processed_tests] = True

    n_available_tests = round(
        params.loc[("testing", "processing", "rel_available_capacity"), "value"]
        * len(states)
        * RELATIVE_POPULATION_PARAMETER
    )

    if n_available_tests < all_to_be_processed_tests.sum():
        warnings.warn(
            "The test processing models started processing more tests than possible."
        )

    return all_to_be_processed_tests
