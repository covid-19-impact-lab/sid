import warnings

import pandas as pd


def process_tests(states, testing_processing_models, params):
    """Process tests which have been taken by individuals and are pending.

    In ``states`` there is a column called ``"pending_tests"`` which is ``True`` for
    individuals which took as test which has not been processed, yet. Processing means
    that a countdown starts at which end the individual receives the test result.

    Args:
        states (pandas.DataFrame): The states of all individuals.
        testing_processing_models (dict): A dictionary containing the demand models for
            testing.
        params (pandas.DataFrame): The parameter DataFrame.

    Returns:
        to_be_processed_tests (pandas.Series): A boolean series indicating which tests
            have been chose for processing.

    """
    to_be_processed_tests = pd.Series(index=states.index, data=False)

    for model in testing_processing_models.values():
        loc = model.get("loc", params.index)
        func = model["model"]

        to_be_processed_tests = func(to_be_processed_tests, states, params.loc[loc])

    n_available_tests = params.loc[
        ("testing", "processing", "available_capacity"), "value"
    ]

    if n_available_tests < to_be_processed_tests.sum():
        warnings.warn("The test processing models processed more tests than available.")

    return to_be_processed_tests
