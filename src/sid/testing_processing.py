import itertools
import warnings
from typing import Any
from typing import Dict

import pandas as pd
from sid.config import RELATIVE_POPULATION_PARAMETER
from sid.validation import validate_return_is_series_or_ndarray


def process_tests(
    states: pd.DataFrame,
    testing_processing_models: Dict[str, Dict[str, Any]],
    params: pd.DataFrame,
    date: pd.Timestamp,
    seed: itertools.count,
) -> pd.Series:
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
        seed (itertools.count): The seed counter.

    Returns:
        all_to_be_processed_tests (pandas.Series): A boolean series indicating which
            tests have been chose for processing.

    """
    all_to_be_processed_tests = pd.Series(index=states.index, data=False)

    for name, model in testing_processing_models.items():
        loc = model.get("loc", params.index)
        func = model["model"]

        if model["start"] <= date <= model["end"]:
            to_be_processed_tests = func(
                n_to_be_processed_tests=all_to_be_processed_tests.sum(),
                states=states,
                params=params.loc[loc],
                seed=next(seed),
            )

            to_be_processed_tests = validate_return_is_series_or_ndarray(
                to_be_processed_tests, name, "testing_allocation_models", states.index
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
